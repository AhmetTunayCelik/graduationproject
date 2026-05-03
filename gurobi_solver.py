"""
gurobi_solver.py
================

Exact solver for the Maximum Assignment Problem with Conflicts (MAX-APC)
using Gurobi. Given an instance (cost matrix, conflict list), it builds
a binary linear program and solves it to optimality.

The model uses:
    - Assignment constraints: sum_j x[i,j] = 1 for each row i,
                              sum_i x[i,j] = 1 for each column j.
    - Conflict constraints: x[i1,j1] + x[i2,j2] <= 1 for each conflict.
    - Objective: maximise sum c[i,j] * x[i,j].

Usage:
    python gurobi_solver.py --instance path/to/instance.json

Or programmatically:
    from gurobi_solver import solve_instance
    result = solve_instance(instance_dict)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

import apc_base as ab  # for loading instances and type definitions
from parameters import config


# Maps Gurobi numeric Status codes to readable names. Anything not listed
# is reported as "STATUS_<code>" so unexpected outcomes surface in logs
# instead of being silently bucketed as UNKNOWN.
_STATUS_NAMES = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.INPROGRESS: "INPROGRESS",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}


def _status_name(code: int) -> str:
    return _STATUS_NAMES.get(code, f"STATUS_{int(code)}")


def _make_first_feasible_callback(start_time: float):
    """Build a Gurobi callback that records wall-clock when the first MIP
    incumbent is found. The callback closes over a mutable dict so the
    timestamp survives back to the caller after optimize() returns.
    """
    state = {"first_feasible_time": None}

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            if state["first_feasible_time"] is None:
                state["first_feasible_time"] = time.time() - start_time

    return cb, state


def _validate_assignment(assignment: List[Tuple[int, int]], n: int) -> bool:
    """Cheap sanity check: assignment must be an exact n-permutation.
    Catches numerical breakdown that Gurobi's solver may report as success
    but produce a malformed solution for.
    """
    if assignment is None or len(assignment) != n:
        return False
    rows = {i for i, _ in assignment}
    cols = {j for _, j in assignment}
    return len(rows) == n and len(cols) == n


def _sanitised_gap(model: gp.Model, obj_val: Optional[float]) -> Optional[float]:
    """Return MIPGap only when it is meaningful.

    Gurobi's MIPGap = |ObjBound - ObjVal| / |ObjVal|. When |ObjVal| is
    near zero (e.g. a degenerate instance where Gurobi only proved a
    cost-0 feasible) this blows up to inf and poisons aggregate stats.
    """
    if obj_val is None or abs(obj_val) < 1e-9:
        return None
    try:
        gap = float(model.MIPGap)
    except (AttributeError, gp.GurobiError):
        return None
    if not math.isfinite(gap):
        return None
    return gap


def _safe_obj_bound(model: gp.Model) -> Optional[float]:
    """Read Gurobi's best dual bound (ObjBound). Not always defined
    (e.g. INFEASIBLE, model unsolved). Returns None when unavailable.
    """
    try:
        bound = float(model.ObjBound)
    except (AttributeError, gp.GurobiError):
        return None
    if not math.isfinite(bound):
        return None
    return bound


def solve_instance(
    instance: ab.Instance,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> Dict:
    """
    Solve a MAX-APC instance to optimality using Gurobi.

    Parameters
    ----------
    instance : dict
        Instance dictionary as produced by instance_generator.generate_instance()
        or loaded from JSON. Must contain 'n', 'cost_matrix', 'conflicts'.
    time_limit : float, optional
        Maximum allowed runtime in seconds. If None, no limit.
    verbose : bool
        If True, print Gurobi solver output.

    Returns
    -------
    dict
        Contains:
            'status' : str  (GRB.Status name; e.g., "OPTIMAL", "TIME_LIMIT",
                             "INFEASIBLE", "NUMERIC", "INTERRUPTED", ...)
            'objective' : float or None  (best feasible objective; None if no feasible found)
            'best_bound' : float or None (Gurobi dual bound, ObjBound)
            'assignment' : list of (i, j) or None
            'runtime' : float (wall-clock incl. model build + optimize)
            'gap' : float or None (sanitised MIPGap; None when |obj| ~ 0 or non-finite)
            'nodes_explored' : int
            'solutions_found' : int (model.SolCount)
            'first_feasible_time' : float or None (seconds from start to first incumbent)
            'assignment_valid' : bool (n-permutation sanity check; False signals numerical issue)
    """
    # Wall-clock starts at the top so model construction is included in runtime.
    # Heuristics' runtime_seconds includes their setup; this matches.
    start_time = time.time()

    n = instance["n"]
    cost = instance["cost_matrix"]
    conflicts = instance["conflicts"]

    # Build edge list from graph_edges if present; fall back to complete graph.
    # This excludes non-graph edges (cost = -1e15 sentinel) from Gurobi's model,
    # giving presolve a clean formulation with no poisoned coefficients.
    graph_edges_raw = instance.get("graph_edges")
    if graph_edges_raw is not None:
        edge_list = [tuple(e) for e in graph_edges_raw]
    else:
        edge_list = [(i, j) for i in range(n) for j in range(n)]

    model = gp.Model("MAX-APC")
    if not verbose:
        model.setParam("OutputFlag", config.GUROBI_OUTPUT_FLAG)
    # Pinned thread count -> fair runtime comparison vs single-threaded heuristics.
    model.setParam("Threads", config.GUROBI_THREADS)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    # Memory soft-cap: prevents silent OOM kills (no result file written) on
    # dense Goldilocks/Extreme instances. Gurobi raises GRB.MEM_LIMIT instead.
    mem_limit = getattr(config, "GUROBI_MEM_LIMIT_GB", None)
    if mem_limit is not None:
        # SoftMemLimit is GB; available in Gurobi 9.5+. Fall back silently
        # on older versions where the parameter is unknown.
        try:
            model.setParam("SoftMemLimit", float(mem_limit))
        except gp.GurobiError:
            pass

    # Decision variables: x[i,j] binary, one per valid graph edge only
    x = {}
    for i, j in edge_list:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Objective: maximise sum(cost * x)  -- graph-edge costs are 0 or positive
    obj = gp.quicksum(cost[i][j] * x[i, j] for i, j in edge_list)
    model.setObjective(obj, GRB.MAXIMIZE)

    # Assignment constraints: each row and column covered exactly once,
    # summing only over variables that exist (graph edges).
    row_vars = {i: [] for i in range(n)}
    col_vars = {j: [] for j in range(n)}
    for i, j in edge_list:
        row_vars[i].append(x[i, j])
        col_vars[j].append(x[i, j])

    for i in range(n):
        model.addConstr(gp.quicksum(row_vars[i]) == 1, name=f"row_{i}")
    for j in range(n):
        model.addConstr(gp.quicksum(col_vars[j]) == 1, name=f"col_{j}")

    # Conflict constraints
    for c in conflicts:
        i1, j1, i2, j2 = c
        model.addConstr(x[i1, j1] + x[i2, j2] <= 1, name=f"conflict_{i1}_{j1}_{i2}_{j2}")

    # Solve, with a callback that timestamps the first incumbent.
    cb, cb_state = _make_first_feasible_callback(start_time)
    model.optimize(cb)
    runtime = time.time() - start_time

    status_code = model.Status
    status = _status_name(status_code)
    sol_count = int(model.SolCount)
    nodes = int(model.NodeCount)
    best_bound = _safe_obj_bound(model)
    first_feasible_time = cb_state["first_feasible_time"]

    # Common assignment extraction: any time SolCount > 0, an incumbent is
    # available regardless of status (TIME_LIMIT, SUBOPTIMAL, INTERRUPTED,
    # NUMERIC with partial result, ...).
    assignment: Optional[List[Tuple[int, int]]] = None
    obj_val: Optional[float] = None
    assignment_valid = False
    if sol_count > 0:
        try:
            assignment = [(i, j) for i, j in edge_list if x[i, j].X > 0.5]
            assignment.sort(key=lambda e: e[0])
            obj_val = float(model.ObjVal)
            assignment_valid = _validate_assignment(assignment, n)
        except (AttributeError, gp.GurobiError):
            # Solution attribute not retrievable despite SolCount > 0
            # (rare, e.g. NUMERIC). Treat as no feasible found.
            assignment = None
            obj_val = None
            assignment_valid = False

    if status_code == GRB.INFEASIBLE:
        # No solution exists; clear any best_bound we read (it's not meaningful).
        return {
            "status": status,
            "objective": None,
            "best_bound": None,
            "assignment": None,
            "runtime": runtime,
            "gap": None,
            "nodes_explored": nodes,
            "solutions_found": 0,
            "first_feasible_time": None,
            "assignment_valid": False,
        }

    gap = _sanitised_gap(model, obj_val) if status_code == GRB.OPTIMAL or sol_count > 0 else None
    # When OPTIMAL, MIPGap should be ~0; force exactly 0.0 for cleanliness.
    if status_code == GRB.OPTIMAL and obj_val is not None:
        gap = 0.0

    return {
        "status": status,
        "objective": obj_val,
        "best_bound": best_bound,
        "assignment": assignment,
        "runtime": runtime,
        "gap": gap,
        "nodes_explored": nodes,
        "solutions_found": sol_count,
        "first_feasible_time": first_feasible_time,
        "assignment_valid": assignment_valid,
    }


def solve_from_file(instance_path: str, **kwargs) -> Dict:
    """Load an instance from JSON and solve it."""
    instance = ab.load_instance(instance_path)
    return solve_instance(instance, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Solve MAX-APC instance with Gurobi.")
    parser.add_argument("--instance", required=True, help="Path to instance JSON file")
    parser.add_argument("--time-limit", type=float, help="Time limit in seconds")
    parser.add_argument("--verbose", action="store_true", help="Show Gurobi output")
    args = parser.parse_args()

    result = solve_from_file(args.instance, time_limit=args.time_limit, verbose=args.verbose)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
