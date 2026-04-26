"""
gurobi_solver.py
================

Exact solver for the Maximum Assignment Problem with Conflicts (MAX-APC)
using Gurobi. Given an instance (cost matrix, conflict list), it builds
a binary linear program and solves it to optimality.

The model uses:
    - Assignment constraints: sum_j x[i,j] = 1 for each row i,
                              sum_i x[i,j] = 1 for each column j.
    - Conflict constraints: x[i1,j1] + x[i2,j2] ≤ 1 for each conflict.
    - Objective: maximise Σ c[i,j] * x[i,j].

Usage:
    python gurobi_solver.py --instance path/to/instance.json

Or programmatically:
    from gurobi_solver import solve_instance
    result = solve_instance(instance_dict)
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

import apc_base as ab  # for loading instances and type definitions


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
            'status' : str (e.g., "OPTIMAL", "TIME_LIMIT", "INFEASIBLE")
            'objective' : float (optimal objective, or None if not optimal)
            'assignment' : list of (i, j) (optimal assignment, or None)
            'runtime' : float (solver wall-clock time)
            'gap' : float (relative MIP gap, if not optimal)
    """
    n = instance["n"]
    cost = instance["cost_matrix"]
    conflicts = instance["conflicts"]

    model = gp.Model("MAX-APC")
    if not verbose:
        model.setParam("OutputFlag", 0)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)

    # Decision variables: x[i,j] binary
    x = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Objective: maximise sum(cost * x)
    obj = gp.quicksum(cost[i][j] * x[i, j] for i in range(n) for j in range(n))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Assignment constraints: one per row
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"row_{i}")

    # Assignment constraints: one per column
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1, name=f"col_{j}")

    # Conflict constraints
    for c in conflicts:
        i1, j1, i2, j2 = c
        model.addConstr(x[i1, j1] + x[i2, j2] <= 1, name=f"conflict_{i1}_{j1}_{i2}_{j2}")

    # Solve
    start_time = time.time()
    model.optimize()
    runtime = time.time() - start_time

    status = model.Status
    if status == GRB.OPTIMAL:
        obj_val = model.ObjVal
        assignment = [(i, j) for i in range(n) for j in range(n) if x[i, j].X > 0.5]
        # assignment should already have n edges; sort for consistency
        assignment.sort(key=lambda e: e[0])
        return {
            "status": "OPTIMAL",
            "objective": float(obj_val),
            "assignment": assignment,
            "runtime": runtime,
            "gap": 0.0,
        }
    elif status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            assignment = [(i, j) for i in range(n) for j in range(n) if x[i, j].X > 0.5]
            assignment.sort(key=lambda e: e[0])
            obj_val = model.ObjVal
            gap = model.MIPGap if hasattr(model, "MIPGap") else None
        else:
            # Timed out before finding any feasible solution.
            assignment = []
            obj_val = 0.0
            gap = float("inf")
        return {
            "status": "TIME_LIMIT",
            "objective": float(obj_val),
            "assignment": assignment,
            "runtime": runtime,
            "gap": gap,
        }
    elif status == GRB.INFEASIBLE:
        return {
            "status": "INFEASIBLE",
            "objective": None,
            "assignment": None,
            "runtime": runtime,
            "gap": None,
        }
    else:
        return {
            "status": "UNKNOWN",
            "objective": None,
            "assignment": None,
            "runtime": runtime,
            "gap": None,
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