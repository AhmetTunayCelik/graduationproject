"""
apc_base.py
============

Core module for the Maximum Assignment Problem with Conflict pair constraints
(MAX-APC). Provides instance loading/storage, the Lagrangean relaxation
framework, the subgradient optimisation loop, and result storage utilities.

This module is intentionally heuristic-agnostic. It does not contain any
heuristic; the subgradient loop accepts a caller-provided callable repair_fn.
All heuristics reside in the heuristics/ package and conform to a standard
interface (see :func:`subgradient_solve` for details).

Key design decisions (as per project supervisor's pseudocode)
--------------------------------------------------------------
1. The seed feasible solution E0 is stored in each instance (costs = 0).
2. Conflicts are only defined among non-E0 edges with distinct rows/columns.
3. Subgradient update uses the *current* Lagrangian bound Z_Lag in the
   step‑size numerator (not the best seen so far), and UB is unconditionally
   set to Z_Lag every iteration (no running minimum).
4. Step length is halved after 20 stagnant iterations (no LB improvement).
5. All persistent artefacts (instances, subgradient caches, heuristic results)
   are stored as JSON with a stable schema for downstream analysis.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from parameters import config

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
Edge = Tuple[int, int]
Assignment = List[Edge]
Conflict = List[int]                     # [i1, j1, i2, j2]
Instance = Dict[str, Any]                # schema defined in save/load
RepairFn = Callable[..., Tuple[Assignment, float, bool]]


# -----------------------------------------------------------------------------
# Core solvers and feasibility utilities
# -----------------------------------------------------------------------------
def hungarian_max(profit_matrix: np.ndarray) -> Tuple[Assignment, float]:
    """Solve the maximum-weight assignment problem via the Hungarian method.

    Parameters
    ----------
    profit_matrix : (n, n) array_like
        Profit matrix (maximisation).

    Returns
    -------
    assignment : list of (int, int)
        Optimal assignment, sorted by row.
    objective : float
        Total profit.
    """
    P = np.asarray(profit_matrix, dtype=float)
    row_ind, col_ind = linear_sum_assignment(-P)
    assignment = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
    objective = float(P[row_ind, col_ind].sum())
    return assignment, objective


def find_violations(
    assignment: Assignment,
    conflicts: List[Conflict],
    n: int,
) -> List[int]:
    """Return indices of conflicts violated by an assignment.

    Parameters
    ----------
    assignment : list of (i,j)
        Assignment to test.
    conflicts : list of [i1,j1,i2,j2]
        Explicit conflict list.
    n : int
        Problem size.

    Returns
    -------
    list of int
        Positions in conflicts that are violated.
    """
    asgn_flat = np.zeros(n * n, dtype=bool)
    for i, j in assignment:
        asgn_flat[i * n + j] = True
    if not conflicts:
        return []
    c = np.array(conflicts, dtype=int)
    flat_e1 = c[:, 0] * n + c[:, 1]
    flat_e2 = c[:, 2] * n + c[:, 3]
    violated = asgn_flat[flat_e1] & asgn_flat[flat_e2]
    return list(np.where(violated)[0])


# -----------------------------------------------------------------------------
# Storage (instances, heuristic results, subgradient cache)
# -----------------------------------------------------------------------------
def _density_tags(instance: Instance):
    """Return (c_tag, g_tag_or_None) for filename construction.

    Supports both legacy schema (density key) and new schema
    (conflict_graph_density + graph_density).
    """
    c_density = instance.get("conflict_graph_density", instance.get("density"))
    g_density = instance.get("graph_density", 1.0)
    c_tag = f"d{int(round(c_density * 10000)):04d}"
    if g_density is None or abs(g_density - 1.0) < 1e-9:
        return c_tag, None
    g_tag = f"g{int(round(g_density * 10000)):04d}"
    return c_tag, g_tag


def _instance_filename(instance: Instance) -> str:
    """Deterministic filename for an instance.

    New sparse instances:  instance_n{n}_d{c_density}_g{g_density}_s{seed}.json
    Complete-graph (legacy-compatible): instance_n{n}_d{density}_s{seed}.json
    """
    c_tag, g_tag = _density_tags(instance)
    if g_tag is None:
        return f"instance_n{instance['n']}_{c_tag}_s{instance['seed']}.json"
    return f"instance_n{instance['n']}_{c_tag}_{g_tag}_s{instance['seed']}.json"


def _result_filename(instance: Instance, heuristic_name: str) -> str:
    """Filename for a heuristic result."""
    c_tag, g_tag = _density_tags(instance)
    mid = f"{c_tag}_{g_tag}" if g_tag else c_tag
    return (f"result_n{instance['n']}_{mid}_s{instance['seed']}"
            f"_{heuristic_name}.json")


def _subgradient_filename(instance: Instance) -> str:
    """Filename for a cached subgradient result."""
    c_tag, g_tag = _density_tags(instance)
    mid = f"{c_tag}_{g_tag}" if g_tag else c_tag
    return (f"subgradient_n{instance['n']}_{mid}_s{instance['seed']}"
            f".json")


def save_instance(instance: Instance, directory: str = "instances") -> str:
    """Persist an instance to disk as JSON."""
    os.makedirs(directory, exist_ok=True)
    fpath = os.path.join(directory, _instance_filename(instance))
    payload = dict(instance)
    payload["E0"] = [list(e) for e in instance["E0"]]
    if "graph_edges" in payload:
        payload["graph_edges"] = [list(e) for e in instance["graph_edges"]]
    with open(fpath, "w") as f:
        json.dump(payload, f, indent=2)
    return fpath


def load_instance(fpath: str) -> Instance:
    """Load an instance previously saved with save_instance()."""
    with open(fpath) as f:
        instance = json.load(f)
    instance["E0"] = [tuple(e) for e in instance["E0"]]
    if "graph_edges" in instance:
        instance["graph_edges"] = [tuple(e) for e in instance["graph_edges"]]
    # Back-compat: old instances used 'density' for conflict density
    if "density" in instance and "conflict_graph_density" not in instance:
        instance["conflict_graph_density"] = instance["density"]
    if "graph_density" not in instance:
        instance["graph_density"] = 1.0
    if "graph_edges" not in instance:
        n = instance["n"]
        instance["graph_edges"] = [(i, j) for i in range(n) for j in range(n)]
    return instance


def save_result(
    instance: Instance,
    heuristic_name: str,
    result: Dict[str, Any],
    directory: str = "results",
) -> str:
    """Persist a single heuristic's result on an instance.

    The result dict may contain any JSON‑serialisable data. It is enriched
    with instance identifiers (n, seed, num_conflicts, E0_objective) and the
    heuristic name for easy tabular loading.
    """
    os.makedirs(directory, exist_ok=True)
    fpath = os.path.join(directory, _result_filename(instance, heuristic_name))
    payload = {
        "n": instance["n"],
        "seed": instance["seed"],
        "num_conflicts": len(instance["conflicts"]),
        "E0_objective": sum(instance["cost_matrix"][i][j] for i, j in instance["E0"]),
        "heuristic": heuristic_name,
        **_jsonify(result),
    }
    with open(fpath, "w") as f:
        json.dump(payload, f, indent=2)
    return fpath


def save_subgradient_result(
    instance: Instance,
    subgradient_output: Dict[str, Any],
    directory: str = "results",
) -> str:
    """Cache the output of subgradient_solve() for reuse by multiple heuristics.

    The output dict must contain at least the fields that subsequent
    heuristics need: 'x_star_final', 'LB', 'UB', 'gap_pct', 'iterations',
    'lambdas_final', 'terminated_reason', 'runtime_seconds'.
    """
    os.makedirs(directory, exist_ok=True)
    fpath = os.path.join(directory, _subgradient_filename(instance))
    payload = _jsonify(subgradient_output)
    with open(fpath, "w") as f:
        json.dump(payload, f, indent=2)
    return fpath


def load_subgradient_result(instance: Instance, directory: str = "results") -> Optional[Dict[str, Any]]:
    """Load a cached subgradient result, or None if not present."""
    fpath = os.path.join(directory, _subgradient_filename(instance))
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        return json.load(f)


def _jsonify(obj: Any) -> Any:
    """Recursively convert tuples, numpy scalars, and arrays to JSON‑friendly types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    return obj


# -----------------------------------------------------------------------------
# Subgradient algorithm (Lagrangean dual solver)
# -----------------------------------------------------------------------------
def subgradient_solve(
    instance: Instance,
    repair_fn: Optional[RepairFn] = None,
    K_max: int = config.SUBG_MAX_ITERS,
    epsilon: float = config.SUBG_EPSILON,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve the Lagrangean dual of a MAX-APC instance by subgradient ascent.

    The relaxation dualises explicit conflict constraints using non‑negative
    multipliers λ_ef. At each iteration the subproblem (an assignment problem)
    is solved with Hungarian. The subgradient step follows the pseudocode
    provided by the project supervisor.

    Parameters
    ----------
    instance : dict
        Instance dictionary as produced by instance_generator.generate_instance().
    repair_fn : callable, optional
        Heuristic that converts the subproblem solution into a feasible
        assignment, used to update the lower bound LB. Signature:
        repair_fn(x_star, cost, conflicts, n, E0) -> (assignment, obj, feasible).
    K_max : int
        Maximum number of iterations.
    epsilon : float
        Tolerance for subgradient norm.
    verbose : bool
        Print iteration progress.

    Returns
    -------
    dict
        Contains: LB, UB, gap_pct, x_LB, x_star_final, iterations,
        lambdas_final, runtime_seconds, terminated_reason.
    """
    n = instance["n"]
    cost = np.array(instance["cost_matrix"], dtype=float)
    conflicts = instance["conflicts"]
    num_conflicts = len(conflicts)
    E0 = instance["E0"]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Subgradient solver — n = {n}, |C| = {num_conflicts}")
        print(f"{'=' * 60}")

    # Initialise LB with E0 (cost = 0 by construction)
    LB = float(sum(cost[i, j] for i, j in E0))
    x_LB = list(E0)

    # Initial UB: unconstrained assignment optimum (valid upper bound)
    _, z0 = hungarian_max(cost)
    UB = z0

    k = 0
    t_no_improve = 0
    pi_k = 2.0
    lambdas = np.zeros(num_conflicts, dtype=float)

    # Precompute flat indices for conflicts
    if num_conflicts > 0:
        c_arr = np.array(conflicts, dtype=int)
        c_e1_flat = c_arr[:, 0] * n + c_arr[:, 1]
        c_e2_flat = c_arr[:, 2] * n + c_arr[:, 3]
    else:
        c_e1_flat = c_e2_flat = np.array([], dtype=int)

    t_start = time.time()
    terminated_reason = "iteration_limit"
    x_star = list(E0)      # placeholder

    while k < K_max:
        k += 1

        # Build penalised profit matrix
        p_tilde = cost.copy()
        if num_conflicts > 0:
            np.add.at(p_tilde.ravel(), c_e1_flat, -lambdas)
            np.add.at(p_tilde.ravel(), c_e2_flat, -lambdas)

        x_star, z_star = hungarian_max(p_tilde)
        Z_Lag = z_star + float(np.sum(lambdas))
        UB = Z_Lag        # per spec: unconditional assignment

        # Check feasibility of subproblem solution
        asgn_flat = np.zeros(n * n, dtype=bool)
        for i, j in x_star:
            asgn_flat[i * n + j] = True
        if num_conflicts > 0:
            violated_mask = asgn_flat[c_e1_flat] & asgn_flat[c_e2_flat]
            num_violations = int(violated_mask.sum())
        else:
            num_violations = 0

        if num_violations == 0:
            # Feasible solution found
            obj = float(sum(cost[i, j] for i, j in x_star))

            # Complementary slackness optimality test
            if num_conflicts > 0:
                active = asgn_flat[c_e1_flat] | asgn_flat[c_e2_flat]
                slackness_ok = bool(np.all(active[lambdas > 0]))
            else:
                slackness_ok = True

            if slackness_ok:
                if obj > LB:
                    LB = obj
                    x_LB = list(x_star)
                if verbose:
                    print(f"  Iter {k}: feasible & complementary slackness → optimum")
                terminated_reason = "complementary_slackness"
                break

            if obj > LB:
                LB = obj
                x_LB = list(x_star)
                t_no_improve = 0
            else:
                t_no_improve += 1
        else:
            # Infeasible subproblem solution: try to repair
            if repair_fn is not None:
                try:
                    x_hat, z_hat, feasible = repair_fn(x_star, cost, conflicts, n, E0)
                except TypeError:
                    # Compatibility with older signatures
                    x_hat, z_hat, feasible = repair_fn(x_star, cost, conflicts, n)
                if feasible and z_hat > LB:
                    LB = float(z_hat)
                    x_LB = list(x_hat)
                    t_no_improve = 0
                else:
                    t_no_improve += 1
            else:
                t_no_improve += 1

        # Halve step length after stagnant iterations
        if t_no_improve >= config.SUBG_STAGNATION_LIMIT:
            t_no_improve = 0
            pi_k /= 2.0

        # Subgradient direction and multiplier update
        if num_conflicts > 0:
            s = 1.0 - asgn_flat[c_e1_flat].astype(float) - asgn_flat[c_e2_flat].astype(float)
            s_norm_sq = float(np.dot(s, s))
            if s_norm_sq < epsilon:
                if verbose:
                    print(f"  Iter {k}: subgradient norm below tolerance")
                terminated_reason = "small_subgradient"
                break
            alpha = pi_k * (Z_Lag - LB) / s_norm_sq
            lambdas = np.maximum(0.0, lambdas + alpha * s)

        if verbose and (k % 50 == 0 or k <= 5):
            gap_pct = ((UB - LB) / max(abs(LB), 1e-10)) * 100.0
            print(f"  Iter {k:4d}: LB = {LB:.2f}, UB = {UB:.2f}, "
                  f"gap = {gap_pct:.2f}%, pi = {pi_k:.6f}")

    runtime = time.time() - t_start
    gap_pct = ((UB - LB) / max(abs(LB), 1e-10)) * 100.0

    if verbose:
        print(f"\n  Termination: {terminated_reason}")
        print(f"  Final LB = {LB:.2f}, UB = {UB:.2f}, gap = {gap_pct:.2f}%")
        print(f"  Runtime = {runtime:.3f} s over {k} iterations")

    return {
        "LB": float(LB),
        "UB": float(UB),
        "gap_pct": float(gap_pct),
        "x_LB": [tuple(e) for e in x_LB],
        "x_star_final": [tuple(e) for e in x_star],
        "iterations": int(k),
        "lambdas_final": [float(v) for v in lambdas],
        "runtime_seconds": float(runtime),
        "terminated_reason": terminated_reason,
    }


__all__ = [
    "Edge", "Assignment", "Conflict", "Instance", "RepairFn",
    "hungarian_max", "find_violations",
    "subgradient_solve",
    "save_instance", "load_instance", "save_result",
    "save_subgradient_result", "load_subgradient_result",
]