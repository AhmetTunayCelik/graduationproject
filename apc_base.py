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
   step‑size numerator, while reported UB is the best dual bound seen so far.
4. Step length is halved after 20 stagnant iterations (no LB improvement).
5. All persistent artefacts (instances, heuristic results) are stored as JSON
   with a stable schema for downstream analysis. The subgradient loop is
   recomputed in memory for every heuristic evaluation (no cache) so that
   each heuristic pays the full CPU cost of its own dual ascent.
"""

from __future__ import annotations

import json
import os
import platform
import sys
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


_EMPTY_INT32_ARR = np.empty(0, dtype=np.int32)


def build_conflict_adjacency_int(
    conflicts: List[Conflict],
    n: int,
) -> List[np.ndarray]:
    """Build an integer-keyed conflict adjacency list as numpy int32 arrays.

    Returns a list of length n*n where index e_id = i*n + j holds an
    np.ndarray(dtype=int32) of edge IDs that conflict with edge (i, j).
    Empty slots share a single empty int32 array sentinel (no per-slot cost).

    Memory:
        - Each int32 entry: 4 bytes (vs ~50 bytes per Python int in a set)
        - Per-array overhead: ~96 bytes (vs ~232 bytes per set object)
        - For 1.88M conflict relations: ~10 MB total (vs ~94 MB with sets)
        - **~10× memory reduction** over the previous Set[int] form.

    Trade-off: numpy arrays don't support O(1) `in` membership checks. Callers
    that need fast membership should build a bool bitmask of size n*n
    (np.zeros(n*n, dtype=bool)) and use mask[ids] for vectorised set tests.
    """
    nn = n * n

    # First pass: count degree of each edge id to size arrays exactly
    degree = np.zeros(nn, dtype=np.int32)
    for c in conflicts:
        degree[c[0] * n + c[1]] += 1
        degree[c[2] * n + c[3]] += 1

    # Allocate per-edge int32 arrays at the correct size (no overgrowth)
    adj: List[np.ndarray] = [None] * nn  # type: ignore[list-item]
    cursor = np.zeros(nn, dtype=np.int32)  # write index per edge
    for eid in range(nn):
        d = int(degree[eid])
        if d == 0:
            adj[eid] = _EMPTY_INT32_ARR  # shared sentinel
        else:
            adj[eid] = np.empty(d, dtype=np.int32)

    # Second pass: fill the arrays
    for c in conflicts:
        e1 = c[0] * n + c[1]
        e2 = c[2] * n + c[3]
        adj[e1][cursor[e1]] = e2
        cursor[e1] += 1
        adj[e2][cursor[e2]] = e1
        cursor[e2] += 1

    return adj


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


def is_valid_assignment(
    assignment: Assignment,
    conflicts: List[Conflict],
    n: int,
    graph_edges: Optional[List[Edge]] = None,
) -> bool:
    """Return True iff assignment satisfies APC primal feasibility.

    Checks the assignment permutation constraints, explicit conflict-pair
    constraints, bounds on edge indices, and optional sparse-graph membership.
    """
    if len(assignment) != n:
        return False

    rows = set()
    cols = set()
    normalised: Assignment = []
    for edge in assignment:
        if len(edge) != 2:
            return False
        i, j = int(edge[0]), int(edge[1])
        if i < 0 or i >= n or j < 0 or j >= n:
            return False
        rows.add(i)
        cols.add(j)
        normalised.append((i, j))

    if len(rows) != n or len(cols) != n:
        return False

    if graph_edges is not None:
        allowed = {tuple(e) for e in graph_edges}
        if any(edge not in allowed for edge in normalised):
            return False

    return len(find_violations(normalised, conflicts, n)) == 0


# -----------------------------------------------------------------------------
# Storage (instances, heuristic results, subgradient cache)
# -----------------------------------------------------------------------------

_CATEGORY_PREFIX = {
    "standard":   "instance",
    "goldilocks": "difficult_instance_goldilockzone",
    "degen":      "difficult_instance_degen",
    "extreme":    "difficult_instance_extreme",
}


def _alpha_tag(instance: Instance) -> str:
    """Graph density → tag string, e.g. 0.4 → 'a04', 1.0 → 'a10'."""
    g = instance.get("graph_density", 1.0) or 1.0
    return f"a{int(round(g * 10)):02d}"


def _beta_tag(instance: Instance) -> str:
    """Conflict density → tag string, e.g. 0.01 → 'b010', 0.001 → 'b001'."""
    c = instance.get("conflict_graph_density", instance.get("density", 0.0))
    return f"b{int(round(c * 1000)):03d}"


def _instance_filename(instance: Instance) -> str:
    """Deterministic filename for an instance.

    instance_n{n}_a{alpha}_b{beta}_s{seed}.json
    difficult_instance_goldilockzone_n{n}_a{alpha}_b{beta}_s{seed}.json
    difficult_instance_degen_n{n}_...  |  difficult_instance_extreme_n{n}_...
    """
    category = instance.get("instance_category", "standard")
    prefix = _CATEGORY_PREFIX.get(category, f"unknown_{category}")
    n, seed = instance["n"], instance["seed"]
    return f"{prefix}_n{n}_{_alpha_tag(instance)}_{_beta_tag(instance)}_s{seed}.json"


def _result_filename(instance: Instance, heuristic_name: str) -> str:
    """Filename for a heuristic result.

    standard:  {heuristic_name}_n{n}_a{alpha}_b{beta}_s{seed}.json
    difficult: difficult_{heuristic_name}_{type}_n{n}_a{alpha}_b{beta}_s{seed}.json
    """
    category = instance.get("instance_category", "standard")
    n, seed = instance["n"], instance["seed"]
    tags = f"n{n}_{_alpha_tag(instance)}_{_beta_tag(instance)}_s{seed}"

    if category == "standard":
        return f"{heuristic_name}_{tags}.json"
    else:
        return f"difficult_{heuristic_name}_{category}_{tags}.json"


def _config_snapshot() -> Dict[str, Any]:
    """Capture the public scalar/list attributes of parameters.config.

    Saved into batch metadata for the reproducibility appendix.
    """
    snap: Dict[str, Any] = {}
    for k in dir(config):
        if k.startswith("_"):
            continue
        v = getattr(config, k)
        if isinstance(v, (int, float, str, bool, list, tuple)) and not callable(v):
            snap[k] = list(v) if isinstance(v, tuple) else v
    return snap


def write_run_metadata(directory: str, batch_label: str) -> str:
    """Persist a `metadata_<label>.json` capturing host, library, and config
    state at the start of a batch. Required for any reproducibility-grade
    write-up.

    Captures: timestamp, OS, CPU count, Python/numpy/scipy versions, Gurobi
    version (best-effort import), and a flat snapshot of parameters.config.
    """
    os.makedirs(directory, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        import scipy
        scipy_ver = scipy.__version__
    except Exception:
        scipy_ver = "unknown"

    try:
        import gurobipy
        gurobi_ver = ".".join(str(x) for x in gurobipy.gurobi.version())
    except Exception:
        gurobi_ver = "unavailable"

    metadata = {
        "batch_label": batch_label,
        "timestamp": ts,
        "host": {
            "platform":   platform.platform(),
            "machine":    platform.machine(),
            "processor":  platform.processor(),
            "cpu_count":  os.cpu_count(),
            "python":     sys.version.split()[0],
        },
        "libraries": {
            "numpy":  np.__version__,
            "scipy":  scipy_ver,
            "gurobi": gurobi_ver,
        },
        "config": _config_snapshot(),
    }

    fpath = os.path.join(directory, f"metadata_{batch_label}.json")
    _atomic_write_json(fpath, metadata)
    return fpath


def _atomic_write_json(fpath: str, payload: Any, indent: int = 2) -> None:
    """Write a JSON payload atomically.

    Why: a process kill mid-`json.dump` leaves an empty/truncated file at the
    destination path. Skip-if-exists logic (in gurobi_batch / batch_experiment)
    then permanently treats that cell as 'already solved'. Writing to a .tmp
    sibling and renaming guarantees the destination either contains the full
    JSON or doesn't exist.
    """
    tmp = fpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=indent)
    os.replace(tmp, fpath)


def save_instance(instance: Instance, directory: str = "instances") -> str:
    """Persist an instance to disk as JSON."""
    os.makedirs(directory, exist_ok=True)
    fpath = os.path.join(directory, _instance_filename(instance))
    payload = dict(instance)
    payload["E0"] = [list(e) for e in instance["E0"]]
    if "graph_edges" in payload:
        payload["graph_edges"] = [list(e) for e in instance["graph_edges"]]
    _atomic_write_json(fpath, payload)
    return fpath


def _validate_E0(instance: Instance) -> None:
    """Light sanity check that the seed assignment is a valid permutation and
    conflict-free. Issues a warning (does not raise) so legacy or hand-crafted
    instances still load."""
    n = instance["n"]
    E0 = instance["E0"]
    rows = {i for i, _ in E0}
    cols = {j for _, j in E0}
    if len(E0) != n or len(rows) != n or len(cols) != n:
        print(f"  [Warning] E0 in instance is not a valid n-permutation "
              f"(|E0|={len(E0)}, rows={len(rows)}, cols={len(cols)})")
        return
    violations = find_violations(E0, instance.get("conflicts", []), n)
    if violations:
        print(f"  [Warning] E0 in instance violates {len(violations)} "
              f"conflict pair(s); the seed feasible solution is not feasible.")


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
    _validate_E0(instance)
    return instance


def save_result(
    instance: Instance,
    heuristic_name: str,
    result: Dict[str, Any],
    directory: str = "results",
    subgradient_output: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist a single heuristic's result on an instance.

    The result dict may contain any JSON‑serialisable data. It is enriched
    with instance identifiers (n, seed, num_conflicts, E0_objective) and the
    heuristic name for easy tabular loading.

    If *subgradient_output* is provided (the dict returned by
    subgradient_solve()), the best incumbent found during the subgradient
    loop (``x_LB`` / ``LB``) is written as the top-level
    ``incumbent_objective`` and ``incumbent_assignment`` fields so that the
    result file always reports the true best feasible solution, not merely
    the last repair attempt.
    """
    os.makedirs(directory, exist_ok=True)
    fpath = os.path.join(directory, _result_filename(instance, heuristic_name))
    e0_objective = float(sum(instance["cost_matrix"][i][j] for i, j in instance["E0"]))

    def _is_nontrivial_obj(obj: Optional[float]) -> bool:
        """True when the objective improves over the guaranteed E0 fallback."""
        return obj is not None and float(obj) > e0_objective + 1e-9

    # Best feasible solution from the subgradient loop (x_LB) takes priority.
    # If subgradient_output is not supplied we fall back to the best feasible
    # variant found among the heuristic's own ordering results.
    if subgradient_output is not None:
        # `LB` is None when the loop found no feasible solution beyond E0.
        lb_val = subgradient_output.get("LB")
        incumbent_obj = float(lb_val) if lb_val is not None else None
        incumbent_asgn = _jsonify(subgradient_output.get("x_LB")) if subgradient_output.get("x_LB") else None
        # Explicit failure flag for academic-integrity reporting in analysis.py.
        # Falls back to "is the incumbent something" if the field is absent
        # (legacy result caches predating feasible_found tracking).
        feasible_found = bool(
            subgradient_output.get(
                "feasible_found", _is_nontrivial_obj(incumbent_obj)
            )
        )
        if not _is_nontrivial_obj(incumbent_obj):
            incumbent_obj = None
            incumbent_asgn = None
            feasible_found = False
    else:
        # Fallback: scan heuristic_output for best feasible ordering
        ordering_variants = (
            result.get("heuristic_output", {}).get("ordering_variants", {})
        )
        best_obj, best_asgn = None, None
        for rec in ordering_variants.values():
            rec_obj = rec.get("objective")
            if (
                rec.get("feasible")
                and _is_nontrivial_obj(rec_obj)
                and (best_obj is None or rec_obj > best_obj)
            ):
                best_obj = rec_obj
                best_asgn = rec["assignment"]
        incumbent_obj = best_obj
        incumbent_asgn = best_asgn
        feasible_found = incumbent_obj is not None

    # Allow the *post-loop* heuristic call (run with converged lambdas) to
    # update the incumbent if it strictly beats the best in-loop repair.
    # This matters for lambda-aware heuristics whose dual information is
    # only fully informative once the subgradient ascent has converged.
    heur_out = result.get("heuristic_output", {}) or {}
    candidate_assign: Optional[Any] = None
    candidate_obj: Optional[float] = None
    if "ordering_variants" in heur_out:
        for variant in heur_out.get("ordering_variants", {}).values():
            if (
                variant.get("feasible")
                and _is_nontrivial_obj(variant.get("objective"))
            ):
                v_obj = float(variant["objective"])
                if candidate_obj is None or v_obj > candidate_obj:
                    candidate_obj = v_obj
                    candidate_assign = variant.get("assignment")
    elif heur_out.get("feasible") and _is_nontrivial_obj(heur_out.get("objective")):
        candidate_obj = float(heur_out["objective"])
        candidate_assign = heur_out.get("assignment")

    if candidate_obj is not None:
        if incumbent_obj is None or candidate_obj > incumbent_obj:
            incumbent_obj = candidate_obj
            incumbent_asgn = _jsonify(candidate_assign) if candidate_assign else incumbent_asgn
            feasible_found = True

    payload = {
        "n": instance["n"],
        "seed": instance["seed"],
        "num_conflicts": len(instance["conflicts"]),
        "E0_objective": e0_objective,
        "heuristic": heuristic_name,
        "incumbent_objective": incumbent_obj,
        "incumbent_assignment": incumbent_asgn,
        "feasible_found": feasible_found,
        **_jsonify(result),
    }
    _atomic_write_json(fpath, payload)
    return fpath


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
    time_limit: float = config.HEURISTIC_TIME_LIMIT,
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
    time_limit : float
        Wall-clock budget in seconds (default 600 s = 10 min). The loop exits
        at the start of the first iteration that would exceed this limit.
        terminated_reason will be "time_limit" in that case.
    verbose : bool
        Print iteration progress.

    Returns
    -------
    dict
        Contains: LB (or None), UB, gap_pct (or None), feasible_found, x_LB (or None),
        x_star_final, iterations, lambdas_final, runtime_seconds, terminated_reason,
        iteration_history.

        LB, x_LB, and gap_pct are None when no feasible solution beyond E0 is found
        during time/iteration-limited runs (to prevent artificial 0s in analysis).
        feasible_found is True only when LB strictly improves in the loop.
        iteration_history is a list of per-iteration dicts with keys
        {iter, LB, UB, pi_k, num_violations, elapsed_s} for plotting bound convergence.
    """
    n = instance["n"]
    # Downcast to float32: precision is plenty for subgradient ascent and halves memory.
    cost = np.asarray(instance["cost_matrix"], dtype=np.float32)
    conflicts = instance["conflicts"]
    num_conflicts = len(conflicts)
    E0 = instance["E0"]
    # Forwarded to repair heuristics so they restrict candidates to the
    # underlying graph. None for legacy / complete-graph instances.
    graph_edges = instance.get("graph_edges")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Subgradient solver — n = {n}, |C| = {num_conflicts}")
        print(f"{'=' * 60}")

    # Initialise LB with E0 (cost = 0 by construction)
    LB = float(sum(cost[i, j] for i, j in E0))
    x_LB = list(E0)
    # True only when a real feasible solution better than E0 is found in the loop.
    # Stays False if the run exits on time/iteration limit with only E0 as fallback.
    feasible_found = False

    # Initial UB: unconstrained assignment optimum (valid upper bound)
    _, z0 = hungarian_max(cost)
    UB_current = z0
    UB_best = z0
    UB = UB_best

    k = 0
    t_no_improve = 0
    pi_k = 2.0
    lambdas = np.zeros(num_conflicts, dtype=np.float32)
    neighbours = build_conflict_adjacency_int(conflicts, n)

    # Precompute flat indices for conflicts (int32 is plenty: max id = n*n ≤ 150²)
    if num_conflicts > 0:
        c_arr = np.asarray(conflicts, dtype=np.int32)
        c_e1_flat = (c_arr[:, 0] * n + c_arr[:, 1]).astype(np.int32, copy=False)
        c_e2_flat = (c_arr[:, 2] * n + c_arr[:, 3]).astype(np.int32, copy=False)
        del c_arr  # 4-column intermediate no longer needed
        # Preallocated buffer for subgradient direction
        s_buf = np.empty(num_conflicts, dtype=np.float32)
    else:
        c_e1_flat = c_e2_flat = np.empty(0, dtype=np.int32)
        s_buf = np.empty(0, dtype=np.float32)

    # Reusable buffers (avoid allocating fresh arrays each iteration)
    p_tilde = np.empty_like(cost)
    asgn_flat = np.zeros(n * n, dtype=bool)

    t_start = time.time()
    terminated_reason = "iteration_limit"
    x_star = list(E0)      # placeholder
    iteration_history: List[Dict[str, float]] = []
    num_violations = -1    # Sentinel: -1 indicates incomplete iteration (time-limit before solve)

    def _record_iter():
        iteration_history.append({
            "iter": int(k),
            "LB": float(LB),
            "UB": float(UB_best),
            "UB_current": float(UB_current),
            "pi_k": float(pi_k),
            "num_violations": int(num_violations),
            "elapsed_s": float(time.time() - t_start),
        })

    while k < K_max:
        k += 1

        if time.time() - t_start >= time_limit:
            terminated_reason = "time_limit"
            _record_iter()
            break

        # Build penalised profit matrix in-place into preallocated buffer
        np.copyto(p_tilde, cost)
        if num_conflicts > 0:
            np.subtract.at(p_tilde.ravel(), c_e1_flat, lambdas)
            np.subtract.at(p_tilde.ravel(), c_e2_flat, lambdas)

        x_star, z_star = hungarian_max(p_tilde)
        Z_Lag = z_star + float(np.sum(lambdas))
        UB_current = Z_Lag
        if UB_current < UB_best:
            UB_best = UB_current
        UB = UB_best

        # Check feasibility of subproblem solution (reuse preallocated buffer)
        asgn_flat.fill(False)
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

            # Complementary slackness optimality test:
            #   for all i: lambdas[i] > 0  ⇒  active[i] is True
            # Equivalent: NOT (lambdas > 0 AND not active).
            # Done with a single boolean reduction to avoid the prior 2-step
            # `active[lambdas > 0]` indexing that allocated two temps.
            if num_conflicts > 0:
                active = asgn_flat[c_e1_flat] | asgn_flat[c_e2_flat]
                slackness_ok = not bool(((lambdas > 0) & ~active).any())
            else:
                slackness_ok = True

            if slackness_ok:
                if obj > LB:
                    LB = obj
                    x_LB = list(x_star)
                    feasible_found = True
                if verbose:
                    print(f"  Iter {k}: feasible & complementary slackness → optimum")
                terminated_reason = "complementary_slackness"
                _record_iter()
                break

            if obj > LB:
                LB = obj
                x_LB = list(x_star)
                feasible_found = True
                t_no_improve = 0
            else:
                t_no_improve += 1
        else:
            # Infeasible subproblem solution: try to repair.
            # Pass current `lambdas` so lambda-aware heuristics can use the
            # *evolving* dual signal (not just the converged final values),
            # and `graph_edges` so the repair stays inside the underlying
            # sparse graph for instances with alpha < 1.
            if repair_fn is not None:
                try:
                    x_hat, z_hat, feasible = repair_fn(
                        x_star, cost, conflicts, n, E0,
                        lambdas=lambdas, graph_edges=graph_edges,
                        neighbours=neighbours,
                    )
                except TypeError:
                    try:
                        x_hat, z_hat, feasible = repair_fn(
                            x_star, cost, conflicts, n, E0
                        )
                    except TypeError:
                        # Oldest signature (no E0, no lambdas).
                        x_hat, z_hat, feasible = repair_fn(
                            x_star, cost, conflicts, n
                        )
                if feasible and z_hat > LB:
                    LB = float(z_hat)
                    x_LB = list(x_hat)
                    feasible_found = True
                    t_no_improve = 0
                else:
                    t_no_improve += 1
            else:
                t_no_improve += 1

        # Halve step length after stagnant iterations
        if t_no_improve >= config.SUBG_STAGNATION_LIMIT:
            t_no_improve = 0
            pi_k /= 2.0

        # Subgradient direction (in-place into preallocated s_buf) and multiplier update
        if num_conflicts > 0:
            # s_buf = a1.astype(float32) + a2.astype(float32) - 1
            s_buf[:] = asgn_flat[c_e1_flat]   # bool → float32 via assignment
            s_buf += asgn_flat[c_e2_flat]     # in-place add (bool → float32)
            s_buf -= 1.0                      # in-place subtract
            s_norm_sq = float(np.dot(s_buf, s_buf))
            if s_norm_sq < epsilon:
                if verbose:
                    print(f"  Iter {k}: subgradient norm below tolerance")
                terminated_reason = "small_subgradient"
                _record_iter()
                break
            alpha = np.float32(pi_k * (Z_Lag - LB) / s_norm_sq)
            # In-place update: lambdas += alpha * s_buf; clip to [0, ∞)
            s_buf *= alpha
            lambdas += s_buf
            np.maximum(lambdas, 0, out=lambdas)

        if verbose and (k % 50 == 0 or k <= 5):
            gap_pct = ((UB_best - LB) / max(abs(LB), 1e-10)) * 100.0
            print(f"  Iter {k:4d}: LB = {LB:.2f}, UB = {UB:.2f}, "
                  f"gap = {gap_pct:.2f}%, pi = {pi_k:.6f}")

        _record_iter()

    runtime = time.time() - t_start

    # Invariant: a valid Lagrangean relaxation must satisfy LB <= UB.
    # Use a tolerance to absorb float32 roundoff in the penalty arithmetic.
    if feasible_found and LB > UB_best + 1e-3 * max(abs(UB_best), 1.0):
        print(f"  [Warning] LB ({LB:.4f}) exceeds UB ({UB_best:.4f}); "
              f"likely a numerical issue or a bug in the repair heuristic.")

    # If only E0 was ever available, report None so downstream analysis
    # excludes these runs from objective averages rather than pulling them to 0.
    if feasible_found:
        lb_out = float(LB)
        x_lb_out = [tuple(e) for e in x_LB]
        gap_pct = ((UB_best - LB) / max(abs(LB), 1e-10)) * 100.0
    else:
        lb_out = None
        x_lb_out = None
        gap_pct = None

    if verbose:
        print(f"\n  Termination: {terminated_reason}")
        if feasible_found:
            print(f"  Final LB = {LB:.2f}, UB = {UB_best:.2f}, gap = {gap_pct:.2f}%")
        else:
            print(f"  No feasible solution found beyond E0 (UB = {UB_best:.2f})")
        print(f"  Runtime = {runtime:.3f} s over {k} iterations")

    return {
        "LB": lb_out,
        "UB": float(UB_best),
        "UB_current": float(UB_current),
        "gap_pct": gap_pct,
        "feasible_found": feasible_found,
        "x_LB": x_lb_out,
        "x_star_final": [tuple(e) for e in x_star],
        "iterations": int(k),
        "lambdas_final": [float(v) for v in lambdas],
        "runtime_seconds": float(runtime),
        "terminated_reason": terminated_reason,
        "iteration_history": iteration_history,
    }


__all__ = [
    "Edge", "Assignment", "Conflict", "Instance", "RepairFn",
    "hungarian_max", "find_violations", "build_conflict_adjacency_int",
    "is_valid_assignment", "subgradient_solve",
    "save_instance", "load_instance", "save_result",
]
