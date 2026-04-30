"""
heuristics/lagrangean_iterative_tabu.py
========================================

Iterative Hungarian with Dynamic Tabu Penalties for the Lagrangean
relaxation of MAX-APC.

Strategy
--------
Unlike the existing core-based or elimination-based repair heuristics,
this approach never makes greedy edge-level decisions. Instead, it
delegates *every* assignment decision to `scipy.optimize.linear_sum_assignment`
on a flattened, dynamically-penalised cost vector:

    1. Apply dual information: subtract `mu * sum(lambdas_e)` from each edge
       (when lambdas are provided by the subgradient loop).
    2. Solve the assignment problem.
    3. Identify violating edge IDs via a bool bitmask probe against the
       per-edge conflict-neighbour list.
    4. If feasible: return with the *original* (un-penalised) objective.
       Else: subtract a massive `TABU_PENALTY` from each violating edge ID
       and re-solve.
    5. Repeat up to `max_micro_iters` times. Hungarian's optimality on the
       penalised matrix forces it to route around tabooed edges, while the
       intact rest of the matrix continues to optimise true profit.

Memory + performance contract (consistent with the rest of the codebase)
------------------------------------------------------------------------
- Edge state stored as integer IDs `eid = i * n + j`. No tuple-keyed dicts.
- Conflict adjacency built once via `ab.build_conflict_adjacency_int`,
  i.e. `List[np.ndarray(int32)]` (~10x lighter than Set[int]).
- All cost / penalty arithmetic in `float32`.
- The working cost vector and assignment bitmask are pre-allocated.
- Violation detection uses a single bool bitmask of size n*n indexed by
  the precomputed neighbour arrays.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import apc_base as ab


HEURISTIC_NAME = "lagrangean_iterative_tabu"
DEFAULT_MU = 1.0
DEFAULT_MAX_MICRO_ITERS = 5
# Two orders of magnitude above the [1, 100] cost range, plus an extra
# buffer for accumulated lambda penalties: large enough that Hungarian
# strongly prefers any feasible alternative routing.
TABU_PENALTY = 10000.0


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _build_edge_lambda_sum_arr(
    conflicts: List[ab.Conflict],
    lambdas: Optional[List[float]],
    n: int,
) -> np.ndarray:
    """For each edge id, sum of lambdas of conflict constraints containing it.

    Returns a numpy float32 array of length n*n. Mirrors the helper in
    lagrangean_lambda.py so dual penalties are applied identically.
    """
    arr = np.zeros(n * n, dtype=np.float32)
    if lambdas is None:
        return arr
    m = min(len(conflicts), len(lambdas))
    for idx in range(m):
        c = conflicts[idx]
        lam = float(lambdas[idx])
        arr[c[0] * n + c[1]] += lam
        arr[c[2] * n + c[3]] += lam
    return arr


def _find_violating_eids(
    asgn_eids: np.ndarray,
    asgn_mask: np.ndarray,
    neighbours: List[np.ndarray],
) -> List[int]:
    """Return the subset of assigned edge IDs that conflict with at least
    one other assigned edge.

    Uses the assignment bitmask directly: for each assigned eid, peek at
    `asgn_mask[neighbours[eid]]` — a vectorised lookup over int32 indices.
    """
    violating: List[int] = []
    for eid in asgn_eids:
        eid_int = int(eid)
        nbrs = neighbours[eid_int]
        if nbrs.size and asgn_mask[nbrs].any():
            violating.append(eid_int)
    return violating


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def run(
    x_star,
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    lambdas: Optional[List[float]] = None,
    mu: float = DEFAULT_MU,
    max_micro_iters: int = DEFAULT_MAX_MICRO_ITERS,
    **kwargs,
) -> Tuple[ab.Assignment, float, bool]:
    """Iterative Hungarian with dynamic tabu penalties.

    Parameters
    ----------
    x_star
        Ignored — this heuristic re-solves the assignment from scratch.
        Accepted for interface compatibility with `subgradient_solve`.
    cost_matrix
        n x n matrix of edge profits (list-of-lists or ndarray).
    conflicts
        List of conflict pairs `[i1, j1, i2, j2]`.
    n
        Problem size.
    E0
        Seed feasible assignment, used as the fallback if the tabu loop
        cannot eliminate all conflicts within `max_micro_iters`.
    lambdas
        Current Lagrange multipliers (passed by `subgradient_solve` each
        iteration). When None, no dual penalty is applied.
    mu
        Scaling factor for the lambda penalty term.
    max_micro_iters
        Tabu micro-iterations before bailing to E0.

    Returns
    -------
    (assignment, objective, feasible)
    """
    # Original cost matrix in float32 for objective accounting.
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.asarray(cost_matrix, dtype=np.float32))
    cost = cost.astype(np.float32, copy=False)

    nn = n * n
    neighbours = ab.build_conflict_adjacency_int(conflicts, n)

    # 1D working cost vector — penalties are applied to flat edge IDs.
    C_working = cost.ravel().astype(np.float32, copy=True)

    # Apply Lagrangean dual signal (if any) before the first solve.
    if lambdas is not None and conflicts:
        edge_lambda_sum = _build_edge_lambda_sum_arr(conflicts, lambdas, n)
        # In-place: C_working -= mu * edge_lambda_sum
        if mu != 1.0:
            C_working -= np.float32(mu) * edge_lambda_sum
        else:
            C_working -= edge_lambda_sum

    # Pre-allocated assignment bitmask (re-used every micro-iteration).
    asgn_mask = np.zeros(nn, dtype=bool)

    for _ in range(max_micro_iters):
        # Step 4: solve assignment on the current penalised matrix.
        C2D = C_working.reshape(n, n)
        row_ind, col_ind = linear_sum_assignment(-C2D)
        asgn_eids = (row_ind.astype(np.int32) * n
                     + col_ind.astype(np.int32))

        # Refresh the bitmask in-place (cheaper than reallocating each loop).
        asgn_mask.fill(False)
        asgn_mask[asgn_eids] = True

        # Step 5: enumerate violating edge IDs.
        violating = _find_violating_eids(asgn_eids, asgn_mask, neighbours)

        if not violating:
            # Feasible: report the *original* (un-penalised) objective.
            assignment = sorted(
                [(int(r), int(c)) for r, c in zip(row_ind, col_ind)],
                key=lambda e: e[0],
            )
            objective = float(sum(cost[i, j] for i, j in assignment))
            return assignment, objective, True

        # Step 6: stamp tabu penalty onto every violating edge ID.
        # Subtraction (rather than absolute set) lets penalties compound if
        # the same edge keeps getting picked across micro-iterations.
        idx = np.fromiter(violating, dtype=np.int32, count=len(violating))
        C_working[idx] -= np.float32(TABU_PENALTY)

    # Step 8: fallback. E0 is feasible by instance construction; verify
    # defensively in case of a corrupted instance.
    e0_assignment = sorted([(int(i), int(j)) for i, j in E0],
                           key=lambda e: e[0])
    feasible = len(ab.find_violations(e0_assignment, conflicts, n)) == 0
    objective = (float(sum(cost[i, j] for i, j in e0_assignment))
                 if feasible else 0.0)
    return e0_assignment, objective, feasible


__all__ = [
    "HEURISTIC_NAME",
    "DEFAULT_MU",
    "DEFAULT_MAX_MICRO_ITERS",
    "TABU_PENALTY",
    "run",
]
