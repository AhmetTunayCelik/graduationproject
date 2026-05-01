"""
heuristics/lagrangean_repair_lambda.py
======================================

Lambda-aware core-based repair heuristic for the Lagrangean relaxation of MAX-APC.

Memory-optimised:
    - Conflict adjacency: List[np.ndarray(int32)] via ab.build_conflict_adjacency_int
    - State sets are bool bitmasks of size n*n where fast membership matters
    - edge_lambda_sum is a np.ndarray(float32) of length n*n
    - Hungarian residual matrix is float32

Extends lagrangean_repair.py by incorporating dual information from the
final Lagrange multipliers (lambdas) into the ordering criteria.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

import apc_base as ab


# -----------------------------------------------------------------------------
# Ordering criteria
# -----------------------------------------------------------------------------
ORDERINGS = (
    "lambda_penalized_weight",
    "lambda_weight_over_degree",
    "lambda_over_degree_then_weight",
)

ORDERING_LABELS = {
    "lambda_penalized_weight":       "Weight - mu*lambda_sum",
    "lambda_weight_over_degree":     "Weight / (1 + degree + mu*lambda_sum)",
    "lambda_over_degree_then_weight":"Increasing lambda+degree, tie by weight",
}

DEFAULT_ORDERING = "lambda_weight_over_degree"
HEURISTIC_NAME = "lagrangean_repair_lambda"

DEFAULT_MU = 1.0


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _build_edge_lambda_sum_arr(
    conflicts: List[ab.Conflict],
    lambdas: Optional[List[float]],
    n: int,
) -> np.ndarray:
    """For each edge id, sum of lambdas of conflict constraints containing it.

    Returns a numpy float32 array of length n*n.
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


def _sort_key_id(
    ordering: str,
    eid: int,
    n: int,
    cost: np.ndarray,
    degree: int,
    lambda_sum: float,
    mu: float,
):
    """Sort key for an edge by id under the given lambda-aware ordering."""
    i, j = divmod(eid, n)
    c = float(cost[i, j])

    if ordering == "lambda_penalized_weight":
        return -(c - mu * lambda_sum)

    if ordering == "lambda_weight_over_degree":
        return -(c / (1.0 + degree + mu * lambda_sum))

    if ordering == "lambda_over_degree_then_weight":
        return (degree + mu * lambda_sum, -c)

    raise ValueError(f"Unknown ordering: {ordering}")


def _phase1_core(
    x_star: ab.Assignment,
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    edge_lambda_sum: np.ndarray,
    n: int,
    ordering: str,
    mu: float,
) -> Tuple[List[int], set, set, np.ndarray]:
    """Build a conflict-free core Q from x_star (returns int ids + bitmask)."""
    nn = n * n
    x_ids = [i * n + j for i, j in x_star]

    x_mask = np.zeros(nn, dtype=bool)
    x_mask[x_ids] = True

    degree_in_x = {}
    for eid in x_ids:
        nbrs = neighbours[eid]
        degree_in_x[eid] = int(x_mask[nbrs].sum()) if nbrs.size else 0

    sorted_ids = sorted(
        x_ids,
        key=lambda eid: _sort_key_id(
            ordering=ordering, eid=eid, n=n, cost=cost,
            degree=degree_in_x[eid],
            lambda_sum=float(edge_lambda_sum[eid]),
            mu=mu,
        ),
    )

    core_ids: List[int] = []
    rows_used: set = set()
    cols_used: set = set()
    forbidden_mask = np.zeros(nn, dtype=bool)

    for eid in sorted_ids:
        if forbidden_mask[eid]:
            continue
        i, j = divmod(eid, n)
        if i in rows_used or j in cols_used:
            continue
        core_ids.append(eid)
        rows_used.add(i)
        cols_used.add(j)
        nbrs = neighbours[eid]
        if nbrs.size:
            forbidden_mask[nbrs] = True

    return core_ids, rows_used, cols_used, forbidden_mask


def _phase2_completion(
    core_ids: List[int],
    n: int,
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    edge_lambda_sum: np.ndarray,
    rows_used: set,
    cols_used: set,
    forbidden_mask: np.ndarray,
    ordering: str,
    E0: ab.Assignment,
    mu: float,
    graph_edge_mask: np.ndarray = None,
) -> List[int]:
    """Complete the core to a full feasible assignment, returning int ids."""
    nn = n * n

    # Greedy extension
    if graph_edge_mask is not None:
        candidate_mask = graph_edge_mask & ~forbidden_mask
        row_free = np.ones(n, dtype=bool)
        col_free = np.ones(n, dtype=bool)
        for r in rows_used:
            row_free[r] = False
        for c in cols_used:
            col_free[c] = False
        candidate_mask &= np.outer(row_free, col_free).ravel()
        pool = np.flatnonzero(candidate_mask).tolist()
    else:
        pool = [
            i * n + j
            for i in range(n) if i not in rows_used
            for j in range(n) if j not in cols_used
            if not forbidden_mask[i * n + j]
        ]

    extended = list(core_ids)
    rows = set(rows_used)
    cols = set(cols_used)
    block_mask = forbidden_mask.copy()

    if pool:
        pool_mask = np.zeros(nn, dtype=bool)
        pool_mask[pool] = True
        degree_in_pool = {}
        for eid in pool:
            nbrs = neighbours[eid]
            degree_in_pool[eid] = int(pool_mask[nbrs].sum()) if nbrs.size else 0
        pool.sort(
            key=lambda eid: _sort_key_id(
                ordering=ordering, eid=eid, n=n, cost=cost,
                degree=degree_in_pool[eid],
                lambda_sum=float(edge_lambda_sum[eid]),
                mu=mu,
            )
        )

        for eid in pool:
            if len(extended) >= n:
                break
            if block_mask[eid]:
                continue
            i, j = divmod(eid, n)
            if i in rows or j in cols:
                continue
            extended.append(eid)
            rows.add(i)
            cols.add(j)
            nbrs = neighbours[eid]
            if nbrs.size:
                block_mask[nbrs] = True

        if len(extended) == n:
            return extended

    # Hungarian completion
    free_rows = sorted(set(range(n)) - rows)
    free_cols = sorted(set(range(n)) - cols)

    if len(free_rows) == len(free_cols) and len(free_rows) > 0:
        m = len(free_rows)
        MASK = -1e15
        sub_profit = np.full((m, m), MASK, dtype=np.float32)

        for a, i in enumerate(free_rows):
            for b, j in enumerate(free_cols):
                eid = i * n + j
                if graph_edge_mask is not None and not graph_edge_mask[eid]:
                    continue
                if block_mask[eid]:
                    continue
                # Lambda-aware residual scoring
                sub_profit[a, b] = cost[i, j] - mu * edge_lambda_sum[eid]

        try:
            row_ind, col_ind = linear_sum_assignment(-sub_profit)

            if all(sub_profit[row_ind[a], col_ind[a]] > MASK / 2.0 for a in range(m)):
                completion_ids = [
                    free_rows[row_ind[a]] * n + free_cols[col_ind[a]] for a in range(m)
                ]
                comp_mask = np.zeros(nn, dtype=bool)
                comp_mask[completion_ids] = True
                core_mask = np.zeros(nn, dtype=bool)
                core_mask[core_ids] = True
                ok = True
                for eid in completion_ids:
                    nbrs = neighbours[eid]
                    if nbrs.size and (comp_mask[nbrs].any() or core_mask[nbrs].any()):
                        ok = False
                        break
                if ok:
                    return extended + completion_ids
        except ValueError:
            pass

    # Fallback to E0
    return [i * n + j for i, j in E0]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def repair(
    x_star: ab.Assignment,
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    ordering: str = DEFAULT_ORDERING,
    lambdas: Optional[List[float]] = None,
    mu: float = DEFAULT_MU,
    graph_edges=None,
    neighbours=None,
) -> Tuple[ab.Assignment, float, bool]:
    """Convert a Lagrangean subproblem solution into a feasible assignment."""
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.asarray(cost_matrix, dtype=np.float32))

    if neighbours is None:
        neighbours = ab.build_conflict_adjacency_int(conflicts, n)
    edge_lambda_sum = _build_edge_lambda_sum_arr(conflicts, lambdas, n)

    graph_edge_mask = None
    if graph_edges is not None:
        graph_edge_mask = np.zeros(n * n, dtype=bool)
        for i, j in graph_edges:
            graph_edge_mask[i * n + j] = True

    core_ids, rows_used, cols_used, forbidden_mask = _phase1_core(
        x_star=x_star, cost=cost, neighbours=neighbours,
        edge_lambda_sum=edge_lambda_sum, n=n, ordering=ordering, mu=mu,
    )

    completed_ids = _phase2_completion(
        core_ids=core_ids, n=n, cost=cost, neighbours=neighbours,
        edge_lambda_sum=edge_lambda_sum,
        rows_used=rows_used, cols_used=cols_used, forbidden_mask=forbidden_mask,
        ordering=ordering, E0=E0, mu=mu, graph_edge_mask=graph_edge_mask,
    )

    assignment = sorted([divmod(eid, n) for eid in completed_ids], key=lambda e: e[0])
    feasible = ab.is_valid_assignment(assignment, conflicts, n, graph_edges)
    objective = float(sum(cost[i, j] for i, j in assignment)) if feasible else 0.0
    return assignment, objective, feasible


def run(
    x_star: ab.Assignment,
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    ordering: str = DEFAULT_ORDERING,
    lambdas: Optional[List[float]] = None,
    mu: float = DEFAULT_MU,
    graph_edges=None,
    neighbours=None,
    **kwargs,
) -> Tuple[ab.Assignment, float, bool]:
    """Standard heuristic interface expected by batch_experiment.py."""
    return repair(
        x_star=x_star, cost_matrix=cost_matrix, conflicts=conflicts,
        n=n, E0=E0, ordering=ordering, lambdas=lambdas, mu=mu,
        graph_edges=graph_edges, neighbours=neighbours,
    )


def run_all_orderings(
    x_star: ab.Assignment,
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    lambdas: Optional[List[float]] = None,
    mu: float = DEFAULT_MU,
    graph_edges=None,
    neighbours=None,
) -> Dict[str, Dict[str, Any]]:
    """Run the repair heuristic under every lambda-aware ordering criterion."""
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.asarray(cost_matrix, dtype=np.float32))
    if neighbours is None:
        neighbours = ab.build_conflict_adjacency_int(conflicts, n)
    edge_lambda_sum = _build_edge_lambda_sum_arr(conflicts, lambdas, n)

    records = {}
    for ordering in ORDERINGS:
        t0 = time.time()

        assignment, objective, feasible = repair(
            x_star=x_star, cost_matrix=cost, conflicts=conflicts,
            n=n, E0=E0, ordering=ordering, lambdas=lambdas, mu=mu,
            graph_edges=graph_edges, neighbours=neighbours,
        )

        elapsed = time.time() - t0
        core_ids, _, _, _ = _phase1_core(
            x_star=x_star, cost=cost, neighbours=neighbours,
            edge_lambda_sum=edge_lambda_sum, n=n, ordering=ordering, mu=mu,
        )

        records[ordering] = {
            "ordering": ordering,
            "ordering_label": ORDERING_LABELS[ordering],
            "objective": float(objective),
            "feasible": bool(feasible),
            "runtime_seconds": float(elapsed),
            "core_size": int(len(core_ids)),
            "assignment": [tuple(e) for e in assignment],
        }

    return records


__all__ = [
    "HEURISTIC_NAME",
    "ORDERINGS",
    "ORDERING_LABELS",
    "DEFAULT_ORDERING",
    "DEFAULT_MU",
    "repair",
    "run",
    "run_all_orderings",
]
