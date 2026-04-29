"""
heuristics/lagrangean_repair.py
================================

Core-based repair heuristic for the Lagrangean relaxation of MAX-APC.

Memory-optimised:
    - Conflict adjacency: List[np.ndarray(int32)] (built by ab.build_conflict_adjacency_int)
    - State "sets" of edge ids: bool bitmasks of size n*n (np.ndarray(bool))
        - O(1) membership via mask[eid]
        - O(|nbrs|) batch insert via mask[nbrs_arr] = True (vectorised)
        - 1 bit per slot vs ~50 bytes per Python int in a set

This combination cuts heuristic working memory by ~10× compared to the
previous tuple-keyed dict-of-sets layout.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import apc_base as ab


# -----------------------------------------------------------------------------
# Ordering criteria
# -----------------------------------------------------------------------------
ORDERINGS = (
    "dec_weight",
    "inc_weight",
    "inc_degree",
    "weight_over_degree",
)

ORDERING_LABELS = {
    "dec_weight":         "Decreasing weight",
    "inc_weight":         "Increasing weight",
    "inc_degree":         "Increasing conflict degree",
    "weight_over_degree": "Weight-over-degree ratio",
}

DEFAULT_ORDERING = "weight_over_degree"
HEURISTIC_NAME = "lagrangean_repair"


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _sort_key_id(ordering: str, eid: int, n: int, cost: np.ndarray, degree: int):
    """Sort key for an edge by integer id under the given ordering."""
    i, j = divmod(eid, n)
    c = float(cost[i, j])
    if ordering == "dec_weight":
        return -c
    if ordering == "inc_weight":
        return c
    if ordering == "inc_degree":
        return (degree, -c)
    if ordering == "weight_over_degree":
        return -(c / (1.0 + degree))
    raise ValueError(f"Unknown ordering: {ordering}")


def _phase1_core(
    x_star: ab.Assignment,
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    n: int,
    ordering: str,
) -> Tuple[List[int], set, set, np.ndarray]:
    """Build a conflict-free core Q from x_star.

    Returns (core_ids, rows_used, cols_used, forbidden_mask).
    forbidden_mask is a bool np.ndarray of size n*n.
    """
    nn = n * n
    x_ids = [i * n + j for i, j in x_star]

    # Bitmask of edges in x_star (for vectorised degree computation)
    x_mask = np.zeros(nn, dtype=bool)
    x_mask[x_ids] = True

    degree_in_x = {}
    for eid in x_ids:
        nbrs = neighbours[eid]
        degree_in_x[eid] = int(x_mask[nbrs].sum()) if nbrs.size else 0

    sorted_ids = sorted(
        x_ids,
        key=lambda eid: _sort_key_id(ordering, eid, n, cost, degree_in_x[eid]),
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
    rows_used: set,
    cols_used: set,
    forbidden_mask: np.ndarray,
    ordering: str,
    E0: ab.Assignment,
    graph_edge_mask: np.ndarray = None,
) -> List[int]:
    """Complete the core to a full feasible assignment, returning int ids.

    graph_edge_mask: bool np.ndarray of size n*n (True iff edge id present).
    None means complete graph (all edges allowed).
    """
    nn = n * n

    # ------------------------------------------------------------------
    # Greedy extension
    # ------------------------------------------------------------------
    # Build candidate pool. For sparse graphs, iterate the small id list;
    # for complete graphs, iterate free row/col combinations.
    if graph_edge_mask is not None:
        # Find edge ids present in graph but not in forbidden, with free row/col
        candidate_mask = graph_edge_mask & ~forbidden_mask
        # Filter by free rows/cols (cheap: precompute row/col masks)
        row_free = np.ones(n, dtype=bool)
        col_free = np.ones(n, dtype=bool)
        for r in rows_used:
            row_free[r] = False
        for c in cols_used:
            col_free[c] = False
        # Build a 2D mask of allowed (i, j) and AND with candidate_mask
        rc_free = np.outer(row_free, col_free).ravel()
        candidate_mask &= rc_free
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
        pool.sort(key=lambda eid: _sort_key_id(ordering, eid, n, cost, degree_in_pool[eid]))

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

    # ------------------------------------------------------------------
    # Hungarian completion on residual rows/cols
    # ------------------------------------------------------------------
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
                sub_profit[a, b] = cost[i, j]
        try:
            row_ind, col_ind = linear_sum_assignment(-sub_profit)
            if all(sub_profit[row_ind[a], col_ind[a]] > MASK / 2.0 for a in range(m)):
                completion_ids = [
                    free_rows[row_ind[a]] * n + free_cols[col_ind[a]] for a in range(m)
                ]
                # Vectorised intra/cross-conflict check via bitmasks
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
    graph_edges=None,
) -> Tuple[ab.Assignment, float, bool]:
    """Convert a Lagrangean subproblem solution into a feasible assignment."""
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.asarray(cost_matrix, dtype=np.float32))
    neighbours = ab.build_conflict_adjacency_int(conflicts, n)

    # Convert graph_edges (list/set of (i,j)) to bool mask if provided.
    graph_edge_mask = None
    if graph_edges is not None:
        graph_edge_mask = np.zeros(n * n, dtype=bool)
        for i, j in graph_edges:
            graph_edge_mask[i * n + j] = True

    core_ids, rows_used, cols_used, forbidden_mask = _phase1_core(
        x_star, cost, neighbours, n, ordering
    )
    completed_ids = _phase2_completion(
        core_ids, n, cost, neighbours,
        rows_used, cols_used, forbidden_mask, ordering, E0,
        graph_edge_mask=graph_edge_mask,
    )

    assignment = sorted([divmod(eid, n) for eid in completed_ids], key=lambda e: e[0])
    feasible = (
        len(assignment) == n
        and len(set(assignment)) == n
        and len(ab.find_violations(assignment, conflicts, n)) == 0
    )
    objective = float(sum(cost[i, j] for i, j in assignment)) if feasible else 0.0
    return assignment, objective, feasible


def run(
    x_star: ab.Assignment,
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    ordering: str = DEFAULT_ORDERING,
    graph_edges=None,
    **kwargs,
) -> Tuple[ab.Assignment, float, bool]:
    """Alias of repair() with uniform heuristic interface."""
    return repair(x_star, cost_matrix, conflicts, n, E0, ordering=ordering,
                  graph_edges=graph_edges)


def run_all_orderings(
    x_star: ab.Assignment,
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    graph_edges=None,
) -> Dict[str, Dict[str, Any]]:
    """Run the repair heuristic under every ordering criterion."""
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.asarray(cost_matrix, dtype=np.float32))
    neighbours = ab.build_conflict_adjacency_int(conflicts, n)
    records = {}

    for ordering in ORDERINGS:
        t0 = time.time()
        assignment, objective, feasible = repair(
            x_star, cost, conflicts, n, E0,
            ordering=ordering, graph_edges=graph_edges,
        )
        elapsed = time.time() - t0
        core_ids, _, _, _ = _phase1_core(x_star, cost, neighbours, n, ordering)
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
    "repair",
    "run",
    "run_all_orderings",
]
