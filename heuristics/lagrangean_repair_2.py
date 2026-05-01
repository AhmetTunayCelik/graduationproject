"""
heuristics/lagrangean_repair_2.py
==================================

Conflict-elimination repair heuristic for the Lagrangean relaxation of MAX-APC.

Memory-optimised:
    - Conflict adjacency: List[np.ndarray(int32)] via ab.build_conflict_adjacency_int
    - State sets are bool bitmasks of size n*n where fast membership matters
    - Hungarian residual matrix is float32

This heuristic is complementary to lagrangean_repair.py:

    lagrangean_repair   : "keep the good" — greedy core construction.
    lagrangean_repair_2 : "remove the bad" — eliminate worst offenders, refill.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import apc_base as ab

HEURISTIC_NAME = "lagrangean_repair_2"


# ---------------------------------------------------------------------------
# Helpers (all operate on int edge ids: id = i * n + j)
# ---------------------------------------------------------------------------
def _elim_key_id(eid: int, n: int, cost: np.ndarray, active_deg: int) -> float:
    """Elimination sort key (descending). score = active_degree / weight."""
    i, j = divmod(eid, n)
    w = float(cost[i, j])
    if w <= 0.0:
        w = 0.01
    return active_deg / w


def _completion_key_id(eid: int, n: int, cost: np.ndarray, pool_degree: int) -> float:
    """Completion sort key (descending). score = weight / (1 + pool_degree)."""
    i, j = divmod(eid, n)
    w = float(cost[i, j])
    return w / (1.0 + pool_degree)


# ---------------------------------------------------------------------------
# Phase 1: conflict elimination
# ---------------------------------------------------------------------------
def _phase1_eliminate(
    x_star: ab.Assignment,
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    n: int,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Remove the most conflicting edges from x_star until no violations remain.

    Returns
    -------
    survivors : set of edge ids remaining after elimination
    rows_used : set of row indices used by survivors
    cols_used : set of col indices used by survivors
    """
    nn = n * n
    current: Set[int] = {i * n + j for i, j in x_star}
    current_mask = np.zeros(nn, dtype=bool)
    for eid in current:
        current_mask[eid] = True

    # Initial active degree per edge in current solution (vectorised)
    active_deg: Dict[int, int] = {}
    for eid in current:
        nbrs = neighbours[eid]
        active_deg[eid] = int(current_mask[nbrs].sum()) if nbrs.size else 0

    while True:
        violating = [eid for eid in current if active_deg[eid] > 0]
        if not violating:
            break

        violating.sort(key=lambda eid: _elim_key_id(eid, n, cost, active_deg[eid]),
                       reverse=True)
        worst = violating[0]

        current.remove(worst)
        current_mask[worst] = False

        # Recompute active degree for surviving conflict-neighbours of `worst`
        for nb in neighbours[worst]:
            nb = int(nb)
            if nb in current:
                nb_nbrs = neighbours[nb]
                active_deg[nb] = int(current_mask[nb_nbrs].sum()) if nb_nbrs.size else 0
        del active_deg[worst]

    rows_used = {eid // n for eid in current}
    cols_used = {eid % n for eid in current}
    return current, rows_used, cols_used


# ---------------------------------------------------------------------------
# Phase 2: greedy completion
# ---------------------------------------------------------------------------
def _phase2_complete(
    survivors: Set[int],
    n: int,
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    rows_used: Set[int],
    cols_used: Set[int],
    E0: ab.Assignment,
    graph_edge_mask: Optional[np.ndarray] = None,
) -> List[int]:
    """Fill the gaps left by elimination to reach a full assignment of size n."""
    nn = n * n
    solution: List[int] = list(survivors)
    rows = set(rows_used)
    cols = set(cols_used)

    # Bitmasks for fast membership tests
    solution_mask = np.zeros(nn, dtype=bool)
    if solution:
        solution_mask[solution] = True

    forbidden_mask = np.zeros(nn, dtype=bool)
    for eid in solution:
        nbrs = neighbours[eid]
        if nbrs.size:
            forbidden_mask[nbrs] = True

    if len(solution) == n:
        return solution

    # ------------------------------------------------------------------
    # Build initial candidate pool
    # ------------------------------------------------------------------
    if graph_edge_mask is not None:
        candidate_mask = graph_edge_mask & ~forbidden_mask & ~solution_mask
        row_free = np.ones(n, dtype=bool)
        col_free = np.ones(n, dtype=bool)
        for r in rows:
            row_free[r] = False
        for c in cols:
            col_free[c] = False
        candidate_mask &= np.outer(row_free, col_free).ravel()
        pool = np.flatnonzero(candidate_mask).tolist()
    else:
        pool = [
            i * n + j
            for i in range(n) if i not in rows
            for j in range(n) if j not in cols
            if not forbidden_mask[i * n + j] and not solution_mask[i * n + j]
        ]

    if pool:
        pool_mask = np.zeros(nn, dtype=bool)
        pool_mask[pool] = True
        pool_deg: Dict[int, int] = {}
        for eid in pool:
            nbrs = neighbours[eid]
            pool_deg[eid] = int(pool_mask[nbrs].sum()) if nbrs.size else 0
        pool.sort(key=lambda eid: _completion_key_id(eid, n, cost, pool_deg[eid]),
                  reverse=True)

        for eid in pool:
            if len(solution) >= n:
                break
            if forbidden_mask[eid]:
                continue
            i, j = divmod(eid, n)
            if i in rows or j in cols:
                continue
            solution.append(eid)
            rows.add(i)
            cols.add(j)
            nbrs = neighbours[eid]
            if nbrs.size:
                forbidden_mask[nbrs] = True

    if len(solution) == n:
        return solution

    # ------------------------------------------------------------------
    # Hungarian fallback on residual rows / cols
    # ------------------------------------------------------------------
    free_rows = sorted(set(range(n)) - rows)
    free_cols = sorted(set(range(n)) - cols)

    if len(free_rows) == len(free_cols) and free_rows:
        m = len(free_rows)
        MASK = -1e15
        sub = np.full((m, m), MASK, dtype=np.float32)
        for a, i in enumerate(free_rows):
            for b, j in enumerate(free_cols):
                eid = i * n + j
                if graph_edge_mask is not None and not graph_edge_mask[eid]:
                    continue
                if forbidden_mask[eid]:
                    continue
                sub[a, b] = cost[i, j]
        try:
            ri, ci = linear_sum_assignment(-sub)
            if all(sub[ri[a], ci[a]] > MASK / 2.0 for a in range(m)):
                completion = [free_rows[ri[a]] * n + free_cols[ci[a]] for a in range(m)]
                comp_mask = np.zeros(nn, dtype=bool)
                comp_mask[completion] = True
                sol_mask = np.zeros(nn, dtype=bool)
                if solution:
                    sol_mask[solution] = True
                ok = True
                for eid in completion:
                    nbrs = neighbours[eid]
                    if nbrs.size and (comp_mask[nbrs].any() or sol_mask[nbrs].any()):
                        ok = False
                        break
                if ok:
                    return solution + completion
        except ValueError:
            pass

    # Fallback: E0
    return [i * n + j for i, j in E0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def repair(
    x_star: ab.Assignment,
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    graph_edges=None,
    neighbours=None,
) -> Tuple[ab.Assignment, float, bool]:
    """Convert a Lagrangean subproblem solution into a feasible assignment."""
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.asarray(cost_matrix, dtype=np.float32))
    if neighbours is None:
        neighbours = ab.build_conflict_adjacency_int(conflicts, n)

    graph_edge_mask = None
    if graph_edges is not None:
        graph_edge_mask = np.zeros(n * n, dtype=bool)
        for i, j in graph_edges:
            graph_edge_mask[i * n + j] = True

    survivors, rows_used, cols_used = _phase1_eliminate(x_star, cost, neighbours, n)
    completed_ids = _phase2_complete(
        survivors, n, cost, neighbours,
        rows_used, cols_used, E0,
        graph_edge_mask=graph_edge_mask,
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
    graph_edges=None,
    neighbours=None,
    **kwargs,
) -> Tuple[ab.Assignment, float, bool]:
    """Uniform heuristic interface — alias of repair()."""
    return repair(x_star, cost_matrix, conflicts, n, E0,
                  graph_edges=graph_edges, neighbours=neighbours)


__all__ = [
    "HEURISTIC_NAME",
    "repair",
    "run",
]
