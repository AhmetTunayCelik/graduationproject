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
# Safe Partial E0 Patching helper
# ---------------------------------------------------------------------------
def _safe_patch_with_e0(
    extended: List[int],
    rows: set,
    cols: set,
    block_mask: np.ndarray,
    neighbours: List[np.ndarray],
    E0: ab.Assignment,
    n: int,
    graph_edge_mask: Optional[np.ndarray] = None,
) -> Optional[List[int]]:
    """Safe Partial E0 Patching: extend a conflict-free partial core to a
    full size-n assignment using E0 edges to cover the missing rows.

    E0 is a perfect matching of zero-cost edges, internally conflict-free
    by construction. Individual E0 edges MAY conflict with non-E0 edges
    in `extended`, so each candidate E0 edge is screened against the
    existing forbidden mask, the column-occupancy set, and (if provided)
    the graph_edge_mask before being accepted.

    Returns the patched edge-id list of length n on success, or None if
    any required E0 edge cannot be safely added (caller falls back to
    the full E0 assignment in that case).
    """
    e0_row2col = {i: j for i, j in E0}

    patched = list(extended)
    patched_rows = set(rows)
    patched_cols = set(cols)
    patched_block = block_mask.copy()

    for i in range(n):
        if i in patched_rows:
            continue
        j = e0_row2col.get(i)
        if j is None or j in patched_cols:
            return None
        eid = i * n + j
        if graph_edge_mask is not None and not graph_edge_mask[eid]:
            return None
        if patched_block[eid]:
            return None
        patched.append(eid)
        patched_rows.add(i)
        patched_cols.add(j)
        nbrs = neighbours[eid]
        if nbrs.size:
            patched_block[nbrs] = True

    if len(patched) == n:
        return patched
    return None


def _dynamic_evict_and_patch_with_e0(
    base_ids: List[int],
    base_rows: set,
    base_cols: set,
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    E0: ab.Assignment,
    n: int,
    graph_edge_mask: Optional[np.ndarray] = None,
) -> Optional[List[int]]:
    """Tier 2.5 Cost-Aware Dynamic Eviction Patch (bounded augmenting path).

    Starting from the Phase 1 base core, forcibly insert E0 edges to
    cover missing rows. When an E0 edge collides with an edge already
    in `patched_set` (same-column or conflict-pair), evict the
    blocker(s) and re-queue their rows so they too get filled by E0.

    The augmenting path is COST-BOUNDED: a running objective tracks the
    total non-E0 profit remaining in `patched_set`. Each eviction
    subtracts the blocker's cost from the running objective. The moment
    the running objective drops to zero or below, the cascade is aborted
    — no point continuing once we've lost more value than we'd save by
    avoiding the full-E0 fallback.

    A 3n loop guard breaks pathological cycles (e.g. when the
    graph_edge_mask makes some E0 edges unusable).

    Returns:
        Sorted-by-row list of n edge IDs forming a feasible
        conflict-free assignment with strictly positive non-E0 profit.
        Returns None on cost-bound abort, graph-violation, or cycle abort.
    """
    e0_row2col = {i: j for i, j in E0}
    patched_set = set(base_ids)
    queue = [i for i in range(n) if i not in base_rows]

    # Initial running objective = total cost of the Phase 1 base core.
    # E0 edges have cost 0 by construction, so adding them never grows
    # the objective; only evictions subtract from it.
    current_obj = float(sum(cost[eid // n, eid % n] for eid in patched_set))

    loop_counter = 0
    max_loops = 3 * n

    while queue:
        loop_counter += 1
        if loop_counter > max_loops:
            return None

        row = queue.pop(0)

        already_covered = False
        for e in patched_set:
            if e // n == row:
                already_covered = True
                break
        if already_covered:
            continue

        j = e0_row2col.get(row)
        if j is None:
            return None
        eid = row * n + j

        if graph_edge_mask is not None and not graph_edge_mask[eid]:
            return None

        blockers = set()
        for b in patched_set:
            if b % n == j and b != eid:
                blockers.add(b)
                break
        nbrs = neighbours[eid]
        if nbrs.size:
            for b in nbrs:
                b_int = int(b)
                if b_int in patched_set:
                    blockers.add(b_int)

        # Cost-aware bound: subtract evicted profit BEFORE committing
        # to the eviction. If the cascade has destroyed all profit, the
        # full-E0 fallback is at least as good — abort now.
        for b in blockers:
            current_obj -= float(cost[b // n, b % n])
        if current_obj <= 0.0:
            return None

        for b in blockers:
            patched_set.discard(b)
            queue.append(b // n)

        patched_set.add(eid)

    # Final safety check (redundant given the in-loop abort, but cheap).
    if current_obj <= 0.0:
        return None

    if len(patched_set) != n:
        return None

    return sorted(patched_set, key=lambda e: e // n)


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
    # Step 1: State Preservation
    # ------------------------------------------------------------------
    # Snapshot the Phase 1 base core (survivor set) BEFORE the greedy
    # extension mutates `solution`/`rows`/`cols`/`forbidden_mask`. Tier 2
    # of the cascade rolls back to this snapshot if the greedy-extended
    # core can't be patched with E0 edges.
    base_ids = list(solution)
    base_rows = set(rows)
    base_cols = set(cols)
    base_block_mask = forbidden_mask.copy()

    # ------------------------------------------------------------------
    # Step 2: Profit Maximization — Build initial candidate pool
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

    # ------------------------------------------------------------------
    # Step 3: Tier 1 Fallback — Patch the EXTENDED core with safe E0 edges
    # ------------------------------------------------------------------
    # Hungarian failed, threw ValueError, or introduced a conflict.
    # Preserve the high-profit greedy-extended `solution` and try to
    # cover the remaining rows with non-conflicting E0 edges.
    patched = _safe_patch_with_e0(
        solution, rows, cols, forbidden_mask, neighbours, E0, n, graph_edge_mask,
    )
    if patched is not None:
        return patched

    # ------------------------------------------------------------------
    # Step 4: Tier 2 Fallback — Roll back to BASE Phase 1 core, patch with E0
    # ------------------------------------------------------------------
    # The greedy extension may have triggered the "n-1 trap" — it filled
    # so many rows/cols that the single remaining slot can't accept E0's
    # column. Rolling back to the smaller Phase 1 base core leaves more
    # rows/cols open, so E0 has a much higher probability of aligning.
    patched = _safe_patch_with_e0(
        base_ids, base_rows, base_cols, base_block_mask,
        neighbours, E0, n, graph_edge_mask,
    )
    if patched is not None:
        return patched

    # ------------------------------------------------------------------
    # Step 4b: Tier 2.5 Fallback — Dynamic Eviction on BASE core
    # ------------------------------------------------------------------
    # Static patches failed. Use an augmenting-path approach: forcibly
    # insert missing E0 edges and evict any core edges that block them.
    patched = _dynamic_evict_and_patch_with_e0(
        base_ids, base_rows, base_cols, cost, neighbours, E0, n, graph_edge_mask,
    )
    if patched is not None:
        return patched

    # ------------------------------------------------------------------
    # Honest failure: every completion strategy failed. Report no feasible
    # solution rather than fabricating an E0 fallback.
    # ------------------------------------------------------------------
    return None


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

    # Phase 2 returns None when every completion strategy failed.
    if completed_ids is None:
        return None, None, False

    assignment = sorted([divmod(eid, n) for eid in completed_ids], key=lambda e: e[0])
    feasible = ab.is_valid_assignment(assignment, conflicts, n, graph_edges)
    if not feasible:
        return None, None, False
    objective = float(sum(cost[i, j] for i, j in assignment))
    return assignment, objective, True


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
