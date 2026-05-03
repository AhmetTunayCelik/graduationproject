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
from typing import Any, Dict, List, Optional, Tuple

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

        # Defensive: a row may have been re-added redundantly; skip if
        # patched_set already covers it.
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

        # Find blockers:
        #   (a) same-column collision (at most one edge per column)
        #   (b) conflict-pair (any neighbour of eid that is in patched_set)
        blockers = set()
        for b in patched_set:
            if b % n == j and b != eid:
                blockers.add(b)
                break  # column hosts at most one edge
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

        # Commit: evict blockers, queue their rows for E0 fill, insert eid
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
    # Step 1: State Preservation
    # ------------------------------------------------------------------
    # Snapshot the Phase 1 base core BEFORE the greedy extension mutates
    # the working state. Tier 2 of the cascade rolls back to this snapshot
    # if the greedy-extended core can't be patched with E0 edges.
    base_ids = list(core_ids)
    base_rows = set(rows_used)
    base_cols = set(cols_used)
    base_block_mask = forbidden_mask.copy()

    # ------------------------------------------------------------------
    # Step 2: Profit Maximization — Greedy extension
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

    # ------------------------------------------------------------------
    # Step 3: Tier 1 Fallback — Patch the EXTENDED core with safe E0 edges
    # ------------------------------------------------------------------
    # Hungarian failed, threw ValueError, or introduced a conflict.
    # Preserve the high-profit greedy-extended core and try to cover the
    # remaining rows with non-conflicting E0 edges.
    patched = _safe_patch_with_e0(
        extended, rows, cols, block_mask, neighbours, E0, n, graph_edge_mask,
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
    # Step 5: Tier 3 Fallback — Full E0 seed assignment
    # ------------------------------------------------------------------
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
    neighbours=None,
) -> Tuple[ab.Assignment, float, bool]:
    """Convert a Lagrangean subproblem solution into a feasible assignment."""
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.asarray(cost_matrix, dtype=np.float32))
    if neighbours is None:
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
    graph_edges=None,
    neighbours=None,
    **kwargs,
) -> Tuple[ab.Assignment, float, bool]:
    """Alias of repair() with uniform heuristic interface."""
    return repair(x_star, cost_matrix, conflicts, n, E0, ordering=ordering,
                  graph_edges=graph_edges, neighbours=neighbours)


def run_all_orderings(
    x_star: ab.Assignment,
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    graph_edges=None,
    neighbours=None,
    **kwargs,   # absorb lambdas/mu when invoked from lambda-aware paths
) -> Dict[str, Dict[str, Any]]:
    """Run the repair heuristic under every ordering criterion."""
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.asarray(cost_matrix, dtype=np.float32))
    if neighbours is None:
        neighbours = ab.build_conflict_adjacency_int(conflicts, n)
    records = {}

    for ordering in ORDERINGS:
        t0 = time.time()
        assignment, objective, feasible = repair(
            x_star, cost, conflicts, n, E0,
            ordering=ordering, graph_edges=graph_edges, neighbours=neighbours,
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
