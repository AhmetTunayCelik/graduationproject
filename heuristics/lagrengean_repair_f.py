"""
heuristics/lagrengean_repair_f.py
==================================

SAVLR-guided repair heuristic for the Lagrangean relaxation of MAX-APC.

Implements the Surrogate Absolute-Value Lagrangean Relaxation (SAVLR) and
Surrogate Level-Based Lagrangean Relaxation (SLBLR) methodology adapted
for the repair heuristic context:

1. Dynamic Level Value (q_j): Estimates an upper-bound target from the
   current Lagrangean solution to calibrate dual penalty influence.
   q_j = L(x, lambda) + (1/gamma) * ||g(x)||^2

2. Polyak-inspired scoring: The dual penalty weight is derived from the
   gap between the level value and current objective:
   w = zeta * (q_j - obj) / ||g||^2
   ||g||^2 is the full subgradient norm (including both violated and
   inactive constraint contributions).  Capped so that the dual term
   never exceeds the cost range.

3. SAVLR Penalty (rho * |g(x)|): Absolute-value penalty on constraint
   violations (measured by edge conflict degree).  rho is scaled
   relative to the cost range so the penalty has meaningful influence.

4. Multi-rho exploration: Multiple rho values are tried (each producing
   a different cost-vs-conflict trade-off).  The feasible solution with
   the highest objective is kept.

5. Selective Repair: Removes only the minimum number of violation-causing
   edges from x_star (prioritised by SAVLR score), preserving the
   high-quality conflict-free portion.

Performance-optimised:
    - Conflict flat indices (c_e1, c_e2) precomputed once per repair call
    - Adjacency list built with vectorised numpy (argsort + bincount)
    - Lambda sums accumulated with np.add.at (no Python loop)
    - Feasibility checks via bitmask (no repeated np.array(conflicts))
    - Bool bitmasks of size n*n for set operations
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import apc_base as ab


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEURISTIC_NAME = "lagrangean_repair_savlr"

_NUM_RHO_TRIALS = 5    # number of rho values to explore
_BETA           = 2.0  # escalation factor between trials: rho *= beta
_RHO_FRAC       = 0.05 # initial rho as fraction of cost range
_ZETA           = 0.8  # Polyak relaxation coefficient (< 1)
_GAMMA          = 1.0  # level-value scaling parameter

_EMPTY_I32 = np.empty(0, dtype=np.int32)

# Module-level cache: within a subgradient run conflicts & n are constant
# across all repair calls.  Caching avoids re-running the O(|C| log |C|)
# precomputation at every iteration.
_cache: dict = {"key": None, "c_e1": None, "c_e2": None, "neighbours": None}


# ---------------------------------------------------------------------------
# Fast data preparation (replaces per-call Python loops)
# ---------------------------------------------------------------------------
def _precompute_conflict_arrays(
    conflicts: List[ab.Conflict],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute flat edge-id arrays for both sides of every conflict.

    Returns (c_e1, c_e2) as int32 arrays, each of length |conflicts|.
    """
    if not conflicts:
        return _EMPTY_I32, _EMPTY_I32
    c_arr = np.asarray(conflicts, dtype=np.int32)
    c_e1 = (c_arr[:, 0] * n + c_arr[:, 1]).astype(np.int32, copy=False)
    c_e2 = (c_arr[:, 2] * n + c_arr[:, 3]).astype(np.int32, copy=False)
    return c_e1, c_e2


def _get_cached(
    conflicts: List[ab.Conflict],
    n: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Return (c_e1, c_e2, neighbours), building them only on cache miss.

    Cache key uses (id(conflicts), len(conflicts), n) to detect when
    a new instance is loaded while staying O(1) to check.
    """
    first = tuple(conflicts[0]) if conflicts else ()
    last = tuple(conflicts[-1]) if conflicts else ()
    key = (id(conflicts), len(conflicts), n, first, last)
    if _cache["key"] != key:
        c_e1, c_e2 = _precompute_conflict_arrays(conflicts, n)
        neighbours = _build_adjacency_fast(c_e1, c_e2, n)
        _cache["key"] = key
        _cache["c_e1"] = c_e1
        _cache["c_e2"] = c_e2
        _cache["neighbours"] = neighbours
    return _cache["c_e1"], _cache["c_e2"], _cache["neighbours"]


def _build_adjacency_fast(
    c_e1: np.ndarray,
    c_e2: np.ndarray,
    n: int,
) -> List[np.ndarray]:
    """Build conflict adjacency list using vectorised numpy operations.

    ~10x faster than the Python-loop version in apc_base for large |C|
    because the heavy work (concatenate, argsort, bincount) runs in C.
    """
    nn = n * n
    num_c = len(c_e1)
    if num_c == 0:
        return [_EMPTY_I32] * nn

    # Both directions: e1->e2 and e2->e1
    all_src = np.concatenate([c_e1, c_e2])
    all_dst = np.concatenate([c_e2, c_e1])

    # Sort by source edge id
    order = np.argsort(all_src, kind='mergesort')
    sorted_dst = all_dst[order].astype(np.int32, copy=False)

    # Boundaries via bincount + cumsum
    degree = np.bincount(all_src, minlength=nn).astype(np.int64)
    bounds = np.zeros(nn + 1, dtype=np.int64)
    np.cumsum(degree, out=bounds[1:])

    # Build per-edge arrays (only non-empty slots)
    adj: List[np.ndarray] = [_EMPTY_I32] * nn
    nz = np.flatnonzero(degree)
    for eid in nz:
        adj[eid] = sorted_dst[bounds[eid]:bounds[eid + 1]]

    return adj


def _build_edge_lambda_sum_fast(
    c_e1: np.ndarray,
    c_e2: np.ndarray,
    lambdas: Optional[List[float]],
    n: int,
) -> np.ndarray:
    """Per-edge sum of lambda values, fully vectorised with np.add.at."""
    arr = np.zeros(n * n, dtype=np.float32)
    if lambdas is None or len(c_e1) == 0:
        return arr
    m = min(len(c_e1), len(lambdas))
    lam_arr = np.asarray(lambdas[:m], dtype=np.float32)
    np.add.at(arr, c_e1[:m], lam_arr)
    np.add.at(arr, c_e2[:m], lam_arr)
    return arr


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
# Dual weight (Polyak-inspired, using precomputed arrays)
# ---------------------------------------------------------------------------
def _compute_dual_weight(
    x_star: ab.Assignment,
    cost: np.ndarray,
    lambdas: Optional[List[float]],
    edge_lambda_sum: np.ndarray,
    n: int,
    c_e1: np.ndarray,
    c_e2: np.ndarray,
) -> float:
    """Polyak-inspired dual penalty weight with proper ||g||^2 and capping.

    Level value:   q_j = L(x, lambda) + (1/gamma) * ||g||^2
    Polyak weight:   w = zeta * (q_j - obj) / ||g||^2

    Uses precomputed c_e1/c_e2 for the subgradient norm (no np.array).
    Capped so that  w * max(lambda_sum_per_edge) <= cost_range.
    """
    if len(c_e1) == 0:
        return 0.0

    nn = n * n
    asgn_flat = np.zeros(nn, dtype=bool)
    for i, j in x_star:
        asgn_flat[i * n + j] = True

    # Full subgradient: g_k = x_{e1} + x_{e2} - 1 in {-1, 0, +1}
    g = (asgn_flat[c_e1].astype(np.float32)
         + asgn_flat[c_e2].astype(np.float32) - 1.0)
    g_norm_sq = float(np.dot(g, g))

    if g_norm_sq < 1e-10:
        return 0.0

    obj = float(sum(cost[i, j] for i, j in x_star))
    lam_sum = (float(np.sum(np.asarray(lambdas, dtype=np.float32)))
               if (lambdas is not None and len(lambdas) > 0) else 0.0)

    q_j = obj + lam_sum + (1.0 / _GAMMA) * g_norm_sq
    gap = q_j - obj
    weight = _ZETA * gap / g_norm_sq

    # Cap: dual_weight * max(edge_lambda) <= cost_range
    # Filter out -1e15 sentinel edges (non-graph) AND E0 edges (cost=0) so that
    # tight-cost instances (e.g. degen [95,100]) get cost_range≈5, not ≈100.
    max_lam = float(edge_lambda_sum.max()) if edge_lambda_sum.any() else 0.0
    if max_lam > 0.0:
        valid = cost.ravel()[cost.ravel() > 0]
        cost_range = float(valid.max() - valid.min()) + 1.0 if valid.size > 0 else 1.0
        weight = min(weight, cost_range / max_lam)

    return max(weight, 0.0)


# ---------------------------------------------------------------------------
# Phase 1: SAVLR selective violation repair
# ---------------------------------------------------------------------------
def _phase1_selective_repair(
    x_star: ab.Assignment,
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    edge_lambda_sum: np.ndarray,
    n: int,
    rho: float,
    dual_weight: float,
) -> Tuple[List[int], set, set, np.ndarray]:
    """Remove minimum violating edges using SAVLR penalty priorities.

    SAVLR score for each edge e:
        score(e) = cost(e) - dual_weight * lambda_sum(e) - rho * degree(e)

    Iteratively removes the LOWEST-scoring violating edge until no conflicts
    remain.  After each removal, recomputes scores for affected neighbours.

    Returns (survivor_ids, rows_used, cols_used, forbidden_mask).
    """
    nn = n * n
    x_ids = [i * n + j for i, j in x_star]

    current_mask = np.zeros(nn, dtype=bool)
    current_mask[x_ids] = True

    # Conflict degree within current solution
    conflict_deg = {}
    for eid in x_ids:
        nbrs = neighbours[eid]
        conflict_deg[eid] = int(current_mask[nbrs].sum()) if nbrs.size else 0

    # Initial SAVLR scores
    scores = {}
    for eid in x_ids:
        i, j = divmod(eid, n)
        scores[eid] = (
            float(cost[i, j])
            - dual_weight * float(edge_lambda_sum[eid])
            - rho * conflict_deg[eid]
        )

    current = set(x_ids)

    while True:
        violating = [eid for eid in current if conflict_deg.get(eid, 0) > 0]
        if not violating:
            break

        # Remove edge with lowest SAVLR score among violators
        worst = min(violating, key=lambda eid: scores[eid])
        current.discard(worst)
        current_mask[worst] = False

        # Update conflict degrees and scores for affected neighbours
        for nb in neighbours[worst]:
            nb = int(nb)
            if nb in current:
                nb_nbrs = neighbours[nb]
                new_deg = (int(current_mask[nb_nbrs].sum())
                           if nb_nbrs.size else 0)
                conflict_deg[nb] = new_deg
                i_nb, j_nb = divmod(nb, n)
                scores[nb] = (
                    float(cost[i_nb, j_nb])
                    - dual_weight * float(edge_lambda_sum[nb])
                    - rho * new_deg
                )
        del conflict_deg[worst]
        del scores[worst]

    survivor_ids = list(current)
    rows_used = {eid // n for eid in survivor_ids}
    cols_used = {eid % n for eid in survivor_ids}

    forbidden_mask = np.zeros(nn, dtype=bool)
    for eid in survivor_ids:
        nbrs = neighbours[eid]
        if nbrs.size:
            forbidden_mask[nbrs] = True

    return survivor_ids, rows_used, cols_used, forbidden_mask


# ---------------------------------------------------------------------------
# Phase 2: SAVLR-scored completion
# ---------------------------------------------------------------------------
def _phase2_completion(
    survivor_ids: List[int],
    n: int,
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    edge_lambda_sum: np.ndarray,
    rows_used: set,
    cols_used: set,
    forbidden_mask: np.ndarray,
    E0: ab.Assignment,
    dual_weight: float,
    graph_edge_mask: Optional[np.ndarray] = None,
) -> List[int]:
    """Complete survivor set to a full assignment using SAVLR-scored selection.

    Greedy: score = (cost - dual_weight * lambda_sum) / (1 + pool_degree)
    Hungarian fallback: profit = cost - dual_weight * lambda_sum
    """
    nn = n * n

    # ------------------------------------------------------------------
    # Step 1: State Preservation
    # ------------------------------------------------------------------
    # Snapshot the Phase 1 base core (survivor set) BEFORE the greedy
    # extension mutates the working state. Tier 2 of the cascade rolls
    # back to this snapshot if the greedy-extended core can't be patched
    # with E0 edges.
    base_ids = list(survivor_ids)
    base_rows = set(rows_used)
    base_cols = set(cols_used)
    base_block_mask = forbidden_mask.copy()

    # ------------------------------------------------------------------
    # Step 2: Profit Maximization — Greedy extension
    # ------------------------------------------------------------------
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
        sol_mask = np.zeros(nn, dtype=bool)
        if survivor_ids:
            sol_mask[survivor_ids] = True
        pool = [
            i * n + j
            for i in range(n) if i not in rows_used
            for j in range(n) if j not in cols_used
            if not forbidden_mask[i * n + j] and not sol_mask[i * n + j]
        ]

    extended = list(survivor_ids)
    rows = set(rows_used)
    cols = set(cols_used)
    block_mask = forbidden_mask.copy()

    if pool:
        pool_mask = np.zeros(nn, dtype=bool)
        pool_mask[pool] = True
        pool_deg = {}
        for eid in pool:
            nbrs = neighbours[eid]
            pool_deg[eid] = (int(pool_mask[nbrs].sum())
                             if nbrs.size else 0)

        pool.sort(
            key=lambda eid: (
                float(cost[eid // n, eid % n])
                - dual_weight * float(edge_lambda_sum[eid])
            ) / (1.0 + pool_deg[eid]),
            reverse=True,
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

    # -- Hungarian completion on residual rows/cols ------------------------
    free_rows = sorted(set(range(n)) - rows)
    free_cols = sorted(set(range(n)) - cols)

    if len(free_rows) == len(free_cols) and free_rows:
        m = len(free_rows)
        MASK_VAL = -1e15
        sub_profit = np.full((m, m), MASK_VAL, dtype=np.float32)

        for a, i in enumerate(free_rows):
            for b, j in enumerate(free_cols):
                eid = i * n + j
                if graph_edge_mask is not None and not graph_edge_mask[eid]:
                    continue
                if block_mask[eid]:
                    continue
                sub_profit[a, b] = (
                    cost[i, j] - dual_weight * edge_lambda_sum[eid]
                )

        try:
            row_ind, col_ind = linear_sum_assignment(-sub_profit)
            if all(sub_profit[row_ind[a], col_ind[a]] > MASK_VAL / 2.0
                   for a in range(m)):
                completion_ids = [
                    free_rows[row_ind[a]] * n + free_cols[col_ind[a]]
                    for a in range(m)
                ]
                comp_mask = np.zeros(nn, dtype=bool)
                comp_mask[completion_ids] = True
                sol_mask = np.zeros(nn, dtype=bool)
                if survivor_ids:
                    sol_mask[survivor_ids] = True
                ok = True
                for eid in completion_ids:
                    nbrs = neighbours[eid]
                    if nbrs.size and (comp_mask[nbrs].any()
                                      or sol_mask[nbrs].any()):
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
    # Honest failure: every completion strategy failed. Report no feasible
    # solution rather than fabricating an E0 fallback.
    # ------------------------------------------------------------------
    return None


# ---------------------------------------------------------------------------
# Multi-rho exploration (inline feasibility via bitmask)
# ---------------------------------------------------------------------------
def _repair_multi_rho(
    x_star: ab.Assignment,
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    edge_lambda_sum: np.ndarray,
    n: int,
    E0: ab.Assignment,
    dual_weight: float,
    c_e1: np.ndarray,
    c_e2: np.ndarray,
    graph_edge_mask: Optional[np.ndarray] = None,
) -> List[int]:
    """Try multiple rho values and return the feasible solution with the
    highest objective.

    Feasibility is checked via precomputed c_e1/c_e2 bitmask operations
    (no repeated np.array(conflicts) allocation).
    """
    nn = n * n
    # Filter out -1e15 sentinels (non-graph edges) AND E0 edges (cost=0) so that
    # tight-cost instances (e.g. degen [95,100]) get cost_range≈5, not ≈100.
    valid = cost.ravel()[cost.ravel() > 0]
    cost_range = float(valid.max() - valid.min()) + 1.0 if valid.size > 0 else 1.0
    rho = cost_range * _RHO_FRAC
    num_c = len(c_e1)

    best_ids = None
    best_obj = -np.inf

    # Reusable buffer for feasibility bitmask
    asgn_buf = np.zeros(nn, dtype=bool)

    for _ in range(_NUM_RHO_TRIALS):
        survivor_ids, rows_used, cols_used, forbidden_mask = \
            _phase1_selective_repair(
                x_star, cost, neighbours, edge_lambda_sum,
                n, rho, dual_weight,
            )
        completed_ids = _phase2_completion(
            survivor_ids, n, cost, neighbours, edge_lambda_sum,
            rows_used, cols_used, forbidden_mask, E0, dual_weight,
            graph_edge_mask=graph_edge_mask,
        )

        # Phase 2 returns None when every completion strategy failed; skip
        # this rho trial and try the next one.
        if completed_ids is None:
            rho *= _BETA
            continue

        # Inline feasibility check (no find_violations call). Track best
        # feasible (incl. obj=0 if a legitimate all-cost-0 completion arises).
        rows_b = {eid // n for eid in completed_ids}
        cols_b = {eid % n for eid in completed_ids}
        if len(completed_ids) == n and len(rows_b) == n and len(cols_b) == n:
            asgn_buf.fill(False)
            for eid in completed_ids:
                asgn_buf[eid] = True
            has_conflict = (
                num_c > 0
                and (asgn_buf[c_e1] & asgn_buf[c_e2]).any()
            )
            if not has_conflict:
                obj = 0.0
                for eid in completed_ids:
                    obj += cost[eid // n, eid % n]
                obj = float(obj)
                if best_ids is None or obj > best_obj:
                    best_obj = obj
                    best_ids = list(completed_ids)

        rho *= _BETA

    # best_ids is None when no rho trial produced a valid feasible.
    # Honest failure: no E0 fallback.
    return best_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def repair(
    x_star: ab.Assignment,
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    lambdas: Optional[List[float]] = None,
    graph_edges=None,
) -> Tuple[ab.Assignment, float, bool]:
    """Convert a Lagrangean subproblem solution into a feasible assignment
    using SAVLR-guided repair with multi-rho exploration."""
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.asarray(cost_matrix, dtype=np.float32))

    # --- Cached: conflict arrays + adjacency (same across subgradient run) ---
    c_e1, c_e2, neighbours = _get_cached(conflicts, n)

    # --- Vectorised lambda sums (changes every iteration, not cached) ---
    edge_lambda_sum = _build_edge_lambda_sum_fast(c_e1, c_e2, lambdas, n)

    graph_edge_mask = None
    if graph_edges is not None:
        graph_edge_mask = np.zeros(n * n, dtype=bool)
        for i, j in graph_edges:
            graph_edge_mask[i * n + j] = True

    # --- Dual weight (uses precomputed c_e1/c_e2, no np.array) ---
    dual_weight = _compute_dual_weight(
        x_star, cost, lambdas, edge_lambda_sum, n, c_e1, c_e2,
    )

    # --- Multi-rho exploration (inline bitmask feasibility) ---
    completed_ids = _repair_multi_rho(
        x_star, cost, neighbours, edge_lambda_sum, n, E0,
        dual_weight, c_e1, c_e2, graph_edge_mask=graph_edge_mask,
    )

    # _repair_multi_rho returns None when no rho trial produced a valid
    # feasible. Report honest failure rather than an E0 fabrication.
    if completed_ids is None:
        return None, None, False

    # --- Final result (inline feasibility, no find_violations) ---
    nn = n * n
    assignment = sorted(
        [divmod(eid, n) for eid in completed_ids], key=lambda e: e[0]
    )
    asgn_flat = np.zeros(nn, dtype=bool)
    for eid in completed_ids:
        asgn_flat[eid] = True
    rows_set = {e[0] for e in assignment}
    cols_set = {e[1] for e in assignment}
    feasible = (
        len(assignment) == n
        and len(rows_set) == n
        and len(cols_set) == n
        and (len(c_e1) == 0 or not (asgn_flat[c_e1] & asgn_flat[c_e2]).any())
        and (graph_edge_mask is None or not asgn_flat[~graph_edge_mask].any())
    )
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
    lambdas: Optional[List[float]] = None,
    graph_edges=None,
    **kwargs,
) -> Tuple[ab.Assignment, float, bool]:
    """Standard heuristic interface expected by batch_experiment.py."""
    return repair(
        x_star=x_star, cost_matrix=cost_matrix, conflicts=conflicts,
        n=n, E0=E0, lambdas=lambdas, graph_edges=graph_edges,
    )


__all__ = [
    "HEURISTIC_NAME",
    "repair",
    "run",
]
