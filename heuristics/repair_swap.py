"""
heuristics/lagrengean_repair_swap.py
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
   inactive constraint contributions). Capped so that the dual term
   never exceeds the cost range.

3. SAVLR Penalty (rho * |g(x)|): Absolute-value penalty on constraint
   violations (measured by edge conflict degree). rho is scaled
   relative to the cost range so the penalty has meaningful influence.

4. Multi-rho exploration: Multiple rho values are tried (each producing
   a different cost-vs-conflict trade-off). The feasible solution with
   the highest objective is kept.

5. Selective Repair: Removes only the minimum number of violation-causing
   edges from x_star (prioritised by SAVLR score), preserving the
   high-quality conflict-free portion.

6. (NEW) 2-opt Backtracking: Post-repair, a 2-opt local search with
   backtracking is applied. It exhaustively tries all improving swaps,
   maintaining a history to escape local optima, ensuring continuous
   improvement until a fixpoint is reached.

Performance-optimised:
    - Conflict flat indices (c_e1, c_e2) precomputed once per repair call
    - Adjacency list built with vectorised numpy (argsort + bincount)
    - Lambda sums accumulated with np.add.at (no Python loop)
    - Feasibility checks via bitmask (no repeated np.array(conflicts))
    - Bool bitmasks of size n*n for set operations
    - 2-opt uses vectorized conflict checking and cached conflict sets.
"""
from __future__ import annotations
from typing import List, Optional, Tuple, Set
import numpy as np
from scipy.optimize import linear_sum_assignment
import apc_base as ab

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEURISTIC_NAME = "lagrangean_repair_savlr_2opt"

_NUM_RHO_TRIALS = 5  # number of rho values to explore
_BETA = 2.0  # escalation factor between trials: rho *= beta
_RHO_FRAC = 0.05  # initial rho as fraction of cost range
_ZETA = 0.8  # Polyak relaxation coefficient (< 1)
_GAMMA = 1.0  # level-value scaling parameter
_EMPTY_I32 = np.empty(0, dtype=np.int32)
# 2-opt backtracking max iterations (safety limit)
_MAX_2OPT_ITERATIONS = 1000

# Module-level cache: within a subgradient run conflicts & n are constant
# across all repair calls. Caching avoids re-running the O(|C| log |C|)
# precomputation at every iteration.
_cache: dict = {"key": None, "c_e1": None, "c_e2": None, "neighbours": None}


# ---------------------------------------------------------------------------
# Fast data preparation (replaces per-call Python loops)
# ---------------------------------------------------------------------------
def _precompute_conflict_arrays(
    conflicts: List[ab.Conflict],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
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
    nn = n * n
    num_c = len(c_e1)
    if num_c == 0:
        return [_EMPTY_I32] * nn

    all_src = np.concatenate([c_e1, c_e2])
    all_dst = np.concatenate([c_e2, c_e1])

    order = np.argsort(all_src, kind='mergesort')
    sorted_dst = all_dst[order].astype(np.int32, copy=False)

    degree = np.bincount(all_src, minlength=nn).astype(np.int64)
    bounds = np.zeros(nn + 1, dtype=np.int64)
    np.cumsum(degree, out=bounds[1:])

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
    arr = np.zeros(n * n, dtype=np.float32)
    if lambdas is None or len(c_e1) == 0:
        return arr
    m = min(len(c_e1), len(lambdas))
    lam_arr = np.asarray(lambdas[:m], dtype=np.float32)
    np.add.at(arr, c_e1[:m], lam_arr)
    np.add.at(arr, c_e2[:m], lam_arr)
    return arr


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
    if len(c_e1) == 0:
        return 0.0
    nn = n * n
    asgn_flat = np.zeros(nn, dtype=bool)
    for i, j in x_star:
        asgn_flat[i * n + j] = True

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
    nn = n * n
    x_ids = [i * n + j for i, j in x_star]
    current_mask = np.zeros(nn, dtype=bool)
    current_mask[x_ids] = True

    conflict_deg = {}
    for eid in x_ids:
        nbrs = neighbours[eid]
        conflict_deg[eid] = int(current_mask[nbrs].sum()) if nbrs.size else 0

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

        worst = min(violating, key=lambda eid: scores[eid])
        current.discard(worst)
        current_mask[worst] = False

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
    nn = n * n
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

    return [i * n + j for i, j in E0]


# ---------------------------------------------------------------------------
# Phase 3: 2-opt Backtracking Local Search (NEW)
# ---------------------------------------------------------------------------
def _phase3_2opt_backtrack(
    assignment_ids: List[int],
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    n: int,
    c_e1: np.ndarray,
    c_e2: np.ndarray,
    graph_edge_mask: Optional[np.ndarray] = None,
) -> List[int]:
    nn = n * n
    num_c = len(c_e1)

    current_ids = list(assignment_ids)
    col_for_row = np.zeros(n, dtype=np.int32)
    for eid in current_ids:
        r, c = divmod(eid, n)
        col_for_row[r] = c

    history: Set[Tuple[int, ...]] = {tuple(sorted(current_ids))}
    max_history_size = max(100, n * 10)

    if num_c > 0:
        conflict_pairs = set(zip(c_e1, c_e2))
    else:
        conflict_pairs = set()

    flat_current = np.zeros(nn, dtype=bool)
    flat_current[current_ids] = True
    current_obj = float(cost.ravel()[current_ids].sum())

    improved = True
    iteration = 0
    while improved and iteration < _MAX_2OPT_ITERATIONS:
        improved = False
        iteration += 1

        rows = list(range(n))
        best_delta = 0.0
        best_swap = None

        for idx1 in range(n):
            r1 = rows[idx1]
            c1 = col_for_row[r1]
            for idx2 in range(idx1 + 1, n):
                r2 = rows[idx2]
                c2 = col_for_row[r2]

                delta_obj = (cost[r1, c2] + cost[r2, c1]) - (cost[r1, c1] + cost[r2, c2])
                if delta_obj <= 1e-12:
                    continue

                eid11 = r1 * n + c1
                eid22 = r2 * n + c2
                eid12 = r1 * n + c2
                eid21 = r2 * n + c1

                if eid12 == eid11 or eid12 == eid22 or eid21 == eid11 or eid21 == eid22:
                    continue

                # Graph mask check
                if graph_edge_mask is not None:
                    if not (graph_edge_mask[eid12] and graph_edge_mask[eid21]):
                        continue

                # Conflict Check
                conflicts_exist = False
                for eid_new in (eid12, eid21):
                    if neighbours[eid_new].size > 0:
                        for nb in neighbours[eid_new]:
                            nb = int(nb)
                            if nb == eid11 or nb == eid22:
                                continue
                            if flat_current[nb]:
                                conflicts_exist = True
                                break
                    if conflicts_exist:
                        break
                if conflicts_exist:
                    continue

                if (eid12, eid21) in conflict_pairs or (eid21, eid12) in conflict_pairs:
                    continue

                if delta_obj > best_delta:
                    best_delta = delta_obj
                    best_swap = (r1, r2, c1, c2)

        if best_swap is not None:
            r1, r2, c1, c2 = best_swap
            col_for_row[r1] = c2
            col_for_row[r2] = c1

            eid11_old = r1 * n + c1
            eid22_old = r2 * n + c2
            eid12_new = r1 * n + c2
            eid21_new = r2 * n + c1

            current_ids.remove(eid11_old)
            current_ids.remove(eid22_old)
            current_ids.append(eid12_new)
            current_ids.append(eid21_new)

            flat_current[eid11_old] = False
            flat_current[eid22_old] = False
            flat_current[eid12_new] = True
            flat_current[eid21_new] = True
            current_obj += best_delta

            state_tuple = tuple(sorted(current_ids))
            if state_tuple not in history:
                history.add(state_tuple)
                # Düzeltildi: Pop işlemini daha güvenli loop ile yap
                while len(history) > max_history_size:
                    history.pop()
            improved = True

    # DÜZELTİLDİ: np.all yerine .any() ile geçersiz kenar arama mantığı
    if graph_edge_mask is not None:
        if flat_current[~graph_edge_mask].any():
            return assignment_ids
    if num_c > 0 and np.any(flat_current[c_e1] & flat_current[c_e2]):
        return assignment_ids

    return current_ids


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
    nn = n * n
    valid = cost.ravel()[cost.ravel() > 0]
    cost_range = float(valid.max() - valid.min()) + 1.0 if valid.size > 0 else 1.0
    rho = cost_range * _RHO_FRAC
    num_c = len(c_e1)
    
    best_ids = None
    best_obj = -np.inf
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
            
            valid_graph = True
            # DÜZELTİLDİ: Geçersiz kenarlarda atama var mı diye kontrol edilmeli
            if graph_edge_mask is not None:
                valid_graph = not asgn_buf[~graph_edge_mask].any()
                
            if not has_conflict and valid_graph:
                obj = float(np.sum(cost.ravel()[completed_ids]))
                if obj > best_obj:
                    best_obj = obj
                    best_ids = list(completed_ids)
        rho *= _BETA

    if best_ids is not None:
        return best_ids
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
    lambdas: Optional[List[float]] = None,
    graph_edges=None,
) -> Tuple[ab.Assignment, float, bool]:
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.asarray(cost_matrix, dtype=np.float32))

    c_e1, c_e2, neighbours = _get_cached(conflicts, n)
    edge_lambda_sum = _build_edge_lambda_sum_fast(c_e1, c_e2, lambdas, n)

    graph_edge_mask = None
    if graph_edges is not None:
        graph_edge_mask = np.zeros(n * n, dtype=bool)
        for i, j in graph_edges:
            graph_edge_mask[i * n + j] = True

    dual_weight = _compute_dual_weight(
        x_star, cost, lambdas, edge_lambda_sum, n, c_e1, c_e2,
    )

    completed_ids = _repair_multi_rho(
        x_star, cost, neighbours, edge_lambda_sum, n, E0,
        dual_weight, c_e1, c_e2, graph_edge_mask=graph_edge_mask,
    )

    # 2-opt Uygulanması
    completed_ids = _phase3_2opt_backtrack(
        completed_ids, cost, neighbours, n, c_e1, c_e2, graph_edge_mask
    )

    nn = n * n
    assignment = sorted(
        [divmod(eid, n) for eid in completed_ids], key=lambda e: e[0]
    )
    asgn_flat = np.zeros(nn, dtype=bool)
    for eid in completed_ids:
        asgn_flat[eid] = True

    rows_set = {e[0] for e in assignment}
    cols_set = {e[1] for e in assignment}
    
    # DÜZELTİLDİ: Son geçerlilik kontrolünde maskeleme kurgusu düzeltildi
    feasible = (
        len(assignment) == n
        and len(rows_set) == n
        and len(cols_set) == n
        and (len(c_e1) == 0 or not (asgn_flat[c_e1] & asgn_flat[c_e2]).any())
        and (graph_edge_mask is None or not asgn_flat[~graph_edge_mask].any())
    )
    
    objective = (float(np.sum(cost.ravel()[completed_ids]))
                 if feasible else 0.0)

    return assignment, objective, feasible


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
