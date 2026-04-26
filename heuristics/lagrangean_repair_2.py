"""
heuristics/lagrangean_repair_2.py
==================================

Conflict-elimination repair heuristic for the Lagrangean relaxation of MAX-APC.

This heuristic is complementary to lagrangean_repair.py:

    lagrangean_repair   : "keep the good" — greedily builds a conflict-free
                          core by selecting edges that cause the least trouble.

    lagrangean_repair_2 : "remove the bad" — starts from x_star and actively
                          eliminates the most conflicting edges first, then
                          fills the resulting gaps from a global candidate pool.

Algorithm
---------
Phase 1 – Conflict elimination
    1. Compute for each edge in x_star its *active conflict degree*: the number
       of conflict pairs in which BOTH endpoints are currently in the solution
       (i.e. live violations, not just potential ones).
    2. Sort x_star edges by  active_degree / weight  descending — the edge
       that causes the most violations relative to its contribution goes first.
    3. Remove the top edge, update active degrees of all its conflict
       neighbours that are still in the solution, repeat until no violations
       remain (active_degree == 0 for all remaining edges).

Phase 2 – Greedy completion
    Build a candidate pool from ALL graph edges (x_star survivors + every other
    edge in the graph) that do not use an already-occupied row/column and do
    not conflict with anything already in the solution.  Sort by
    weight / (1 + conflict_degree_in_pool) and greedily add until n edges.
    Fallback to Hungarian on residual rows/cols, then to E0 if needed.

Why this ordering?
    The elimination ordering targets edges that simultaneously have many live
    conflicts AND low weight — removing them costs little in objective value
    but clears many violations at once.  The completion ordering mirrors the
    first heuristic's best criterion (weight_over_degree).
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import apc_base as ab

HEURISTIC_NAME = "lagrangean_repair_2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_conflict_index(conflicts: List[ab.Conflict]) -> Dict[ab.Edge, Set[ab.Edge]]:
    """Map each edge → set of edges it conflicts with (both directions)."""
    idx: Dict[ab.Edge, Set[ab.Edge]] = defaultdict(set)
    for c in conflicts:
        e = (c[0], c[1])
        f = (c[2], c[3])
        idx[e].add(f)
        idx[f].add(e)
    return idx


def _active_degree(edge: ab.Edge,
                   conflict_idx: Dict[ab.Edge, Set[ab.Edge]],
                   current_set: Set[ab.Edge]) -> int:
    """Number of edges in current_set that conflict with *edge*."""
    return len(conflict_idx[edge] & current_set)


def _elim_key(edge: ab.Edge,
               cost: np.ndarray,
               active_deg: int) -> float:
    """Elimination sort key (descending → negate for min-heap style sort).

    Higher score = remove first.
    score = active_degree / weight   (most disruptive & cheapest goes first)
    Edges with weight=0 (E0 edges) get a large finite score so they are
    still considered but their weight contribution is visible.
    """
    w = float(cost[edge[0], edge[1]])
    if w <= 0.0:
        # E0 edges cost 0; if they have conflicts they should still be removable,
        # but we treat them as low-value (w → small positive) so degree dominates.
        w = 0.01
    return active_deg / w


def _completion_key(edge: ab.Edge,
                    cost: np.ndarray,
                    pool_degree: int) -> float:
    """Completion sort key (descending → negate).

    score = weight / (1 + pool_degree)   (high value, few pool conflicts)
    """
    w = float(cost[edge[0], edge[1]])
    return w / (1.0 + pool_degree)


# ---------------------------------------------------------------------------
# Phase 1: conflict elimination
# ---------------------------------------------------------------------------

def _phase1_eliminate(
    x_star: ab.Assignment,
    cost: np.ndarray,
    conflict_idx: Dict[ab.Edge, Set[ab.Edge]],
) -> Tuple[List[ab.Edge], Set[int], Set[int]]:
    """Remove the most conflicting edges from x_star until no violations remain.

    Returns
    -------
    survivors : list of edges remaining in x_star after elimination
    rows_used : set of row indices used by survivors
    cols_used : set of col indices used by survivors
    """
    current: Set[ab.Edge] = set(tuple(e) for e in x_star)

    # Compute initial active degrees
    active_deg: Dict[ab.Edge, int] = {
        e: _active_degree(e, conflict_idx, current) for e in current
    }

    while True:
        # Find all edges with at least one active conflict
        violating = [e for e in current if active_deg[e] > 0]
        if not violating:
            break  # no violations left

        # Sort by elimination key descending; pick the worst offender
        violating.sort(key=lambda e: _elim_key(e, cost, active_deg[e]), reverse=True)
        worst = violating[0]

        # Remove it
        current.remove(worst)

        # Update active degrees of its conflict neighbours still in solution
        for neighbour in conflict_idx[worst]:
            if neighbour in current:
                # Recompute from scratch is safe & simple (sets are small)
                active_deg[neighbour] = _active_degree(neighbour, conflict_idx, current)

        # No need to keep worst's entry
        del active_deg[worst]

    survivors = list(current)
    rows_used = {e[0] for e in survivors}
    cols_used = {e[1] for e in survivors}
    return survivors, rows_used, cols_used


# ---------------------------------------------------------------------------
# Phase 2: greedy completion
# ---------------------------------------------------------------------------

def _phase2_complete(
    survivors: List[ab.Edge],
    n: int,
    cost: np.ndarray,
    conflict_idx: Dict[ab.Edge, Set[ab.Edge]],
    rows_used: Set[int],
    cols_used: Set[int],
    E0: ab.Assignment,
    graph_edges: Optional[Set[ab.Edge]] = None,
) -> ab.Assignment:
    """Fill the gaps left by elimination to reach a full assignment of size n.

    Candidate pool = all graph edges (or full n×n if graph_edges is None)
    that use a free row AND free col AND do not conflict with any survivor.

    Greedy selection by weight / (1 + pool_conflict_degree), updating the
    pool and conflict degrees after each selection.

    Falls back to Hungarian on residual rows/cols, then to E0.
    """
    solution: List[ab.Edge] = list(survivors)
    solution_set: Set[ab.Edge] = set(tuple(e) for e in solution)
    rows = set(rows_used)
    cols = set(cols_used)

    # Forbidden = everything that conflicts with a survivor
    forbidden: Set[ab.Edge] = set()
    for e in solution_set:
        forbidden |= conflict_idx[e]

    if len(solution) == n:
        return solution

    # ------------------------------------------------------------------
    # Build initial candidate pool
    # ------------------------------------------------------------------
    if graph_edges is not None:
        pool = [
            e for e in graph_edges
            if e[0] not in rows
            and e[1] not in cols
            and e not in forbidden
            and e not in solution_set
        ]
    else:
        pool = [
            (i, j)
            for i in range(n) if i not in rows
            for j in range(n) if j not in cols
            if (i, j) not in forbidden and (i, j) not in solution_set
        ]

    # Conflict degree within the pool (inter-pool conflicts only,
    # since conflicts with solution are already excluded via forbidden)
    pool_set: Set[ab.Edge] = set(pool)
    pool_deg: Dict[ab.Edge, int] = {
        e: len(conflict_idx[e] & pool_set) for e in pool
    }

    # Sort descending by completion key
    pool.sort(key=lambda e: _completion_key(e, cost, pool_deg[e]), reverse=True)

    # ------------------------------------------------------------------
    # Greedy selection — lazy evaluation strategy
    #
    # We sort the pool ONCE at the start and do NOT re-sort on every
    # step. Instead we skip stale (now-invalid) entries on the fly.
    # The sort key (weight / (1 + pool_degree)) is an overestimate once
    # neighbours are removed, so we may occasionally pick a suboptimal
    # edge — but for very large conflict sets this is far faster than
    # recomputing degrees after every selection.
    # ------------------------------------------------------------------
    while pool and len(solution) < n:
        e = pool.pop(0)
        # Skip if row/col taken, or now forbidden
        if e[0] in rows or e[1] in cols:
            continue
        if e in forbidden:
            continue

        # Accept e
        solution.append(e)
        solution_set.add(e)
        rows.add(e[0])
        cols.add(e[1])

        # Update forbidden set — pool pruning done lazily via skip checks above
        forbidden |= conflict_idx[e]

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
        sub = np.full((m, m), MASK, dtype=float)
        for a, i in enumerate(free_rows):
            for b, j in enumerate(free_cols):
                e = (i, j)
                if graph_edges is not None and e not in graph_edges:
                    continue
                if e in forbidden:
                    continue
                sub[a, b] = cost[i, j]
        try:
            ri, ci = linear_sum_assignment(-sub)
            if all(sub[ri[a], ci[a]] > MASK / 2.0 for a in range(m)):
                completion = [(free_rows[ri[a]], free_cols[ci[a]]) for a in range(m)]
                comp_set = set(completion)
                sol_set = set(tuple(e) for e in solution)
                no_intra = all(not (conflict_idx[e] & comp_set) for e in completion)
                no_cross = all(not (conflict_idx[e] & sol_set) for e in completion)
                if no_intra and no_cross:
                    return solution + completion
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Final fallback: E0 (always feasible by construction)
    # ------------------------------------------------------------------
    return list(E0)


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
) -> Tuple[ab.Assignment, float, bool]:
    """Convert a Lagrangean subproblem solution into a feasible assignment.

    Strategy: eliminate the most conflicting (and cheapest) edges from
    x_star first, then greedily refill from the full graph edge pool.

    Parameters
    ----------
    x_star : list of (i, j) tuples
        Current (possibly infeasible) assignment from the Lagrangean subproblem.
    cost_matrix : array-like or np.ndarray, shape (n, n)
        Original cost matrix (without Lagrangean penalties).
    conflicts : list of [i, j, k, l] lists
        Conflict pairs.
    n : int
        Problem size.
    E0 : list of (i, j) tuples
        Known feasible seed solution used as last-resort fallback.
    graph_edges : list or set of (i, j) tuples, optional
        Edges present in the graph (sparse graph support).  None = complete n×n.

    Returns
    -------
    (assignment, objective, feasible)
    """
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.array(cost_matrix, dtype=float))
    conflict_idx = _build_conflict_index(conflicts)
    ge_set = (set(tuple(e) for e in graph_edges)
              if graph_edges is not None else None)

    survivors, rows_used, cols_used = _phase1_eliminate(x_star, cost, conflict_idx)
    completed = _phase2_complete(
        survivors, n, cost, conflict_idx,
        rows_used, cols_used, E0,
        graph_edges=ge_set,
    )

    assignment = sorted([tuple(e) for e in completed], key=lambda e: e[0])
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
    graph_edges=None,
    **kwargs,  # absorbs unexpected keyword args (e.g. lambdas) from batch_experiment
) -> Tuple[ab.Assignment, float, bool]:
    """Uniform heuristic interface — alias of repair().

    **kwargs silently absorbs keyword arguments passed by batch_experiment
    (e.g. ``lambdas=...``) that this heuristic does not use.
    """
    return repair(x_star, cost_matrix, conflicts, n, E0, graph_edges=graph_edges)


__all__ = [
    "HEURISTIC_NAME",
    "repair",
    "run",
]
