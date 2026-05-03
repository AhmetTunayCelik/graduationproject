"""
heuristics/greedy_baseline.py
=============================

Conflict-aware greedy baseline for MAX-APC.

This is the *control* heuristic for the experimental comparison: it has
no Lagrangean dual information, no subgradient ascent, no penalty signal.
It exists so the thesis can answer "does Lagrangean relaxation actually
help?" with a side-by-side comparison rather than an assertion.

Algorithm (single pass, deterministic given costs and conflicts):

    1. Sort all graph edges by descending profit.
    2. Walk the sorted list; accept an edge iff its row and column are
       still free AND it conflicts with no previously-accepted edge.
    3. If fewer than n edges were accepted, fill the remaining rows/cols
       via Hungarian on the residual sub-matrix, masking out edges that
       conflict with the accepted set.
    4. If that still fails, report no feasible solution (returns None).

The heuristic carries a module-level `SKIP_SUBGRADIENT = True` flag so
batch_experiment.py runs it standalone, without paying the Lagrangean
loop's CPU cost it doesn't need.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import apc_base as ab


HEURISTIC_NAME = "greedy_baseline"
SKIP_SUBGRADIENT = True   # picked up by batch_experiment.run_single_combination


def _greedy_pass(
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    n: int,
    graph_edge_mask: np.ndarray,
) -> Tuple[List[int], set, set, np.ndarray]:
    """Pass 1: sort by descending weight, accept conflict-free edges greedily."""
    nn = n * n
    flat_cost = cost.ravel()

    # Candidate edges = graph edges with strictly positive cost
    # (E0 edges have cost 0 by construction; we don't want to lock those in
    # at the greedy stage — the residual Hungarian / E0 fallback handles them).
    candidate_ids = np.flatnonzero(graph_edge_mask & (flat_cost > 0))
    # Sort descending by cost
    order = candidate_ids[np.argsort(-flat_cost[candidate_ids], kind="stable")]

    accepted: List[int] = []
    rows: set = set()
    cols: set = set()
    forbidden_mask = np.zeros(nn, dtype=bool)

    for eid in order:
        eid = int(eid)
        if forbidden_mask[eid]:
            continue
        i, j = divmod(eid, n)
        if i in rows or j in cols:
            continue
        accepted.append(eid)
        rows.add(i)
        cols.add(j)
        nbrs = neighbours[eid]
        if nbrs.size:
            forbidden_mask[nbrs] = True
        if len(accepted) >= n:
            break

    return accepted, rows, cols, forbidden_mask


def _hungarian_complete(
    accepted: List[int],
    rows: set,
    cols: set,
    forbidden_mask: np.ndarray,
    cost: np.ndarray,
    neighbours: List[np.ndarray],
    n: int,
    graph_edge_mask: np.ndarray,
) -> List[int]:
    """Pass 2: Hungarian on the residual rows/cols, masking out conflicts."""
    nn = n * n
    free_rows = sorted(set(range(n)) - rows)
    free_cols = sorted(set(range(n)) - cols)
    if not (len(free_rows) == len(free_cols) > 0):
        return accepted

    m = len(free_rows)
    MASK = -1e15
    sub = np.full((m, m), MASK, dtype=np.float32)
    for a, i in enumerate(free_rows):
        for b, j in enumerate(free_cols):
            eid = i * n + j
            if not graph_edge_mask[eid] or forbidden_mask[eid]:
                continue
            sub[a, b] = cost[i, j]

    try:
        ri, ci = linear_sum_assignment(-sub)
    except ValueError:
        return accepted

    if not all(sub[ri[a], ci[a]] > MASK / 2.0 for a in range(m)):
        return accepted

    completion = [free_rows[ri[a]] * n + free_cols[ci[a]] for a in range(m)]

    # Verify completion is internally and cross-conflict-free.
    comp_mask = np.zeros(nn, dtype=bool)
    comp_mask[completion] = True
    acc_mask = np.zeros(nn, dtype=bool)
    if accepted:
        acc_mask[accepted] = True
    for eid in completion:
        nbrs = neighbours[eid]
        if nbrs.size and (comp_mask[nbrs].any() or acc_mask[nbrs].any()):
            return accepted   # would re-introduce a conflict; bail out
    return accepted + completion


def run(
    x_star=None,
    cost_matrix=None,
    conflicts: List[ab.Conflict] = None,
    n: int = 0,
    E0: ab.Assignment = None,
    graph_edges=None,
    neighbours=None,
    **kwargs,
) -> Tuple[ab.Assignment, float, bool]:
    """Run greedy baseline. `x_star` and `lambdas` are accepted and ignored
    (this baseline has no Lagrangean information by design)."""
    cost = (cost_matrix if isinstance(cost_matrix, np.ndarray)
            else np.asarray(cost_matrix, dtype=np.float32))
    if neighbours is None:
        neighbours = ab.build_conflict_adjacency_int(conflicts or [], n)

    if graph_edges is not None:
        graph_edge_mask = np.zeros(n * n, dtype=bool)
        for i, j in graph_edges:
            graph_edge_mask[i * n + j] = True
    else:
        graph_edge_mask = np.ones(n * n, dtype=bool)

    accepted, rows, cols, forbidden_mask = _greedy_pass(
        cost, neighbours, n, graph_edge_mask,
    )

    if len(accepted) < n:
        accepted = _hungarian_complete(
            accepted, rows, cols, forbidden_mask,
            cost, neighbours, n, graph_edge_mask,
        )

    if len(accepted) < n:
        # Honest failure: greedy + Hungarian could not produce a complete
        # assignment. No artificial E0 fallback.
        return None, None, False

    assignment = sorted([divmod(eid, n) for eid in accepted], key=lambda e: e[0])
    feasible = ab.is_valid_assignment(assignment, conflicts or [], n, graph_edges)
    if not feasible:
        return None, None, False
    objective = float(sum(cost[i, j] for i, j in assignment))
    return assignment, objective, True


__all__ = ["HEURISTIC_NAME", "SKIP_SUBGRADIENT", "run"]
