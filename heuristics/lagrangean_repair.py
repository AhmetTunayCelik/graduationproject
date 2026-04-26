"""
heuristics/lagrangean_repair.py
================================

Core-based repair heuristic for the Lagrangean relaxation of MAX-APC.

This heuristic converts a (possibly infeasible) solution x_star of the
Lagrangean subproblem into a feasible assignment for the original problem.
It follows Algorithms 3 and 4 of the MSTC reference (Minimum Spanning Tree
with Conflicts), adapted to assignment:

    Phase 1: Build a conflict-free core Q from x_star by a greedy rule.
    Phase 2: Complete Q to a full assignment using greedy extension,
             Hungarian sub-assignment, and finally falling back to E0.

Four ordering criteria are available (see ORDERINGS).
"""

from __future__ import annotations

import time
from collections import defaultdict
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
def _build_edge_conflict_index(conflicts: List[ab.Conflict]) -> Dict[ab.Edge, set]:
    """Return a dict mapping each edge to the set of edges it conflicts with."""
    neighbours = defaultdict(set)
    for c in conflicts:
        e = (c[0], c[1])
        f = (c[2], c[3])
        neighbours[e].add(f)
        neighbours[f].add(e)
    return neighbours


def _sort_key(ordering: str, edge: ab.Edge, cost: np.ndarray, degree: int):
    """Return a sort key for an edge under the given ordering.

    Sorted in ascending order, so for descending criteria we negate the value.
    """
    i, j = edge
    c = float(cost[i, j])
    if ordering == "dec_weight":
        return -c
    if ordering == "inc_weight":
        return c
    if ordering == "inc_degree":
        return (degree, -c)          # tie‑break by higher weight
    if ordering == "weight_over_degree":
        # Avoid division by zero
        return -(c / (1.0 + degree))
    raise ValueError(f"Unknown ordering: {ordering}")


def _phase1_core(
    x_star: ab.Assignment,
    cost: np.ndarray,
    neighbours: Dict[ab.Edge, set],
    ordering: str,
) -> Tuple[List[ab.Edge], set, set, set]:
    """Construct a conflict‑free core Q from x_star.

    Returns (core, rows_used, cols_used, forbidden), where forbidden is the
    set of edges that conflict with any core edge.
    """
    x_set = set(tuple(e) for e in x_star)
    degree_in_x = {e: len(neighbours[e] & x_set) for e in x_set}
    sorted_edges = sorted(
        x_set,
        key=lambda e: _sort_key(ordering, e, cost, degree_in_x[e]),
    )

    core = []
    rows_used = set()
    cols_used = set()
    forbidden = set()

    remaining = list(sorted_edges)
    while remaining:
        e = remaining.pop(0)
        if e in forbidden:
            continue
        if e[0] in rows_used or e[1] in cols_used:
            continue
        core.append(e)
        rows_used.add(e[0])
        cols_used.add(e[1])
        for f in neighbours[e]:
            forbidden.add(f)
        remaining = [f for f in remaining if f not in neighbours[e]]

    return core, rows_used, cols_used, forbidden


def _phase2_completion(
    core: List[ab.Edge],
    n: int,
    cost: np.ndarray,
    neighbours: Dict[ab.Edge, set],
    rows_used: set,
    cols_used: set,
    forbidden: set,
    ordering: str,
    E0: ab.Assignment,
    graph_edges: set = None,
) -> ab.Assignment:
    """Complete the core to a full feasible assignment.

    Uses greedy extension first, then Hungarian on residual submatrix,
    and finally falls back to E0.

    Parameters
    ----------
    graph_edges : set of (i, j) tuples, optional
        The set of edges present in the graph. When provided (sparse graph),
        only these edges are considered as candidates. When None, all n x n
        edges are available (complete graph, legacy behaviour).
    """
    core_set = set(tuple(e) for e in core)

    # ------------------------------------------------------------------
    # Greedy extension
    # ------------------------------------------------------------------
    # Candidate edges: free rows x free cols, not forbidden, in the graph.
    if graph_edges is not None:
        pool = [
            e for e in graph_edges
            if e[0] not in rows_used
            and e[1] not in cols_used
            and e not in forbidden
        ]
    else:
        pool = [
            (i, j)
            for i in range(n) if i not in rows_used
            for j in range(n) if j not in cols_used
            if (i, j) not in forbidden
        ]

    extended = list(core)
    rows = set(rows_used)
    cols = set(cols_used)
    block = set(forbidden)

    if pool:
        pool_set = set(pool)
        degree_in_pool = {e: len(neighbours[e] & pool_set) for e in pool}
        pool.sort(key=lambda e: _sort_key(ordering, e, cost, degree_in_pool[e]))

        while pool and len(extended) < n:
            e = pool.pop(0)
            if e[0] in rows or e[1] in cols:
                continue
            if e in block:
                continue
            extended.append(e)
            rows.add(e[0])
            cols.add(e[1])
            for f in neighbours[e]:
                block.add(f)
            pool = [
                f for f in pool
                if f not in neighbours[e] and f[0] != e[0] and f[1] != e[1]
            ]

        if len(extended) == n:
            return extended

    # ------------------------------------------------------------------
    # Hungarian completion on residual rows/cols.
    # NOTE: use `rows`/`cols` (updated by greedy), not the original
    # `rows_used`/`cols_used`, so we only fill truly unfilled slots.
    # ------------------------------------------------------------------
    free_rows = sorted(set(range(n)) - rows)
    free_cols = sorted(set(range(n)) - cols)

    if len(free_rows) == len(free_cols) and len(free_rows) > 0:
        m = len(free_rows)
        MASK = -1e15
        sub_profit = np.full((m, m), MASK, dtype=float)
        for a, i in enumerate(free_rows):
            for b, j in enumerate(free_cols):
                e = (i, j)
                if graph_edges is not None and e not in graph_edges:
                    continue   # edge not in sparse graph
                if e in block:
                    continue   # forbidden by core or greedy extension
                sub_profit[a, b] = cost[i, j]
        try:
            row_ind, col_ind = linear_sum_assignment(-sub_profit)
            if all(sub_profit[row_ind[a], col_ind[a]] > MASK / 2.0 for a in range(m)):
                completion = [
                    (free_rows[row_ind[a]], free_cols[col_ind[a]]) for a in range(m)
                ]
                comp_set = set(completion)
                # Check conflicts within completion AND against the core
                no_intra = all(not (neighbours[e] & comp_set) for e in completion)
                no_cross = all(not (neighbours[e] & core_set) for e in completion)
                if no_intra and no_cross:
                    return extended + completion
        except ValueError:
            pass

    # Fallback to E0
    return list(E0)


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
    """Convert a Lagrangean subproblem solution into a feasible assignment.

    Returns (assignment, objective, feasible). The fallback to E0 guarantees
    that a feasible assignment is always returned.

    Parameters
    ----------
    graph_edges : list or set of (i, j) tuples, optional
        Edges present in the graph (sparse graph support). When None, the
        full n x n matrix is used (complete graph, legacy behaviour).
        Typically obtained from instance["graph_edges"].
    """
    cost = cost_matrix if isinstance(cost_matrix, np.ndarray) else np.array(cost_matrix, dtype=float)
    neighbours = _build_edge_conflict_index(conflicts)
    ge_set = set(tuple(e) for e in graph_edges) if graph_edges is not None else None

    core, rows_used, cols_used, forbidden = _phase1_core(x_star, cost, neighbours, ordering)
    completed = _phase2_completion(
        core, n, cost, neighbours, rows_used, cols_used, forbidden, ordering, E0,
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
    ordering: str = DEFAULT_ORDERING,
    graph_edges=None,
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
    """Run the repair heuristic under every ordering criterion.

    Returns a dict mapping ordering name to a record containing objective,
    feasibility, runtime, core size, and the final assignment.

    Parameters
    ----------
    graph_edges : list or set of (i, j) tuples, optional
        Sparse graph edge set. Passed through to repair(). None = complete graph.
    """
    cost = cost_matrix if isinstance(cost_matrix, np.ndarray) else np.array(cost_matrix, dtype=float)
    neighbours = _build_edge_conflict_index(conflicts)
    records = {}

    for ordering in ORDERINGS:
        t0 = time.time()
        assignment, objective, feasible = repair(
            x_star, cost, conflicts, n, E0,
            ordering=ordering, graph_edges=graph_edges,
        )
        elapsed = time.time() - t0
        core, _, _, _ = _phase1_core(x_star, cost, neighbours, ordering)
        records[ordering] = {
            "ordering": ordering,
            "ordering_label": ORDERING_LABELS[ordering],
            "objective": float(objective),
            "feasible": bool(feasible),
            "runtime_seconds": float(elapsed),
            "core_size": int(len(core)),
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