"""
heuristics/lagrangean_repair_lambda.py
======================================

Lambda-aware core-based repair heuristic for the Lagrangean relaxation of MAX-APC.

This heuristic extends the existing lagrangean_repair.py idea by incorporating
dual information from the final Lagrange multipliers (lambdas). The goal is to
prefer edges that are:
    - high-weight,
    - low-conflict,
    - involved in lower-penalty conflict structures.

It follows the same two-phase design:
    Phase 1: Build a conflict-free core Q from x_star.
    Phase 2: Complete Q to a full assignment via greedy extension,
             then Hungarian residual completion, then fallback to E0.

Compatible with the framework:
    - HEURISTIC_NAME
    - run(...)
    - run_all_orderings(...)
"""

from __future__ import annotations

import time
from collections import defaultdict
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

# Strength of lambda penalty in ordering score
DEFAULT_MU = 1.0


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _build_edge_conflict_index(conflicts: List[ab.Conflict]) -> Dict[ab.Edge, set]:
    """Map each edge to the set of explicit-conflict neighbours."""
    neighbours = defaultdict(set)
    for c in conflicts:
        e = (c[0], c[1])
        f = (c[2], c[3])
        neighbours[e].add(f)
        neighbours[f].add(e)
    return neighbours


def _build_edge_lambda_sum(
    conflicts: List[ab.Conflict],
    lambdas: Optional[List[float]],
) -> Dict[ab.Edge, float]:
    """
    For each edge e, compute the sum of lambdas of all conflict constraints
    that contain e.
    """
    lam_sum = defaultdict(float)
    if lambdas is None:
        return lam_sum

    # Safe handling if lambdas is numpy array or shorter/longer than expected
    m = min(len(conflicts), len(lambdas))
    for idx in range(m):
        c = conflicts[idx]
        lam = float(lambdas[idx])
        e = (c[0], c[1])
        f = (c[2], c[3])
        lam_sum[e] += lam
        lam_sum[f] += lam
    return lam_sum


def _sort_key(
    ordering: str,
    edge: ab.Edge,
    cost: np.ndarray,
    degree: int,
    lambda_sum: float,
    mu: float,
):
    """
    Return a sort key for an edge under the given lambda-aware ordering.
    Sorted in ascending order.
    """
    i, j = edge
    c = float(cost[i, j])

    if ordering == "lambda_penalized_weight":
        # Prefer high weight and low lambda burden
        return -(c - mu * lambda_sum)

    if ordering == "lambda_weight_over_degree":
        # Prefer high weight normalized by both degree and lambda penalty
        return -(c / (1.0 + degree + mu * lambda_sum))

    if ordering == "lambda_over_degree_then_weight":
        # Prefer low combined risk, tie-break by higher weight
        return (degree + mu * lambda_sum, -c)

    raise ValueError(f"Unknown ordering: {ordering}")


def _phase1_core(
    x_star: ab.Assignment,
    cost: np.ndarray,
    neighbours: Dict[ab.Edge, set],
    edge_lambda_sum: Dict[ab.Edge, float],
    ordering: str,
    mu: float,
) -> Tuple[List[ab.Edge], set, set, set]:
    """
    Construct a conflict-free core Q from x_star.

    Returns:
        core, rows_used, cols_used, forbidden
    """
    x_set = set(tuple(e) for e in x_star)
    degree_in_x = {e: len(neighbours[e] & x_set) for e in x_set}

    sorted_edges = sorted(
        x_set,
        key=lambda e: _sort_key(
            ordering=ordering,
            edge=e,
            cost=cost,
            degree=degree_in_x[e],
            lambda_sum=edge_lambda_sum[e],
            mu=mu,
        ),
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
    edge_lambda_sum: Dict[ab.Edge, float],
    rows_used: set,
    cols_used: set,
    forbidden: set,
    ordering: str,
    E0: ab.Assignment,
    mu: float,
    graph_edges: set = None,
) -> ab.Assignment:
    """
    Complete the core to a full feasible assignment.

    Strategy:
        1. Greedy extension using lambda-aware ordering
        2. Hungarian residual completion
        3. Fallback to E0
    """
    core_set = set(tuple(e) for e in core)

    # ------------------------------------------------------------------
    # Greedy extension
    # ------------------------------------------------------------------
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

        pool.sort(
            key=lambda e: _sort_key(
                ordering=ordering,
                edge=e,
                cost=cost,
                degree=degree_in_pool[e],
                lambda_sum=edge_lambda_sum[e],
                mu=mu,
            )
        )

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
    # Hungarian completion on residual rows/cols
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
                    continue
                if e in block:
                    continue

                # Lambda-aware residual scoring:
                # still optimize weight, but softly discourage high-lambda edges
                sub_profit[a, b] = cost[i, j] - mu * edge_lambda_sum[e]

        try:
            row_ind, col_ind = linear_sum_assignment(-sub_profit)

            if all(sub_profit[row_ind[a], col_ind[a]] > MASK / 2.0 for a in range(m)):
                completion = [
                    (free_rows[row_ind[a]], free_cols[col_ind[a]])
                    for a in range(m)
                ]
                comp_set = set(completion)

                # Check conflicts inside completion and against core
                no_intra = all(not (neighbours[e] & comp_set) for e in completion)
                no_cross = all(not (neighbours[e] & core_set) for e in completion)

                if no_intra and no_cross:
                    return extended + completion
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
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
    lambdas: Optional[List[float]] = None,
    mu: float = DEFAULT_MU,
    graph_edges=None,
) -> Tuple[ab.Assignment, float, bool]:
    """
    Convert a Lagrangean subproblem solution into a feasible assignment.

    Parameters
    ----------
    lambdas : list or np.ndarray, optional
        Final Lagrange multipliers from subgradient. If None, the heuristic
        still works, but lambda-aware terms become zero and the method
        degenerates to a cost/degree-based variant.
    mu : float
        Strength of lambda penalty in ordering and residual completion.
    graph_edges : optional sparse graph edge set

    Returns
    -------
    assignment, objective, feasible
    """
    cost = cost_matrix if isinstance(cost_matrix, np.ndarray) else np.array(cost_matrix, dtype=float)

    neighbours = _build_edge_conflict_index(conflicts)
    edge_lambda_sum = _build_edge_lambda_sum(conflicts, lambdas)
    ge_set = set(tuple(e) for e in graph_edges) if graph_edges is not None else None

    core, rows_used, cols_used, forbidden = _phase1_core(
        x_star=x_star,
        cost=cost,
        neighbours=neighbours,
        edge_lambda_sum=edge_lambda_sum,
        ordering=ordering,
        mu=mu,
    )

    completed = _phase2_completion(
        core=core,
        n=n,
        cost=cost,
        neighbours=neighbours,
        edge_lambda_sum=edge_lambda_sum,
        rows_used=rows_used,
        cols_used=cols_used,
        forbidden=forbidden,
        ordering=ordering,
        E0=E0,
        mu=mu,
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
    lambdas: Optional[List[float]] = None,
    mu: float = DEFAULT_MU,
    graph_edges=None,
) -> Tuple[ab.Assignment, float, bool]:
    """
    Standard heuristic interface expected by batch_experiment.py.
    """
    return repair(
        x_star=x_star,
        cost_matrix=cost_matrix,
        conflicts=conflicts,
        n=n,
        E0=E0,
        ordering=ordering,
        lambdas=lambdas,
        mu=mu,
        graph_edges=graph_edges,
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
) -> Dict[str, Dict[str, Any]]:
    """
    Run the repair heuristic under every lambda-aware ordering criterion.
    """
    cost = cost_matrix if isinstance(cost_matrix, np.ndarray) else np.array(cost_matrix, dtype=float)
    neighbours = _build_edge_conflict_index(conflicts)
    edge_lambda_sum = _build_edge_lambda_sum(conflicts, lambdas)

    records = {}
    for ordering in ORDERINGS:
        t0 = time.time()

        assignment, objective, feasible = repair(
            x_star=x_star,
            cost_matrix=cost,
            conflicts=conflicts,
            n=n,
            E0=E0,
            ordering=ordering,
            lambdas=lambdas,
            mu=mu,
            graph_edges=graph_edges,
        )

        elapsed = time.time() - t0
        core, _, _, _ = _phase1_core(
            x_star=x_star,
            cost=cost,
            neighbours=neighbours,
            edge_lambda_sum=edge_lambda_sum,
            ordering=ordering,
            mu=mu,
        )

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
    "DEFAULT_MU",
    "repair",
    "run",
    "run_all_orderings",
]