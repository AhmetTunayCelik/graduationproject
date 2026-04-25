"""
heuristics/dual_grasp.py
========================

Dual-Guided GRASP (Greedy Randomized Adaptive Search Procedure) with Local Search
for the Maximum Assignment Problem with Conflicts (MAX-APC).

Phase 1: Dual Penalization - Adjusts raw costs using Subgradient lambdas.
Phase 2: GRASP Construction - Iteratively builds an assignment using a Restricted
         Candidate List (RCL) based on the penalized costs.
Phase 3: Local Search - Applies a 2-opt neighborhood search (swapping assignments)
         to push the lower bound up, strictly enforcing conflict-freedom.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

import apc_base as ab


# -----------------------------------------------------------------------------
# Heuristic Configurations
# -----------------------------------------------------------------------------
HEURISTIC_NAME = "dual_grasp"

# Alpha controls the size of the Restricted Candidate List (RCL).
# alpha=1 is pure greedy. Higher alpha means more exploration/randomness.
ALPHAS = {
    "alpha_1": 1,
    "alpha_2": 2,
    "alpha_3": 3,
    "alpha_5": 5,
}

DEFAULT_ALPHA_KEY = "alpha_3"
GRASP_ITERATIONS = 10


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _build_edge_conflict_index(conflicts: List[ab.Conflict]) -> Dict[ab.Edge, set]:
    """Map each edge to the set of edges it conflicts with."""
    neighbors = defaultdict(set)
    for c in conflicts:
        e = (c[0], c[1])
        f = (c[2], c[3])
        neighbors[e].add(f)
        neighbors[f].add(e)
    return neighbors


def _compute_penalized_costs(
    cost: np.ndarray, 
    conflicts: List[ab.Conflict], 
    lambdas: List[float]
) -> np.ndarray:
    """Subtract the Lagrangian multipliers from the costs of conflicting edges."""
    p_cost = cost.copy()
    if not lambdas or len(lambdas) != len(conflicts):
        return p_cost  # Fallback to standard cost if lambdas are missing
        
    for k, c in enumerate(conflicts):
        if lambdas[k] > 0:
            p_cost[c[0], c[1]] -= lambdas[k]
            p_cost[c[2], c[3]] -= lambdas[k]
    return p_cost


def _local_search_2opt(
    assignment: List[ab.Edge],
    cost: np.ndarray,
    neighbors: Dict[ab.Edge, set],
    n: int
) -> List[ab.Edge]:
    """First-improvement 2-opt local search."""
    curr_assignment = list(assignment)
    improved = True
    
    while improved:
        improved = False
        for a in range(n):
            for b in range(a + 1, n):
                i1, j1 = curr_assignment[a]
                i2, j2 = curr_assignment[b]
                
                # The proposed swap
                e1_new = (i1, j2)
                e2_new = (i2, j1)
                
                # Check objective change (raw cost, not penalized)
                delta = cost[i1, j2] + cost[i2, j1] - cost[i1, j1] - cost[i2, j2]
                
                if delta > 1e-6:
                    # Check conflicts with the REST of the assignment
                    rest_of_asgn = set(curr_assignment) - {(i1, j1), (i2, j2)}
                    
                    # If the new edges conflict with anything, or with each other, abort swap
                    if (neighbors[e1_new] & rest_of_asgn) or \
                       (neighbors[e2_new] & rest_of_asgn) or \
                       (e1_new in neighbors[e2_new]):
                        continue
                    
                    # Valid improvement found: Apply swap and restart search
                    curr_assignment[a] = e1_new
                    curr_assignment[b] = e2_new
                    improved = True
                    break # First improvement loop break
            if improved:
                break
                
    return curr_assignment


# -----------------------------------------------------------------------------
# Main Algorithm
# -----------------------------------------------------------------------------
def repair(
    x_star: ab.Assignment, # Unused here, but kept for interface compliance
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    lambdas: List[float] = None,
    alpha: int = 3,
    seed: int = 42
) -> Tuple[ab.Assignment, float, bool]:
    """Constructs a feasible assignment using Dual-GRASP and Local Search."""
    
    cost = cost_matrix if isinstance(cost_matrix, np.ndarray) else np.array(cost_matrix, dtype=float)
    neighbors = _build_edge_conflict_index(conflicts)
    p_cost = _compute_penalized_costs(cost, conflicts, lambdas)
    
    rng = random.Random(seed)
    
    best_assignment = list(E0)
    best_obj = float(sum(cost[i, j] for i, j in E0))
    found_feasible = False

    for iteration in range(GRASP_ITERATIONS):
        # 1. GRASP Construction
        curr_assignment = []
        rows_used = set()
        cols_used = set()
        forbidden = set()
        dead_end = False
        
        while len(curr_assignment) < n:
            # Build Candidate List
            candidates = []
            for i in range(n):
                if i in rows_used: continue
                for j in range(n):
                    if j in cols_used: continue
                    e = (i, j)
                    if e not in forbidden:
                        candidates.append((p_cost[i, j], e))
            
            if not candidates:
                dead_end = True
                break
                
            # Sort candidates by penalized profit (descending)
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Form Restricted Candidate List (RCL)
            rcl_size = min(alpha, len(candidates))
            chosen_edge = rng.choice(candidates[:rcl_size])[1]
            
            curr_assignment.append(chosen_edge)
            rows_used.add(chosen_edge[0])
            cols_used.add(chosen_edge[1])
            forbidden.update(neighbors[chosen_edge])
            
        # 2. Local Search (If construction was successful)
        if not dead_end:
            curr_assignment = _local_search_2opt(curr_assignment, cost, neighbors, n)
            curr_obj = float(sum(cost[i, j] for i, j in curr_assignment))
            
            if curr_obj > best_obj:
                best_obj = curr_obj
                best_assignment = curr_assignment
                found_feasible = True

    # Final feasibility check
    is_feasible = len(best_assignment) == n and len(ab.find_violations(best_assignment, conflicts, n)) == 0
    return sorted(best_assignment, key=lambda e: e[0]), best_obj, is_feasible


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def run(
    x_star: ab.Assignment,
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    lambdas: List[float] = None
) -> Tuple[ab.Assignment, float, bool]:
    return repair(x_star, cost_matrix, conflicts, n, E0, lambdas=lambdas, alpha=ALPHAS[DEFAULT_ALPHA_KEY])


def run_all_orderings(
    x_star: ab.Assignment,
    cost_matrix,
    conflicts: List[ab.Conflict],
    n: int,
    E0: ab.Assignment,
    lambdas: List[float] = None
) -> Dict[str, Dict[str, Any]]:
    """Runs the GRASP heuristic for different Alpha (RCL) sizes."""
    
    records = {}
    for alpha_key, alpha_val in ALPHAS.items():
        t0 = time.time()
        assignment, objective, feasible = repair(
            x_star, cost_matrix, conflicts, n, E0, lambdas=lambdas, alpha=alpha_val
        )
        elapsed = time.time() - t0
        
        records[alpha_key] = {
            "ordering": alpha_key,
            "ordering_label": f"RCL Size {alpha_val}",
            "objective": float(objective),
            "feasible": bool(feasible),
            "runtime_seconds": float(elapsed),
            "assignment": [tuple(e) for e in assignment],
        }
    return records


__all__ = [
    "HEURISTIC_NAME",
    "ALPHAS",
    "run",
    "run_all_orderings",
]