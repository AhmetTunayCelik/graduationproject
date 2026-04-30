"""
instance_generator.py
======================

Instance generation for MAX-APC, following the specification provided by
the project supervisor:

    n                = |V(G)|
    alpha            = |E(G)| / n^2                  (graph density)
    V(C) == E(G)
    beta             = |E(C)| / [|E(G)| * (|E(G)|-1) / 2]    (conflict density)

From these definitions:
    |E(G)| = alpha * n^2
    |E(C)| = (beta / 2) * |E(G)| * (|E(G)| - 1)

The algorithm has two steps:
    (i)  Generate a feasible solution as a matrix sudoku (a random
         permutation), used as the seed feasible assignment E0. This
         guarantees the instance admits at least one feasible solution.
    (ii) Sample the conflict graph: pick conflict pairs from among the
         graph edges according to the formula above.

All edges in E0 receive cost 0 so that E0 is never the preferred feasible
solution for a nontrivial instance; all other edges receive uniformly
random positive costs.

Generated instances are saved to the `instances/` directory and consumed
downstream by `batch_experiment.py`.

Notes on graph density (alpha)
------------------------------
The default behaviour uses a complete graph (alpha = 1), in which case
|E(G)| = n^2 and the cost matrix is dense. When `graph_density` is set to
a value strictly less than 1, the returned instance additionally records
the selected edge set in the `graph_edges` field. Note, however, that the
downstream subgradient solver currently operates on the dense n-by-n cost
matrix; restricting the solver to a proper subgraph of edges would require
an additional masking step that is left as a possible extension. For most
experiments the intended configuration is alpha = 1.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np

import apc_base as ab


def generate_instance(
    n: int = 10,
    num_conflicts: int = 20,
    seed: Optional[int] = None,
    conflict_graph_density: Optional[float] = None,
    graph_density: Optional[float] = None,
    cost_low: int = 1,
    cost_high: int = 100,
    instance_category: str = "standard",
    # Legacy alias: old callers used `density` for conflict density
    density: Optional[float] = None,
) -> ab.Instance:
    """Generate a random MAX-APC instance with a known feasible solution.

    The instance is constructed as follows:
        1. A random permutation E0 (seed feasible solution) is generated.
        2. If graph_density is given, a random subset of non-E0 edges of size
           max(0, int(graph_density * n²) - n) is selected to form the graph
           together with E0. Otherwise all n²-n non-E0 edges are included.
        3. All edges in E0 receive cost 0; graph edges outside E0 receive
           random integer costs in [cost_low, cost_high].
        4. Conflicts are sampled from graph edges only. Each conflict pair
           (e1, e2) must satisfy:
             - e1 ≠ e2
             - different rows AND different columns (not a natural AP conflict)
             - at least one of e1, e2 is a non-E0 edge  (E0 vs E0 forbidden)
           This allows E0-vs-non-E0 conflicts, but E0 remains feasible by
           construction since no two E0 edges appear in the same conflict.

    Parameters
    ----------
    n : int
        Problem size (number of agents = number of tasks).
    num_conflicts : int
        Desired number of explicit conflict pairs. Ignored when
        conflict_graph_density is provided.
    seed : int, optional
        Random seed. If None, a time‑derived seed is used.
    conflict_graph_density : float, optional
        Conflict density relative to n². When provided, num_conflicts is set
        to max(1, int(conflict_graph_density * n²)).
    graph_density : float, optional
        Edge density of the full graph relative to n². When provided, the
        graph contains E0 plus a random subset of non-E0 edges so that the
        total edge count ≈ int(graph_density * n²). Must satisfy
        graph_density >= n/n² = 1/n so that E0 itself fits.
        If None, the graph is complete (all n² edges included).
    cost_low, cost_high : int
        Inclusive range for random costs (positive integers).
    density : float, optional
        Legacy alias for conflict_graph_density. Ignored when
        conflict_graph_density is also provided.

    Returns
    -------
    dict
        Instance dictionary with keys: n, cost_matrix, conflicts, E0, seed,
        conflict_graph_density, graph_density, num_conflicts, graph_edges.
        graph_edges is a sorted list of all [i, j] edges in the graph
        (E0 ∪ selected non-E0 edges).
    """
    # --- resolve legacy `density` alias ---
    if conflict_graph_density is None and density is not None:
        conflict_graph_density = density

    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed % (2**31))

    # ------------------------------------------------------------------
    # Step 1: Seed feasible solution — random permutation
    # ------------------------------------------------------------------
    perm = list(range(n))
    rng.shuffle(perm)
    E0 = [(i, perm[i]) for i in range(n)]
    E0_set = set(E0)

    # ------------------------------------------------------------------
    # Step 2: Build the graph edge set
    # ------------------------------------------------------------------
    all_non_E0 = [(i, j) for i in range(n) for j in range(n) if (i, j) not in E0_set]

    if graph_density is not None:
        total_edges_target = int(graph_density * n * n)
        # E0 already provides n edges; fill the rest from non-E0 pool
        non_E0_budget = max(0, total_edges_target - n)
        if non_E0_budget >= len(all_non_E0):
            selected_non_E0 = all_non_E0
        else:
            selected_non_E0 = rng.sample(all_non_E0, non_E0_budget)
        eff_graph_density = (n + len(selected_non_E0)) / (n * n)
    else:
        selected_non_E0 = all_non_E0
        eff_graph_density = 1.0  # complete graph

    selected_non_E0_set = set(selected_non_E0)
    graph_edges = sorted(E0_set | selected_non_E0_set)

    # ------------------------------------------------------------------
    # Step 3: Cost matrix — full n×n (sparse flag carried via graph_edges)
    # ------------------------------------------------------------------
    cost_matrix = np_rng.randint(cost_low, cost_high + 1, size=(n, n)).tolist()
    for i, j in E0:
        cost_matrix[i][j] = 0

    # ------------------------------------------------------------------
    # Steps 4 & 5: Sample conflict pairs.
    #
    # Strategy depends on graph size:
    #
    # SMALL GRAPH (|E(G)|^2 fits in memory): Build the full valid pool
    #   upfront — every structurally valid unordered pair (e1, e2) — then
    #   draw exactly the required number with rng.sample(). This is O(|E|^2)
    #   to build but completely eliminates rejection-sampling waste, and
    #   makes beta mathematically exact against the true valid pool.
    #
    # LARGE GRAPH (pool would exceed memory): Fall back to rejection
    #   sampling with a generous attempt budget. At large n the requested
    #   beta is tiny (per the tiered design in generate_many_instances.py)
    #   so the pool hit-rate stays high and rejections remain rare.
    #
    # Threshold: |E(G)| <= 1600 means the pool has at most ~1.3M pairs
    # (~150 MB), which builds quickly. For alpha=1 this covers n <= 40.
    #
    # Note on `beta` semantics across the boundary
    # --------------------------------------------
    # For the small-graph branch, beta is mathematically exact against the
    # *true* valid pool (filtered by row/col disjointness and E0-vs-E0
    # exclusion). For the large-graph branch we use the cheaper upper bound
    # |E(G)|*(|E(G)|-1)/2 as the denominator — a slight asymmetry.
    # Effect: at the same nominal beta value, the large-graph instances
    # actually request slightly *more* conflicts than the formula would
    # imply against the strict pool. Acceptable because (a) at large n the
    # filtered fraction stays close to 1 (most random pairs are row/col-
    # disjoint), and (b) the experimental design uses small beta values at
    # large n precisely to keep memory tractable. Reviewers should be
    # alerted to this in the methodology section.
    # ------------------------------------------------------------------
    POOL_EDGE_LIMIT = 1600

    num_graph_edges = len(graph_edges)

    if num_graph_edges <= POOL_EDGE_LIMIT:
        # ── Direct pool approach ────────────────────────────────────────
        valid_conflict_pool = []
        for i in range(num_graph_edges):
            e1 = graph_edges[i]
            for j in range(i + 1, num_graph_edges):
                e2 = graph_edges[j]
                if e1[0] == e2[0] or e1[1] == e2[1]:
                    continue      # natural assignment conflict
                if e1 in E0_set and e2 in E0_set:
                    continue      # E0 vs E0 forbidden
                valid_conflict_pool.append([e1[0], e1[1], e2[0], e2[1]])

        max_valid_pairs = len(valid_conflict_pool)
        if conflict_graph_density is not None:
            num_conflicts = max(1, int(round(conflict_graph_density * max_valid_pairs)))
            num_conflicts = min(num_conflicts, max_valid_pairs)

        conflicts = rng.sample(valid_conflict_pool, num_conflicts) if (
            num_conflicts > 0 and max_valid_pairs > 0
        ) else []

    else:
        # ── Rejection sampling (large graph) ────────────────────────────
        # beta is kept small by the tiered design, so the hit-rate is high.
        max_valid_pairs = num_graph_edges * (num_graph_edges - 1) // 2
        if conflict_graph_density is not None:
            num_conflicts = max(1, int(round(conflict_graph_density * max_valid_pairs)))
            num_conflicts = min(num_conflicts, max_valid_pairs)

        graph_edges_list = graph_edges   # already a list
        conflicts = []
        conflict_set: set = set()
        attempts = 0
        max_attempts = num_conflicts * 50   # generous but bounded
        while len(conflicts) < num_conflicts and attempts < max_attempts:
            attempts += 1
            e1 = rng.choice(graph_edges_list)
            e2 = rng.choice(graph_edges_list)
            if e1 == e2:
                continue
            if e1[0] == e2[0] or e1[1] == e2[1]:
                continue
            if e1 in E0_set and e2 in E0_set:
                continue
            key = (min(e1, e2), max(e1, e2))
            if key in conflict_set:
                continue
            conflict_set.add(key)
            conflicts.append([e1[0], e1[1], e2[0], e2[1]])

        if len(conflicts) < num_conflicts:
            print(f"  [Warning] Only {len(conflicts)} of {num_conflicts} "
                  f"conflicts generated (attempt budget exhausted).")

    # Back-compute effective density against the valid pool.
    if conflict_graph_density is not None:
        eff_conflict_density = conflict_graph_density
    elif max_valid_pairs > 0:
        eff_conflict_density = len(conflicts) / max_valid_pairs
    else:
        eff_conflict_density = 0.0

# --- ISSUE C FIX: Mask non-graph edges so Gurobi & Subgradient respect sparsity ---
    PENALTY = -999999.0

    # We MUST include E0 edges to ensure the baseline remains strictly feasible
    valid_edges = set(tuple(e) for e in graph_edges)
    valid_edges.update(tuple(e) for e in E0)

    # Poison the non-graph edges in the cost matrix
    for i in range(n):
        for j in range(n):
            if (i, j) not in valid_edges:
                cost_matrix[i][j] = PENALTY
# ----------------------------------------------------------------------------------

    return {
        "n": n,
        "cost_matrix": cost_matrix,
        "conflicts": conflicts,
        "E0": E0,
        "seed": seed,
        "conflict_graph_density": eff_conflict_density,
        "graph_density": eff_graph_density,
        # `density` is retained as a convenience alias of the conflict
        # density (beta in the spec). It is the value referenced by the
        # filename helpers in :mod:`apc_base` and by downstream analysis.
        "density": eff_conflict_density,
        "num_conflicts": len(conflicts),
        "graph_edges": list(graph_edges),  # list of (i,j) tuples, sorted
        "instance_category": instance_category,
    }


def generate_batch(
    n_values: List[int],
    conflict_graph_densities: List[float],
    seeds: List[int],
    graph_densities: Optional[List[Optional[float]]] = None,
    directory: str = "instances",
    force: bool = False,
    instance_category: str = "standard",
    # Legacy alias
    densities: Optional[List[float]] = None,
) -> Tuple[int, int]:
    """Generate all instances for a Cartesian product of parameters.

    Only instances that do not already exist (or force=True) are created.

    Parameters
    ----------
    n_values : list of int
    conflict_graph_densities : list of float
        Conflict densities relative to n².
    seeds : list of int
    graph_densities : list of float or None, optional
        Edge densities for the graph. A None entry means complete graph.
        If omitted entirely, all instances use a complete graph.
    directory : str
    force : bool
    densities : list of float, optional
        Legacy alias for conflict_graph_densities.

    Returns
    -------
    (created, skipped) : tuple of ints
    """
    if densities is not None and not conflict_graph_densities:
        conflict_graph_densities = densities

    if graph_densities is None:
        graph_densities = [None]

    os.makedirs(directory, exist_ok=True)
    created = 0
    skipped = 0

    for n, c_density, g_density, seed in itertools.product(
        n_values, conflict_graph_densities, graph_densities, seeds
    ):
        # Check if file exists BEFORE generating (expensive operation)
        temp_inst = {
            "n": n,
            "seed": seed,
            "conflict_graph_density": c_density,
            "graph_density": g_density or 1.0,
            "instance_category": instance_category,
        }
        fpath = os.path.join(directory, ab._instance_filename(temp_inst))
        if not force and os.path.exists(fpath):
            skipped += 1
            continue

        # Only generate if file doesn't exist
        instance = generate_instance(
            n=n,
            seed=seed,
            conflict_graph_density=c_density,
            graph_density=g_density,
            instance_category=instance_category,
        )
        ab.save_instance(instance, directory=directory)
        created += 1

    return created, skipped


def main():
    parser = argparse.ArgumentParser(description="Generate MAX-APC instances in batch.")
    parser.add_argument("--n-values", type=int, nargs="+", required=True,
                        help="List of problem sizes, e.g. 10 20 50")
    parser.add_argument("--conflict-densities", type=float, nargs="+", required=True,
                        help="Conflict densities, e.g. 0.01 0.05 0.1")
    parser.add_argument("--graph-densities", type=float, nargs="*", default=None,
                        help="Graph edge densities, e.g. 0.3 0.5 (omit for complete graph)")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="List of random seeds")
    parser.add_argument("--instance-dir", default="instances",
                        help="Directory to store instances")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing instance files")
    args = parser.parse_args()

    created, skipped = generate_batch(
        n_values=args.n_values,
        conflict_graph_densities=args.conflict_densities,
        graph_densities=args.graph_densities,
        seeds=args.seeds,
        directory=args.instance_dir,
        force=args.force,
    )
    print(f"Instance generation complete. Created: {created}, Skipped: {skipped}")


if __name__ == "__main__":
    main()