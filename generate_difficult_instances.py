"""
generate_difficult_instances.py
===========================

Generate a balanced set of MAX-APC instances designed specifically to 
stress-test exact solvers (like Gurobi) and demonstrate the value of 
custom heuristics and Lagrangian relaxation.
"""

import os
import itertools
from instance_generator import generate_instance
import apc_base as ab
from parameters import config

def build_custom_batch(n_values, alphas, betas, seeds, cost_low, cost_high, instance_category, directory="instances"):
    os.makedirs(directory, exist_ok=True)
    created, skipped = 0, 0

    for n, alpha, beta, seed in itertools.product(n_values, alphas, betas, seeds):
        # Check if file exists BEFORE generating (expensive operation)
        temp_inst = {
            "n": n,
            "seed": seed,
            "conflict_graph_density": beta,
            "graph_density": alpha,
            "instance_category": instance_category,
        }
        fpath = os.path.join(directory, ab._instance_filename(temp_inst))
        if os.path.exists(fpath):
            skipped += 1
            continue

        # Only generate if file doesn't exist
        instance = generate_instance(
            n=n,
            seed=seed,
            conflict_graph_density=beta,
            graph_density=alpha,
            cost_low=cost_low,
            cost_high=cost_high,
            instance_category=instance_category,
        )
        ab.save_instance(instance, directory=directory)
        created += 1

    print(f"Created {created} instances (Skipped {skipped}) for "
          f"category='{instance_category}', n={n_values}, alphas={alphas}, "
          f"costs=[{cost_low},{cost_high}]")


if __name__ == "__main__":
    print("Generating Gurobi Stress-Test Instances...\n")

    # ---------------------------------------------------------
    # 1. THE GOLDILOCKS ZONE (Phase Transition)
    # Standard costs, but pushing beta higher to find the exact 
    # density where the Branch & Bound tree explodes.
    # ---------------------------------------------------------
    build_custom_batch(
        n_values=config.DIFF_GOLDILOCKS_N,
        alphas=config.DIFF_GOLDILOCKS_ALPHAS,
        betas=config.DIFF_GOLDILOCKS_BETAS,
        seeds=config.UNIVERSAL_SEEDS,
        cost_low=config.DEFAULT_COST_LOW,
        cost_high=config.DEFAULT_COST_HIGH,
        instance_category="goldilocks",
        directory=config.INSTANCE_DIR,
    )

    # ---------------------------------------------------------
    # 2. DEGENERACY (The "Flat Cost" Trap)
    # Tightly packed costs. Gurobi will struggle to prune the tree 
    # because thousands of nodes will have nearly identical bounds.
    # ---------------------------------------------------------
    build_custom_batch(
        n_values=config.DIFF_DEGEN_N,
        alphas=config.DIFF_DEGEN_ALPHAS,
        betas=config.DIFF_DEGEN_BETAS,
        seeds=config.UNIVERSAL_SEEDS,
        cost_low=config.DIFF_DEGEN_COST_LOW,
        cost_high=config.DIFF_DEGEN_COST_HIGH,
        instance_category="degen",
        directory=config.INSTANCE_DIR,
    )

    # ---------------------------------------------------------
    # 3. EXTREME SCALE
    # Testing pure dimensionality limits. Beta must be tiny to 
    # keep the number of conflicts manageable in memory.
    # ---------------------------------------------------------
    build_custom_batch(
        n_values=config.DIFF_EXTREME_N,
        alphas=config.DIFF_EXTREME_ALPHAS,
        betas=config.DIFF_EXTREME_BETAS,
        seeds=config.DIFF_EXTREME_SEEDS,
        cost_low=config.DEFAULT_COST_LOW,
        cost_high=config.DEFAULT_COST_HIGH,
        instance_category="extreme",
        directory=config.INSTANCE_DIR,
    )

    print("\nBatch generation complete!")