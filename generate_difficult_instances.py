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

def build_custom_batch(n_values, betas, seeds, cost_low, cost_high, directory="instances"):
    os.makedirs(directory, exist_ok=True)
    created, skipped = 0, 0
    
    for n, beta, seed in itertools.product(n_values, betas, seeds):
        # We call generate_instance directly to inject custom cost ranges
        instance = generate_instance(
            n=n,
            seed=seed,
            conflict_graph_density=beta,
            cost_low=cost_low,
            cost_high=cost_high
        )
        fpath = os.path.join(directory, ab._instance_filename(instance))
        if os.path.exists(fpath):
            skipped += 1
            continue
            
        ab.save_instance(instance, directory=directory)
        created += 1
        
    print(f"Created {created} instances (Skipped {skipped}) for n={n_values}, costs=[{cost_low},{cost_high}]")


if __name__ == "__main__":
    print("Generating Gurobi Stress-Test Instances...\n")

    # ---------------------------------------------------------
    # 1. THE GOLDILOCKS ZONE (Phase Transition)
    # Standard costs, but pushing beta higher to find the exact 
    # density where the Branch & Bound tree explodes.
    # ---------------------------------------------------------
    build_custom_batch(
        n_values=config.DIFF_GOLDILOCKS_N,
        betas=config.DIFF_GOLDILOCKS_BETAS, # Pushing beta much higher
        seeds=config.DIFF_GOLDILOCKS_SEEDS,
        cost_low=config.DEFAULT_COST_LOW, 
        cost_high=config.DEFAULT_COST_HIGH,
        directory=config.INSTANCE_DIR
    )

    # ---------------------------------------------------------
    # 2. DEGENERACY (The "Flat Cost" Trap)
    # Tightly packed costs. Gurobi will struggle to prune the tree 
    # because thousands of nodes will have nearly identical bounds.
    # ---------------------------------------------------------
    build_custom_batch(
        n_values=config.DIFF_DEGEN_N,
        betas=config.DIFF_DEGEN_BETAS,
        seeds=config.DIFF_DEGEN_SEEDS,
        cost_low=config.DIFF_DEGEN_COST_LOW, 
        cost_high=config.DIFF_DEGEN_COST_HIGH,  # High degeneracy!
        directory=config.INSTANCE_DIR
    )

    # ---------------------------------------------------------
    # 3. EXTREME SCALE
    # Testing pure dimensionality limits. Beta must be tiny to 
    # keep the number of conflicts manageable in memory.
    # ---------------------------------------------------------
    build_custom_batch(
        n_values=config.DIFF_EXTREME_N,
        betas=config.DIFF_EXTREME_BETAS,
        seeds=config.DIFF_EXTREME_SEEDS, # Only 3 seeds per config, these take space
        cost_low=config.DEFAULT_COST_LOW, 
        cost_high=config.DEFAULT_COST_HIGH,
        directory=config.INSTANCE_DIR
    )

    print("\nBatch generation complete!")