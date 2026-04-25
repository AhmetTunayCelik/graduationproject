"""
generate_many_instances.py
===========================

Generate a balanced set of MAX-APC instances using a full factorial design
over problem size (n), conflict density (beta), and random seeds.

This replaces the previous random-sampling approach.
"""

from instance_generator import generate_batch

# Desired total number of instances: 1000 (approx)
n_values = [10, 20, 30, 50, 75, 100]           # 6 values
conflict_graph_densities = [0.01, 0.05, 0.10, 0.20, 0.30]         # 5 values
# Choose number of seeds so that product ≈ 1000
# 6 * 5 * seeds = 30 * seeds ≈ 1000 → seeds ≈ 33.33
seeds = list(range(1, 34))                     # 33 seeds → 6*5*33 = 990 instances

if __name__ == "__main__":
    generate_batch(
        n_values=n_values,
        conflict_graph_densities=conflict_graph_densities,
        seeds=seeds,
        directory="instances",
        force=False,    # set to True if you want to overwrite existing instances
    )