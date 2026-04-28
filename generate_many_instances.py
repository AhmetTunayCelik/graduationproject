"""
generate_many_instances.py
===========================

Generate the standard MAX-APC benchmark library using a unified factorial grid
over problem size (n), conflict density (beta), graph edge density (alpha), and
random seeds.

Grid dimensions:
    n      : GRID_N           = [20, 30, 40, 50]
    beta   : GRID_BETAS       = [0.01, 0.05, 0.10, 0.15]
    alpha  : GRAPH_DENSITIES  = [0.4, 0.6, 0.8, 1.0]
    seeds  : UNIVERSAL_SEEDS  = 10 seeds

Total: 4 * 4 * 4 * 10 = 640 instances
"""

from instance_generator import generate_batch
from parameters import config

if __name__ == "__main__":
    created, skipped = generate_batch(
        n_values=config.GRID_N,
        conflict_graph_densities=config.GRID_BETAS,
        graph_densities=config.GRAPH_DENSITIES,
        seeds=config.UNIVERSAL_SEEDS,
        directory=config.INSTANCE_DIR,
        instance_category="standard",
        force=False,
    )
    print(f"Standard grid: {created} created, {skipped} skipped.")
