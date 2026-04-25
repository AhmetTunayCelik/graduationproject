"""
generate_many_instances.py
===========================

Generate a balanced set of MAX-APC instances using a full factorial design
over problem size (n), conflict density (beta), and random seeds.

Beta is scaled down for larger n because the professor's formula

    |E(C)| = (beta / 2) * n^4

grows extremely fast. The same beta that produces ~4k conflicts at n=10
produces ~450M conflicts at n=100. Each n-group uses a beta range that
keeps |E(C)| in a computationally feasible range (~1k-25k conflicts).

    n in {10, 20} : beta in {0.001, 0.005, 0.010, 0.020, 0.050}
    n in {30, 50} : beta in {0.0002, 0.0005, 0.001, 0.003, 0.008}
    n in {75,100} : beta in {0.00005, 0.0001, 0.00015, 0.0002, 0.0003}

Seeds are also reduced for larger n to keep total runtime manageable:
    n in {10, 20} : 10 seeds  -> 2*5*10 = 100 instances
    n in {30, 50} :  7 seeds  -> 2*5*7  =  70 instances
    n in {75,100} :  5 seeds  -> 2*5*5  =  50 instances

Total: 220 instances
"""

from instance_generator import generate_batch

if __name__ == "__main__":

    # Small instances — wide beta range, more seeds
    generate_batch(
        n_values=[10, 20],
        conflict_graph_densities=[0.001, 0.005, 0.010, 0.020, 0.050],
        seeds=list(range(1, 11)),
        directory="instances",
        force=False,
    )

    # Medium instances — narrower beta range, fewer seeds
    generate_batch(
        n_values=[30, 50],
        conflict_graph_densities=[0.0002, 0.0005, 0.001, 0.003, 0.008],
        seeds=list(range(1, 8)),
        directory="instances",
        force=False,
    )

    # Large instances — tight beta range, fewest seeds
    generate_batch(
        n_values=[75, 100],
        conflict_graph_densities=[0.00005, 0.0001, 0.00015, 0.0002, 0.0003],
        seeds=list(range(1, 6)),
        directory="instances",
        force=False,
    )