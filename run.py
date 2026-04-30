"""
run.py
======

Single-instance runner for MAX-APC.
"""

from __future__ import annotations

import argparse
import os
import time

import apc_base as ab
from batch_experiment import discover_heuristics
from parameters import config


def main():
    parser = argparse.ArgumentParser(description="Single‑instance MAX-APC runner")
    parser.add_argument("n", type=int, default=20, nargs="?",
                        help="Problem size")
    parser.add_argument("density", type=float, default=0.1, nargs="?",
                        help="Conflict density relative to n²")
    parser.add_argument("seed", type=int, default=None, nargs="?",
                        help="Random seed (time‑derived if omitted)")
    parser.add_argument("--heuristics", nargs="+", help="Heuristics to run (default: all discovered)")
    parser.add_argument("--instance-dir", default="instances")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--force", action="store_true", help="Rerun even if result exists")
    parser.add_argument("--quiet", action="store_true", help="Suppress subgradient output")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(time.time() * 1000) % (2**31)

    # Generate or load instance
    from instance_generator import generate_instance
    num_conflicts = max(1, int(args.density * args.n * args.n))
    instance = generate_instance(
        n=args.n, num_conflicts=num_conflicts, seed=args.seed, density=args.density
    )
    ab.save_instance(instance, args.instance_dir)
    print(f"Instance: n={args.n}, density={args.density}, seed={args.seed}")

    available = discover_heuristics()
    if args.heuristics:
        to_run = [(name, mod, run_fn, has_ord) for name, (mod, run_fn, has_ord) in available.items()
                  if name in args.heuristics]
    else:
        to_run = [(name, mod, run_fn, has_ord) for name, (mod, run_fn, has_ord) in available.items()]

    print(f"\nRunning {len(to_run)} heuristic(s): {[name for name,_,_,_ in to_run]}")
    for hname, hmod, hrun, has_ord in to_run:
        # Each heuristic runs its own subgradient ascent fresh (no cache)
        # using itself as the repair_fn so it pays the full CPU cost.
        subg = ab.subgradient_solve(
            instance,
            repair_fn=hrun,
            K_max=config.SUBG_MAX_ITERS,
            verbose=not args.quiet,
        )

        t0 = time.time()
        if has_ord:
            variants = hmod.run_all_orderings(
                subg["x_star_final"],
                instance["cost_matrix"],
                instance["conflicts"],
                instance["n"],
                instance["E0"],
                lambdas=subg.get("lambdas_final"),
                graph_edges=instance.get("graph_edges"),
            )
            elapsed = time.time() - t0
            result_payload = {
                "subgradient_LB": subg.get("LB"),
                "subgradient_UB": subg.get("UB"),
                "subgradient_iterations": subg.get("iterations"),
                "subgradient_runtime": subg.get("runtime_seconds"),
                "subgradient_terminated_reason": subg.get("terminated_reason"),
                "subgradient_history": subg.get("iteration_history"),
                "heuristic_output": {
                    "ordering_variants": variants,
                    "runtime_seconds": elapsed,
                },
            }
        else:
            try:
                assignment, obj, feasible = hrun(
                    subg["x_star_final"],
                    instance["cost_matrix"],
                    instance["conflicts"],
                    instance["n"],
                    instance["E0"],
                    lambdas=subg.get("lambdas_final"),
                    graph_edges=instance.get("graph_edges"),
                )
            except TypeError:
                assignment, obj, feasible = hrun(
                    subg["x_star_final"],
                    instance["cost_matrix"],
                    instance["conflicts"],
                    instance["n"],
                )
            elapsed = time.time() - t0
            result_payload = {
                "subgradient_LB": subg.get("LB"),
                "subgradient_UB": subg.get("UB"),
                "subgradient_iterations": subg.get("iterations"),
                "subgradient_runtime": subg.get("runtime_seconds"),
                "subgradient_terminated_reason": subg.get("terminated_reason"),
                "subgradient_history": subg.get("iteration_history"),
                "heuristic_output": {
                    "assignment": assignment,
                    "objective": obj,
                    "feasible": feasible,
                    "runtime_seconds": elapsed,
                },
            }
        ab.save_result(instance, hname, result_payload, directory=args.result_dir, subgradient_output=subg)
        print(f"  {hname}: subgradient {subg['runtime_seconds']:.3f}s + heuristic {elapsed:.3f}s")


if __name__ == "__main__":
    main()