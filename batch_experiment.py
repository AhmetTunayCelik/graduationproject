"""
batch_experiment.py
====================

Parameter-sweep runner for large-scale MAX-APC experiments.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import apc_base as ab
from parameters import config


# -----------------------------------------------------------------------------
# Dynamic heuristic discovery (exported for run.py)
# -----------------------------------------------------------------------------
def discover_heuristics() -> Dict[str, Tuple[str, any]]:
    """Scan the heuristics/ directory and return a dict name -> (module, run_func, has_orderings)."""
    heuristics_dir = os.path.join(os.path.dirname(__file__), "heuristics")
    if not os.path.isdir(heuristics_dir):
        return {}
    discovered = {}
    for filename in os.listdir(heuristics_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            try:
                module = importlib.import_module(f"heuristics.{module_name}")
                if hasattr(module, "HEURISTIC_NAME") and hasattr(module, "run"):
                    name = module.HEURISTIC_NAME
                    has_orderings = hasattr(module, "run_all_orderings")
                    discovered[name] = (module, module.run, has_orderings)
            except Exception as e:
                print(f"Warning: could not load heuristic {module_name}: {e}", file=sys.stderr)
    return discovered


def enumerate_instances(
    instance_dir: str,
    n_values: Optional[List[int]] = None,
    densities: Optional[List[float]] = None,
) -> List[ab.Instance]:
    """List all instance dictionaries from the instance directory."""
    if not os.path.isdir(instance_dir):
        return []
    instances = []
    for fname in os.listdir(instance_dir):
        if not (fname.startswith("instance_") or fname.startswith("difficult_instance_")) or not fname.endswith(".json"):
            continue
        fpath = os.path.join(instance_dir, fname)
        try:
            inst = ab.load_instance(fpath)
            if n_values is not None and inst["n"] not in n_values:
                continue
            if densities is not None and inst["density"] not in densities:
                continue
            instances.append(inst)
        except Exception:
            print(f"Warning: could not load {fpath}", file=sys.stderr)
    return instances


def run_single_combination(
    instance: ab.Instance,
    heuristic_name: str,
    heuristic_module,
    heuristic_run,
    has_orderings: bool,
    result_dir: str,
    force_heuristic: bool = False,
) -> bool:
    """Run a single heuristic on a single instance.

    Each call performs a fresh in-memory subgradient ascent (no cache) using
    the heuristic itself as the repair_fn, so that the heuristic pays the
    full CPU cost of producing its own dual bounds. This makes runtime
    comparisons against Gurobi fair.
    """
    result_path = os.path.join(result_dir, ab._result_filename(instance, heuristic_name))
    if not force_heuristic and os.path.exists(result_path):
        return False

    cost = instance["cost_matrix"]
    conflicts = instance["conflicts"]
    n = instance["n"]
    E0 = instance["E0"]

    subg = ab.subgradient_solve(
        instance,
        repair_fn=heuristic_run,
        K_max=config.SUBG_MAX_ITERS,
        verbose=False,
    )

    x_star = subg["x_star_final"]

    result_payload = {
        "subgradient_LB": subg.get("LB"),
        "subgradient_UB": subg.get("UB"),
        "subgradient_iterations": subg.get("iterations"),
        "subgradient_runtime": subg.get("runtime_seconds"),
        "subgradient_terminated_reason": subg.get("terminated_reason"),
        "subgradient_history": subg.get("iteration_history"),
        "heuristic_name": heuristic_name,
    }

    # Use run_all_orderings if available (for heuristics with ordering variants)
    if has_orderings:
        start_time = time.time()
        variants = heuristic_module.run_all_orderings(
            x_star, cost, conflicts, n, E0
        )
        elapsed = time.time() - start_time
        result_payload["heuristic_output"] = {
            "ordering_variants": variants,
            "runtime_seconds": elapsed,
        }
    else:
        # Simple heuristic (single output)
        start_time = time.time()
        try:
            assignment, objective, feasible = heuristic_run(
                x_star, cost, conflicts, n, E0, lambdas=subg.get("lambdas_final")
            )
        except TypeError:
            assignment, objective, feasible = heuristic_run(x_star, cost, conflicts, n)
        elapsed = time.time() - start_time
        result_payload["heuristic_output"] = {
            "assignment": assignment,
            "objective": objective,
            "feasible": bool(feasible),
            "runtime_seconds": elapsed,
        }

    ab.save_result(instance, heuristic_name, result_payload, directory=result_dir, subgradient_output=subg)

    # Free large in-memory data (subgradient buffers, conflict adjacency, etc.)
    # before the next heuristic-instance pair to prevent OS-level swapping under
    # accumulated garbage when running long batches.
    del subg, x_star, result_payload
    gc.collect()

    return True


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner for MAX-APC.")
    parser.add_argument("--heuristics", nargs="+", help="List of heuristic names to run (default: all discovered)")
    parser.add_argument("--list-heuristics", action="store_true", help="List available heuristics and exit")
    parser.add_argument("--n-values", type=int, nargs="+", help="Filter instances by n")
    parser.add_argument("--densities", type=float, nargs="+", help="Filter instances by density")
    parser.add_argument("--instance-dir", default="instances", help="Directory containing instance JSONs")
    parser.add_argument("--result-dir", default="results", help="Directory for heuristic result JSONs")
    parser.add_argument("--force-heuristic", action="store_true", help="Rerun heuristic even if result exists")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--tracemalloc", action="store_true",
                        help="Enable tracemalloc and print top-10 memory hotspots at the end")
    args = parser.parse_args()

    if args.tracemalloc:
        import tracemalloc
        tracemalloc.start()

    available = discover_heuristics()
    if args.list_heuristics:
        print("Available heuristics:")
        for name, (_, _, has_ord) in sorted(available.items()):
            ordering_info = " (with ordering variants)" if has_ord else ""
            print(f"  {name}{ordering_info}")
        return

    if args.heuristics:
        to_run = [(name, mod, run_fn, has_ord) for name, (mod, run_fn, has_ord) in available.items()
                  if name in args.heuristics]
        missing = set(args.heuristics) - set(available.keys())
        if missing:
            print(f"Warning: unknown heuristic(s): {missing}", file=sys.stderr)
    else:
        to_run = [(name, mod, run_fn, has_ord) for name, (mod, run_fn, has_ord) in available.items()]

    if not to_run:
        print("No heuristics selected. Exiting.")
        return

    instances = enumerate_instances(args.instance_dir, args.n_values, args.densities)
    if not instances:
        print(f"No instances found in {args.instance_dir}. Please run instance_generator.py first.")
        return

    total_heuristic_calls = 0
    saved = 0

    # Process one heuristic across all instances, then move to next heuristic
    for hname, hmod, hrun, has_ord in to_run:
        if not args.quiet:
            print(f"\n{hname}: processing {len(instances)} instances...")

        for inst in instances:
            # Each run_single_combination checks if result exists and skips if present
            is_new = run_single_combination(inst, hname, hmod, hrun, has_ord,
                                            args.result_dir, args.force_heuristic)
            if is_new:
                saved += 1
                status = "generated"
            else:
                status = "skipped"
            total_heuristic_calls += 1

            if not args.quiet:
                instance_name = f"n{inst['n']}_a{int(round((inst.get('graph_density', 1.0) or 1.0) * 10)):02d}_b{int(round(inst.get('conflict_graph_density', 0) * 1000)):03d}_s{inst['seed']}"
                print(f"  [{total_heuristic_calls}] {instance_name:40s} → {status}")

    print(f"\nBatch finished. Saved {saved} new results (out of {total_heuristic_calls} calls).")

    if args.tracemalloc:
        import tracemalloc
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        print("\n[tracemalloc] Top 10 memory hotspots:")
        for stat in top_stats[:10]:
            print(f"  {stat}")
        tracemalloc.stop()


if __name__ == "__main__":
    main()