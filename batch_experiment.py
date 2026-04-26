"""
batch_experiment.py
====================

Parameter-sweep runner for large-scale MAX-APC experiments.
"""

from __future__ import annotations

import argparse
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
        if not fname.startswith("instance_n") or not fname.endswith(".json"):
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
    subgradient_cache_dir: str = "results",
) -> bool:
    """Run a single heuristic on a single instance (subgradient cached)."""
    result_path = os.path.join(result_dir, ab._result_filename(instance, heuristic_name))
    if not force_heuristic and os.path.exists(result_path):
        return False

    subg = ab.load_subgradient_result(instance, directory=subgradient_cache_dir)
    if subg is None:
        try:
            default_module = importlib.import_module("heuristics.lagrangean_repair")
            repair_fn = default_module.run
        except ImportError:
            repair_fn = None
        subg = ab.subgradient_solve(
            instance,
            repair_fn=repair_fn,
            K_max=config.SUBG_MAX_ITERS,
            verbose=False,
        )
        ab.save_subgradient_result(instance, subg, directory=subgradient_cache_dir)

    x_star = subg["x_star_final"]
    cost = instance["cost_matrix"]
    conflicts = instance["conflicts"]
    n = instance["n"]
    E0 = instance["E0"]

    result_payload = {
        "subgradient_LB": subg.get("LB"),
        "subgradient_UB": subg.get("UB"),
        "subgradient_iterations": subg.get("iterations"),
        "subgradient_runtime": subg.get("runtime_seconds"),
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

    ab.save_result(instance, heuristic_name, result_payload, directory=result_dir)
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner for MAX-APC.")
    parser.add_argument("--heuristics", nargs="+", help="List of heuristic names to run (default: all discovered)")
    parser.add_argument("--list-heuristics", action="store_true", help="List available heuristics and exit")
    parser.add_argument("--n-values", type=int, nargs="+", help="Filter instances by n")
    parser.add_argument("--densities", type=float, nargs="+", help="Filter instances by density")
    parser.add_argument("--instance-dir", default="instances", help="Directory containing instance JSONs")
    parser.add_argument("--result-dir", default="results", help="Directory for results and subgradient cache")
    parser.add_argument("--force-heuristic", action="store_true", help="Rerun heuristic even if result exists")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

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
    for inst in instances:
        for hname, hmod, hrun, has_ord in to_run:
            if run_single_combination(inst, hname, hmod, hrun, has_ord,
                                      args.result_dir, args.force_heuristic):
                saved += 1
            total_heuristic_calls += 1
            if not args.quiet and (total_heuristic_calls % 10 == 0):
                print(f"Processed {total_heuristic_calls} heuristic/instance pairs...")

    print(f"\nBatch finished. Saved {saved} new results (out of {total_heuristic_calls} calls).")


if __name__ == "__main__":
    main()