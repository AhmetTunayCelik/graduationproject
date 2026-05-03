"""
gurobi_batch.py
================

Batch solver: runs Gurobi on all instances in the instances/ folder,
saves Gurobi results as JSON files in results/gurobi_n...json.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import apc_base as ab
from gurobi_solver import solve_instance
from parameters import config


def _optimal_result_filename(instance: ab.Instance) -> str:
    """Build Gurobi result filename from instance, category-aware.

    Naming convention (never changes even if category prefixes shift):
        standard:  gurobi_n{n}_a{alpha}_b{beta}_s{seed}.json
        difficult: difficult_gurobi_{category}_n{n}_a{alpha}_b{beta}_s{seed}.json
    """
    category = instance.get("instance_category", "standard")
    n = instance["n"]
    seed = instance["seed"]
    alpha_tag = f"{int(round(instance.get('graph_density', 1.0) * 10)):02d}"
    beta_tag = f"{int(round(instance.get('conflict_graph_density', 0) * 1000)):03d}"

    if category == "standard":
        return f"gurobi_n{n}_a{alpha_tag}_b{beta_tag}_s{seed}.json"
    else:
        return f"difficult_gurobi_{category}_n{n}_a{alpha_tag}_b{beta_tag}_s{seed}.json"


def main():
    instance_dir = config.INSTANCE_DIR
    result_dir = os.path.join(config.RESULTS_DIR, "gurobi")
    os.makedirs(result_dir, exist_ok=True)

    # Reproducibility appendix: log host + library + config state once per batch.
    meta_path = ab.write_run_metadata(result_dir, batch_label="gurobi")
    print(f"Run metadata written to {meta_path}")

    instance_paths = sorted(
        glob.glob(os.path.join(instance_dir, "instance_*.json")) +
        glob.glob(os.path.join(instance_dir, "difficult_instance_*.json"))
    )
    if not instance_paths:
        print("No instance files found.")
        return

    # Derive result filename from instance filename without loading the JSON.
    # Pattern:  instance_n...           -> gurobi_n...
    #           difficult_instance_cat_n... -> difficult_gurobi_cat_n...
    def _result_name_for_instance(inst_basename: str) -> str:
        if inst_basename.startswith("instance_"):
            return "gurobi_" + inst_basename[len("instance_"):]
        if inst_basename.startswith("difficult_instance_"):
            return "difficult_gurobi_" + inst_basename[len("difficult_instance_"):]
        return inst_basename

    # Phase 1: walk instances in order using only cheap existence checks
    # (no JSON parsing). Find the first instance whose result file is missing.
    # Track the most recent existing result so we can re-validate just that
    # one file at the end — that's where any interruption would have landed.
    first_missing_idx = len(instance_paths)
    last_existing_path: str = None
    last_existing_idx = -1
    for idx, fpath in enumerate(instance_paths):
        base_name = os.path.basename(fpath)
        rname = _result_name_for_instance(base_name)
        rpath = os.path.join(result_dir, rname)
        if os.path.exists(rpath):
            print(f"Skipping {base_name} (already solved)")
            last_existing_path = rpath
            last_existing_idx = idx
        else:
            first_missing_idx = idx
            break

    # Phase 2: check ONLY the last existing result. If it was an interrupted
    # run (Ctrl+C left an INTERRUPTED status, or a kill produced corrupt JSON),
    # rewind one step so that instance is redone.
    start_idx = first_missing_idx
    if last_existing_path is not None:
        try:
            with open(last_existing_path, "r") as f:
                last_status = json.load(f).get("status")
        except Exception:
            last_status = "CORRUPTED"
        if last_status in ("INTERRUPTED", "CORRUPTED"):
            print(f"Last result was {last_status}: redoing "
                  f"{os.path.basename(last_existing_path)}")
            start_idx = last_existing_idx

    if start_idx >= len(instance_paths):
        print("All instances already solved. Nothing to do.")
        return

    # Phase 3: solve from start_idx onward. Atomic write protects against
    # mid-write kills; the INTERRUPTED-status check below stops cleanly on Ctrl+C.
    solved = 0
    for fpath in instance_paths[start_idx:]:
        instance = ab.load_instance(fpath)
        base_name = os.path.basename(fpath)

        opt_name = _result_name_for_instance(base_name)
        opt_path = os.path.join(result_dir, opt_name)

        print(f"Solving {base_name}...")
        result = solve_instance(instance, time_limit=config.GUROBI_TIME_LIMIT, verbose=False)

        # Ctrl+C during model.optimize() is caught by Gurobi itself, which
        # returns status=INTERRUPTED instead of raising KeyboardInterrupt.
        # Do NOT persist the partial result (so re-running picks this up
        # again), and break out of the batch.
        if result.get("status") == "INTERRUPTED":
            print("  Interrupted by user. Not saving. Stopping batch.")
            break

        ab._atomic_write_json(opt_path, result)
        solved += 1
        print(f"  Status: {result['status']}, Objective: {result['objective']}")

    print(f"\nDone. Solved {solved} new instances.")


if __name__ == "__main__":
    main()