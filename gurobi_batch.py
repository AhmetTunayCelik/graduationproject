"""
gurobi_batch.py
================

Batch solver: runs Gurobi on all instances in the instances/ folder,
saves optimal results as JSON files in results/optimal_n...json.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import apc_base as ab
from gurobi_solver import solve_instance
from parameters import config


def main():
    instance_dir = config.INSTANCE_DIR
    result_dir = config.RESULTS_DIR
    os.makedirs(result_dir, exist_ok=True)

    # Reproducibility appendix: log host + library + config state once per batch.
    meta_path = ab.write_run_metadata(result_dir, batch_label="gurobi")
    print(f"Run metadata written to {meta_path}")

    instance_paths = glob.glob(os.path.join(instance_dir, "instance_*.json"))
    if not instance_paths:
        print("No instance files found.")
        return

    solved = 0
    for fpath in instance_paths:
        base_name = os.path.basename(fpath)
        opt_name = base_name.replace("instance", "optimal")
        opt_path = os.path.join(result_dir, opt_name)

        if os.path.exists(opt_path):
            print(f"Skipping {base_name} (already solved)")
            continue

        print(f"Solving {base_name}...")
        instance = ab.load_instance(fpath)

        # Pull timeout limit directly from config
        result = solve_instance(instance, time_limit=config.GUROBI_TIME_LIMIT, verbose=False)

        # Atomic write: prevents truncated/empty result files if the process is
        # killed mid-write (which would otherwise cause skip-if-exists logic to
        # permanently orphan that cell of the experiment).
        ab._atomic_write_json(opt_path, result)

        solved += 1
        print(f"  Status: {result['status']}, Objective: {result['objective']}")

    print(f"\nDone. Solved {solved} new instances.")


if __name__ == "__main__":
    main()