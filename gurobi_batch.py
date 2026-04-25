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


def main():
    instance_dir = "instances"
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)

    instance_paths = glob.glob(os.path.join(instance_dir, "instance_*.json"))
    if not instance_paths:
        print("No instance files found.")
        return

    solved = 0
    for fpath in instance_paths:
        # Derive the optimal result filename
        base_name = os.path.basename(fpath)  # instance_n20_d1000_s42.json
        # Replace "instance" with "optimal"
        opt_name = base_name.replace("instance", "optimal")
        opt_path = os.path.join(result_dir, opt_name)

        if os.path.exists(opt_path):
            print(f"Skipping {base_name} (already solved)")
            continue

        print(f"Solving {base_name}...")
        instance = ab.load_instance(fpath)
        result = solve_instance(instance, time_limit=300.0, verbose=False)  # 5 min limit

        # Save as JSON
        with open(opt_path, "w") as f:
            json.dump(result, f, indent=2)

        solved += 1
        print(f"  Status: {result['status']}, Objective: {result['objective']}")

    print(f"\nDone. Solved {solved} new instances.")


if __name__ == "__main__":
    main()