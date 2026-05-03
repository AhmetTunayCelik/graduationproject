"""
run_overnight.py
=================

Safe overnight wrapper for gurobi_batch.py.

Runs gurobi_batch in repeated bursts with cooldown breaks so the CPU
can breathe between sessions. The existing skip/resume mechanism in
gurobi_batch means any interrupted burst is picked up exactly where it
left off on the next cycle.

Cycle
-----
    [work burst: RUN_LIMIT_SECONDS]  ->  [cooldown: COOLDOWN_SECONDS]  ->  repeat

With the defaults (50 min work + 10 min cool) one full cycle = 1 hour.
At Gurobi's 10-min per-instance time limit that is ~5 instances per burst.

Usage
-----
    python run_overnight.py                 # default settings
    python run_overnight.py --run-limit 1800 --cooldown 300
    python run_overnight.py --log overnight.log
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WORK_SCRIPT = "gurobi_batch.py"
RUN_LIMIT_SECONDS = 3000   # 50 minutes — ~5 instances at max Gurobi time limit
COOLDOWN_SECONDS = 600     # 10 minutes — one "instance equivalent" of rest
LOG_FILE = "overnight.log"


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def _log(msg: str, log_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight Gurobi batch runner.")
    parser.add_argument(
        "--run-limit", type=int, default=RUN_LIMIT_SECONDS, metavar="SECS",
        help=f"Max wall-clock seconds per work burst (default: {RUN_LIMIT_SECONDS})",
    )
    parser.add_argument(
        "--cooldown", type=int, default=COOLDOWN_SECONDS, metavar="SECS",
        help=f"Cooldown seconds between bursts (default: {COOLDOWN_SECONDS})",
    )
    parser.add_argument(
        "--log", default=LOG_FILE, metavar="FILE",
        help=f"Append-only log file (default: {LOG_FILE})",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    run_limit = args.run_limit
    cooldown = args.cooldown

    _log(
        f"run_overnight started. Script={WORK_SCRIPT}, "
        f"burst={run_limit}s, cooldown={cooldown}s",
        log_path,
    )

    cycle = 0
    try:
        while True:
            cycle += 1
            _log(f"--- Cycle {cycle}: starting work burst (max {run_limit}s) ---", log_path)
            t0 = time.time()

            try:
                result = subprocess.run(
                    [sys.executable, WORK_SCRIPT],
                    timeout=run_limit,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.time() - t0
                _log(
                    f"Burst timed out after {elapsed:.0f}s "
                    f"(gurobi_batch will resume from where it left off).",
                    log_path,
                )
            else:
                elapsed = time.time() - t0
                if result.returncode == 0:
                    _log(
                        f"All instances solved (cycle {cycle}, {elapsed:.0f}s). "
                        "run_overnight finished.",
                        log_path,
                    )
                    return
                # Non-zero returncode (e.g. Python error in gurobi_batch)
                _log(
                    f"Burst ended with returncode={result.returncode} after {elapsed:.0f}s.",
                    log_path,
                )

            _log(f"Cooldown: sleeping {cooldown}s...", log_path)
            time.sleep(cooldown)

    except KeyboardInterrupt:
        _log("run_overnight interrupted by user (Ctrl+C). Stopping cleanly.", log_path)
        print("\nStopped. Re-run run_overnight.py to resume from the last completed instance.")


if __name__ == "__main__":
    main()
