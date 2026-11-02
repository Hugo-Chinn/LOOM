#!/usr/bin/env python3
"""Wait for a stably idle GPU, then run a phase-3 worker."""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/rankmixer_preference_margin_phase3"
PYTHON = ROOT / ".venv-unirank/bin/python"
WORKER = ROOT / "scripts/run_rankmixer_preference_margin_phase3_worker.py"


def stats(gpu):
    result = subprocess.run(
        ["nvidia-smi", "--id", str(gpu),
         "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    memory, utilization = next(csv.reader(io.StringIO(result.stdout)))
    return int(memory), int(utilization)


def complete_count():
    count = 0
    for path in RUN_DIR.glob("RankMixer_Branch_*.stdout.log"):
        text = path.read_text(errors="replace")
        if "******** Test evaluation ********" in text and "Deleted model checkpoint:" in text:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--memory-threshold", type=int, default=2048)
    parser.add_argument("--utilization-threshold", type=int, default=10)
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    lock = (RUN_DIR / f"gpu{args.gpu}.worker.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0

    stable = 0
    try:
        while complete_count() < 16:
            memory, utilization = stats(args.gpu)
            if (memory < args.memory_threshold
                    and utilization < args.utilization_threshold):
                stable += 1
            else:
                stable = 0
            if stable >= args.stable_samples:
                return subprocess.run(
                    [str(PYTHON), str(WORKER), "--gpu", str(args.gpu)],
                    cwd=ROOT,
                ).returncode
            time.sleep(args.poll_seconds)
        return 0
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
