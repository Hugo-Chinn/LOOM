#!/usr/bin/env python3
"""Wait for a genuinely idle physical GPU, then run one experiment worker."""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/rankmixer_preference_margin"
WORKER = ROOT / "scripts/run_rankmixer_preference_margin_worker.py"
PYTHON = ROOT / ".venv-unirank/bin/python"


def gpu_stats():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = csv.reader(io.StringIO(result.stdout))
    return [(int(index), int(memory), int(utilization))
            for index, memory, utilization in rows]


def completed_runs():
    count = 0
    for path in RUN_DIR.glob("RankMixer_Branch_*.stdout.log"):
        text = path.read_text(errors="replace")
        if "******** Test evaluation ********" in text and "Deleted model checkpoint:" in text:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", default="0")
    parser.add_argument("--memory-threshold", type=int, default=1024)
    parser.add_argument("--utilization-threshold", type=int, default=10)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    excluded = {int(value) for value in args.exclude.split(",") if value.strip()}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    stable = {}

    while completed_runs() < 16:
        for gpu, memory, utilization in gpu_stats():
            if gpu in excluded:
                continue
            idle = (memory < args.memory_threshold
                    and utilization < args.utilization_threshold)
            stable[gpu] = stable.get(gpu, 0) + 1 if idle else 0
            if stable[gpu] < args.stable_samples:
                continue
            lock = (RUN_DIR / f"gpu{gpu}.worker.lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            # Recheck after taking the lock to close the race between waiters.
            current = {index: (mem, util) for index, mem, util in gpu_stats()}
            mem, util = current[gpu]
            if mem >= args.memory_threshold or util >= args.utilization_threshold:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
                stable[gpu] = 0
                continue
            try:
                return subprocess.run(
                    [str(PYTHON), str(WORKER), "--gpu", str(gpu)],
                    cwd=ROOT,
                ).returncode
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
