#!/usr/bin/env python3
"""Recoverable multi-GPU worker for RankMixer preference-margin runs."""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/rankmixer_preference_margin"
CONFIG = ROOT / "config_rankmixer_preference_margin"
PYTHON = ROOT / ".venv-unirank/bin/python"
SEEDS = (20262027, 20262028, 20262029, 20262030)
VARIANTS = ("Baseline", "FixedMargin", "ProgressiveMargin", "ReverseProgressiveMargin")


def experiments():
    # Interleave variants within each seed so early partial results remain paired.
    for seed in SEEDS:
        for variant in VARIANTS:
            yield f"RankMixer_Branch_{variant}_Seed{seed}"


def complete(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="replace")
    return "******** Test evaluation ********" in text and "Deleted model checkpoint:" in text


def claim(expid: str):
    lock_path = RUN_DIR / f"{expid}.lock"
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    for expid in experiments():
        log_path = RUN_DIR / f"{expid}.stdout.log"
        if complete(log_path):
            continue
        lock = claim(expid)
        if lock is None:
            continue
        try:
            if complete(log_path):
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
            with log_path.open("w") as output:
                result = subprocess.run(
                    [
                        str(PYTHON), "run_expid.py", "--config", str(CONFIG),
                        "--expid", expid, "--gpu", "0",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode != 0:
                raise SystemExit(
                    f"{expid} failed on physical GPU {args.gpu}; see {log_path}"
                )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


if __name__ == "__main__":
    main()
