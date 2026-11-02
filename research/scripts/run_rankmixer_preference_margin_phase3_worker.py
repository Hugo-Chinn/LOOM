#!/usr/bin/env python3
"""Recoverable worker for phase-3 preference-margin calibration."""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/rankmixer_preference_margin_phase3"
CONFIG = ROOT / "config_rankmixer_preference_margin_phase3"
PYTHON = ROOT / ".venv-unirank/bin/python"
SEEDS = (20262027, 20262028, 20262029, 20262030)
VARIANTS = ("W001M000", "W001M005", "W001M010", "W00075M020")


def complete(path):
    if not path.exists():
        return False
    text = path.read_text(errors="replace")
    return "******** Test evaluation ********" in text and "Deleted model checkpoint:" in text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for variant in VARIANTS:
            expid = f"RankMixer_Branch_{variant}_Seed{seed}"
            output_path = RUN_DIR / f"{expid}.stdout.log"
            if complete(output_path):
                continue
            lock = (RUN_DIR / f"{expid}.lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                if complete(output_path):
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
                with output_path.open("w") as output:
                    result = subprocess.run(
                        [str(PYTHON), "run_expid.py", "--config", str(CONFIG),
                         "--expid", expid, "--gpu", "0"],
                        cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                    )
                if result.returncode:
                    raise SystemExit(f"{expid} failed; see {output_path}")
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()


if __name__ == "__main__":
    main()
