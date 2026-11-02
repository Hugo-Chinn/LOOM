#!/usr/bin/env python3
"""Run the three Taobao sample-trust screens on explicitly selected GPUs."""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config_rankmixer_sample_trust"
OUT = ROOT / "runs_local/rankmixer_sample_trust"
PYTHON = ROOT / ".venv-unirank/bin/python"
SEED = 20262028
VARIANTS = ("Stage", "Shared", "StageKeep")


def run_one(gpu, variant):
    expid = f"RankMixer_Taobao_SampleTrust{variant}_S{SEED}"
    log = OUT / f"{expid}.stdout.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with log.open("w") as handle:
        return subprocess.run(
            [str(PYTHON), "run_expid.py", "--config", str(CONFIG),
             "--expid", expid, "--gpu", "0"],
            cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
        ).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if len(args.gpus) < len(VARIANTS):
        raise ValueError("Provide at least three GPUs for the Taobao screen")
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as pool:
        futures = {
            pool.submit(run_one, gpu, variant): variant
            for gpu, variant in zip(args.gpus, VARIANTS)
        }
        failed = 0
        for future in as_completed(futures):
            failed = max(failed, future.result())
    raise SystemExit(failed)


if __name__ == "__main__":
    main()

