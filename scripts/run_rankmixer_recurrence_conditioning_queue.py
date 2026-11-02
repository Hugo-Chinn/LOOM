#!/usr/bin/env python3
"""Run recurrence-conditioning screens serially on an explicitly chosen GPU."""
from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config_rankmixer_recurrence_conditioning"
OUT = ROOT / "runs_local/rankmixer_recurrence_conditioning"
PYTHON = ROOT / ".venv-unirank/bin/python"
SEED = 20262028
DATASETS = ("Taobao", "MerRec", "Kuai")
VARIANTS = (
    "BehaviorZero", "BehaviorNormal002", "BehaviorScaled01",
    "DepthChannelScale", "DepthSinusoidal", "BehaviorDepthMod",
    "BehaviorAdapter16",
)


def completed(log):
    return log.exists() and "******** Test evaluation ********" in log.read_text(errors="ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="*", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    queue_log = OUT / "queue.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    for dataset in args.datasets:
        for variant in args.variants:
            name = f"RankMixer_{dataset}_{variant}_S{SEED}"
            log = OUT / f"{name}.stdout.log"
            if completed(log):
                continue
            start = f"{datetime.now():%F %T} START gpu={args.gpu} {name}\n"
            with queue_log.open("a") as handle:
                handle.write(start)
            with log.open("w") as handle:
                result = subprocess.run(
                    [str(PYTHON), "run_expid.py", "--config", str(CONFIG),
                     "--expid", name, "--gpu", "0"],
                    cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                )
            status = "DONE" if result.returncode == 0 else f"FAILED({result.returncode})"
            with queue_log.open("a") as handle:
                handle.write(
                    f"{datetime.now():%F %T} {status} gpu={args.gpu} {name}\n"
                )
            if result.returncode != 0:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

