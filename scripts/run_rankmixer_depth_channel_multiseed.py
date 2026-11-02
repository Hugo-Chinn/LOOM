#!/usr/bin/env python3
"""Run remaining depth-channel confirmation jobs across explicitly free GPUs."""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config_rankmixer_depth_channel_multiseed"
OUT = ROOT / "runs_local/rankmixer_depth_channel_multiseed"
PYTHON = ROOT / ".venv-unirank/bin/python"
SEEDS = (20262027, 20262029, 20262030)


def completed(log):
    return log.exists() and "******** Test evaluation ********" in log.read_text(errors="ignore")


def run_worker(gpu, jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    for dataset, seed in jobs:
        expid = f"RankMixer_{dataset}_DepthChannelScale_S{seed}"
        log = OUT / f"{expid}.stdout.log"
        if completed(log):
            continue
        with (OUT / "queue.log").open("a") as handle:
            handle.write(f"{datetime.now():%F %T} START gpu={gpu} {expid}\n")
        with log.open("w") as handle:
            result = subprocess.run(
                [str(PYTHON), "run_expid.py", "--config", str(CONFIG),
                 "--expid", expid, "--gpu", "0"],
                cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            )
        status = "DONE" if result.returncode == 0 else f"FAILED({result.returncode})"
        with (OUT / "queue.log").open("a") as handle:
            handle.write(f"{datetime.now():%F %T} {status} gpu={gpu} {expid}\n")
        if result.returncode:
            return result.returncode
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # Finish cheaper confirmations first, then dedicate both workers to the
    # long KuaiRand jobs. Each worker remains strictly serial.
    if len(args.gpus) == 2:
        # Approximate costs: Taobao 4 min, MerRec 40 min, KuaiRand 100 min.
        # This split is about 208 vs 224 minutes instead of round-robin's
        # imbalanced 248 vs 184 minutes.
        buckets = [
            [("Taobao", 20262027), ("Kuai", 20262027),
             ("Taobao", 20262030), ("Kuai", 20262030)],
            [("MerRec", 20262027), ("Taobao", 20262029),
             ("MerRec", 20262029), ("Kuai", 20262029),
             ("MerRec", 20262030)],
        ]
    else:
        jobs = [
            ("Taobao", 20262027), ("MerRec", 20262027), ("Kuai", 20262027),
            ("Taobao", 20262029), ("MerRec", 20262029), ("Kuai", 20262029),
            ("Taobao", 20262030), ("MerRec", 20262030), ("Kuai", 20262030),
        ]
        buckets = [[] for _ in args.gpus]
        for index, job in enumerate(jobs):
            buckets[index % len(buckets)].append(job)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = {
            pool.submit(run_worker, gpu, bucket): gpu
            for gpu, bucket in zip(args.gpus, buckets)
        }
        failed = 0
        for future in as_completed(futures):
            failed = max(failed, future.result())
    raise SystemExit(failed)


if __name__ == "__main__":
    main()
