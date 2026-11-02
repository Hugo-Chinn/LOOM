#!/usr/bin/env python3
"""Run the frozen shared sample-trust gate on MerRec and KuaiRand."""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config_rankmixer_sample_trust"
OUT = ROOT / "runs_local/rankmixer_sample_trust"
PYTHON = ROOT / ".venv-unirank/bin/python"
SEED = 20262028
DATASETS = ("MerRec", "Kuai")


def run_one(gpu, dataset):
    expid = f"RankMixer_{dataset}_SampleTrustShared_S{SEED}"
    log = OUT / f"{expid}.stdout.log"
    with tempfile.TemporaryDirectory(
            prefix=f"torchinductor_{dataset.lower()}_",
            dir=OUT) as cache_dir:
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            TORCHINDUCTOR_CACHE_DIR=cache_dir,
        )
        with log.open("w") as handle:
            return subprocess.run(
                [str(PYTHON), "run_expid.py", "--config", str(CONFIG),
                 "--expid", expid, "--gpu", "0"],
                cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            ).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs=2, type=int, default=[0, 1])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(run_one, gpu, dataset): dataset
            for gpu, dataset in zip(args.gpus, DATASETS)
        }
        failed = 0
        for future in as_completed(futures):
            failed = max(failed, future.result())
    raise SystemExit(failed)


if __name__ == "__main__":
    main()
