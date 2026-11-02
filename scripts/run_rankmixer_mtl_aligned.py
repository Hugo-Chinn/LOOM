#!/usr/bin/env python3
"""One sequential worker per GPU, immutable config/source and completion stamps."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_snapshot(config, manifest):
    for name, expected in manifest["config_sha256"].items():
        if digest(config / name) != expected:
            raise RuntimeError(f"Configuration changed since freezing: {name}")
    for name, expected in manifest["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Source changed since freezing: {name}; rebuild before launching.")


def available_gpus(gpus):
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    uuids = {int(line.split(",")[0]): line.split(",")[1].strip()
             for line in inventory.splitlines()}
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    occupied = {line.split(",")[0].strip() for line in active.splitlines()}
    if any(gpu not in uuids for gpu in gpus):
        raise RuntimeError("Requested physical GPU indices do not exist.")
    return all(uuids[gpu] not in occupied for gpu in gpus)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_rankmixer_mtl_aligned")
    parser.add_argument("--study", choices=("historical_replication", "fixed_backbone_head", "all"),
                        default="fixed_backbone_head")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variants", help="Comma-separated manifest variant names.")
    parser.add_argument("--wait-for-gpus", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    manifest = json.loads((config / "manifest.json").read_text())
    validate_snapshot(config, manifest)
    gpus = [int(i) for i in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("GPU list must be nonempty and unique")
    arms = [n for n, a in manifest["arms"].items()
            if a["status"] == "ready" and (args.study == "all" or a["study"] == args.study)]
    if args.dataset:
        arms = [n for n in arms if manifest["arms"][n]["dataset"] == args.dataset]
    if args.seed is not None:
        arms = [n for n in arms if manifest["arms"][n]["seed"] == args.seed]
    if args.variants:
        requested = set(args.variants.split(","))
        arms = [n for n in arms if manifest["arms"][n]["variant"] in requested]
        selected = {manifest["arms"][n]["variant"] for n in arms}
        if selected != requested:
            parser.error(f"Missing or reference-blocked variants: {requested - selected}")
    if not arms:
        parser.error("No eligible arms matched.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.limit is not None:
        arms = arms[:args.limit]
    snapshot = digest(config / "manifest.json")
    logdir = ROOT / "logs/rankmixer_mtl_aligned" / snapshot[:12]
    jobs = Queue()
    for expid in arms:
        stamp = logdir / f"{expid}.done.json"
        if stamp.exists() and json.loads(stamp.read_text()).get("snapshot") == snapshot:
            continue
        jobs.put(expid)
    print(f"Study={args.study}; jobs={len(arms)}; snapshot={snapshot[:12]}", flush=True)
    print(f"Pending={jobs.qsize()}; GPUs={gpus}; shared work queue, one active run/GPU",
          flush=True)
    if args.dry_run:
        return
    if not args.wait_for_gpus and not available_gpus(gpus):
        raise RuntimeError("Requested GPUs are busy; no jobs launched.")
    logdir.mkdir(parents=True, exist_ok=True)

    def worker(gpu):
        failures = []
        with Path(f"/tmp/unirank_mtl_gpu_{gpu}.lock").open("a") as lease:
            while not jobs.empty():
                validate_snapshot(config, manifest)
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    if not args.wait_for_gpus:
                        raise RuntimeError(f"GPU {gpu} is leased")
                    time.sleep(15)
                    continue
                if not available_gpus([gpu]):
                    fcntl.flock(lease, fcntl.LOCK_UN)
                    time.sleep(15)
                    continue
                try:
                    expid = jobs.get_nowait()
                except Empty:
                    break
                print(f"START {expid} GPU={gpu}; remaining={jobs.qsize()}", flush=True)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                log = logdir / f"{expid}.log"
                with log.open("w") as output:
                    result = subprocess.run(
                        [sys.executable, "run_expid.py", "--config", str(config),
                         "--expid", expid, "--gpu", "0"],
                        cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
                sys.path.insert(0, str(ROOT / "scripts"))
                from summarize_all_local_experiments import parse_log
                parsed = parse_log(log)
                import yaml
                data = yaml.safe_load((config / "dataset_config.yaml").read_text())
                ds = manifest["arms"][expid]["dataset"]
                expected = {col["name"] for col in data[ds]["label_col"]}
                valid_metrics = (
                    parsed and set(parsed["metrics"]) == expected
                    and all(math.isfinite(metric["auc"]) and 0 <= metric["auc"] <= 1
                            and math.isfinite(metric["logloss"]) and metric["logloss"] >= 0
                            for metric in parsed["metrics"].values())
                )
                stamp = logdir / f"{expid}.done.json"
                if result.returncode or not valid_metrics:
                    failures.append(expid)
                    print(f"FAIL {expid} GPU={gpu}", flush=True)
                else:
                    stamp.write_text(json.dumps({"snapshot": snapshot, "expid": expid,
                                                 "metrics": parsed["metrics"]}, indent=2) + "\n")
                    print(f"DONE {expid} GPU={gpu}", flush=True)
                jobs.task_done()
                fcntl.flock(lease, fcntl.LOCK_UN)
        return failures

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        failures = [expid for future in futures for expid in future.result()]
    if failures:
        raise SystemExit(f"{len(failures)} failed arms: {failures}")


if __name__ == "__main__":
    main()
