#!/usr/bin/env python3
"""Fill shared GPU slots without interrupting immutable aligned experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from run_rankmixer_mtl_aligned import validate_snapshot
from summarize_all_local_experiments import parse_log

ROOT = Path(__file__).resolve().parents[1]


def gpu_inventory():
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    devices = {int(parts[0]): {"uuid": parts[1].strip(), "free": int(parts[2])}
               for line in rows.splitlines() if (parts := line.split(","))}
    rows = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True)
    for device in devices.values():
        device["pids"] = {int(parts[1]) for line in rows.splitlines()
                          if len(parts := line.split(",")) == 2
                          and parts[0].strip() == device["uuid"]}
    return devices


def running_expids():
    rows = subprocess.check_output(["ps", "-eo", "args"], text=True)
    active = set()
    for row in rows.splitlines():
        parts = row.split()
        if "run_expid.py" in parts and "--expid" in parts:
            active.add(parts[parts.index("--expid") + 1])
    return active


def slot_occupancy(device, running, gpu):
    own = {j["expid"] for j, g in running.values() if g == gpu}
    external = set()
    rows = subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True)
    commands = {}
    for row in rows.splitlines()[1:]:
        parts = row.split(maxsplit=2)
        if len(parts) == 3:
            commands[int(parts[0])] = (int(parts[1]), parts[2])
    candidates = set()
    for pid, (parent, command) in commands.items():
        if "run_expid.py " in command and "run_expid.py " not in commands.get(parent, (0, ""))[1]:
            try:
                env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
                visible = next((v.split(b"=", 1)[1].decode() for v in env
                                if v.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
                parts = command.split()
                logical = int(parts[parts.index("--gpu") + 1])
                physical = (int(visible.split(",")[logical])
                            if visible is not None else logical)
                if physical == gpu:
                    candidates.add(pid)
            except (OSError, ValueError, IndexError):
                pass
    for pid in candidates:
        try:
            parts = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            expid = parts[parts.index("--expid") + 1] if "--expid" in parts else None
        except (OSError, ValueError):
            expid = None
        if expid not in own:
            external.add(expid or pid)
    # nvidia-smi may expose host PIDs absent from this container's /proc.
    return max(len(device["pids"]), len(own) + len(external))


def load_jobs(configs):
    jobs = []
    for config in configs:
        config = config.resolve()
        manifest = json.loads((config / "manifest.json").read_text())
        validate_snapshot(config, manifest)
        snapshot = hashlib.sha256((config / "manifest.json").read_bytes()).hexdigest()
        data = yaml.safe_load((config / "dataset_config.yaml").read_text())
        logdir = ROOT / "logs/rankmixer_mtl_aligned" / snapshot[:12]
        logdir.mkdir(parents=True, exist_ok=True)
        for expid, arm in manifest["arms"].items():
            if arm["status"] != "ready":
                continue
            jobs.append({"config": config, "manifest": manifest, "snapshot": snapshot,
                         "logdir": logdir, "expid": expid,
                         "labels": {x["name"] for x in data[arm["dataset"]]["label_col"]}})
    # Interleave backbone task lists instead of starving one entire study.
    return sorted(jobs, key=lambda j: (j["expid"], str(j["config"])))


def complete(job):
    stamp = job["logdir"] / f"{job['expid']}.done.json"
    return stamp.exists() and json.loads(stamp.read_text()).get("snapshot") == job["snapshot"]


def adopt_finished(job):
    """Recover full test results from retained jobs whose scheduler was stopped."""
    log = job["logdir"] / f"{job['expid']}.log"
    if not log.exists():
        return False
    parsed = parse_log(log)
    if not parsed or set(parsed["metrics"]) != job["labels"]:
        return False
    text = log.read_text(errors="ignore")
    if "Traceback" in text or not all(
            math.isfinite(v["auc"]) and 0 <= v["auc"] <= 1
            and math.isfinite(v["logloss"]) and v["logloss"] >= 0
            for v in parsed["metrics"].values()):
        return False
    validate_snapshot(job["config"], job["manifest"])
    (job["logdir"] / f"{job['expid']}.done.json").write_text(json.dumps({
        "snapshot": job["snapshot"], "expid": job["expid"], "metrics": parsed["metrics"],
        "scheduler": "adopted_after_scheduler_handoff",
        "exit_code": None,
        "note": "Original scheduler stopped; process disappeared with complete finite test metrics and no traceback."
    }, indent=2) + "\n")
    print(f"ADOPT DONE {job['config'].name} {job['expid']}", flush=True)
    return True


def execute(job, gpu):
    expid = job["expid"]
    validate_snapshot(job["config"], job["manifest"])
    log = job["logdir"] / f"{expid}.log"
    print(f"START {job['config'].name} {expid} GPU={gpu}", flush=True)
    runtime = Path(job["manifest"].get("runtime_root", ROOT))
    with log.open("w") as output:
        result = subprocess.run([
            sys.executable, "run_expid.py", "--config", str(job["config"]),
            "--expid", expid, "--gpu", "0"], cwd=runtime,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=output, stderr=subprocess.STDOUT)
    parsed = parse_log(log)
    valid = parsed and set(parsed["metrics"]) == job["labels"] and all(
        math.isfinite(v["auc"]) and 0 <= v["auc"] <= 1
        and math.isfinite(v["logloss"]) and v["logloss"] >= 0
        for v in parsed["metrics"].values())
    if result.returncode or not valid:
        print(f"FAIL {expid} GPU={gpu} exit={result.returncode}", flush=True)
        return False
    stamp = job["logdir"] / f"{expid}.done.json"
    stamp.write_text(json.dumps({
        "snapshot": job["snapshot"], "expid": expid, "metrics": parsed["metrics"],
        "scheduler": "shared_gpu", "gpu": gpu}, indent=2) + "\n")
    print(f"DONE {expid} GPU={gpu}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-gpu", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=32768)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--configs", nargs="+", type=Path)
    parser.add_argument("--pilot-seed", type=int)
    args = parser.parse_args()
    if args.max_per_gpu < 1 or args.min_free_mib < 1:
        parser.error("Concurrency and memory thresholds must be positive")
    gpus = [int(g) for g in args.gpus.split(",")]
    if len(set(gpus)) != len(gpus):
        parser.error("GPU indices must be unique")
    jobs = load_jobs(args.configs or [ROOT / "config_rankmixer_mtl_aligned",
                                     ROOT / "config_onetrans_mtl_aligned"])
    if args.pilot_seed is not None:
        jobs.sort(key=lambda j: (
            j["manifest"]["arms"][j["expid"]]["seed"] != args.pilot_seed,
            j["expid"]))
    devices = gpu_inventory()
    for gpu in gpus:
        if gpu not in devices:
            parser.error(f"Missing GPU {gpu}")
    active = running_expids()
    adopted = [j for j in jobs if not complete(j) and j["expid"] in active]
    print(f"Ready={len(jobs)} complete={sum(complete(j) for j in jobs)} "
          f"max_per_gpu={args.max_per_gpu} min_free_mib={args.min_free_mib}", flush=True)
    for gpu in gpus:
        print(f"GPU {gpu}: {len(devices[gpu]['pids'])} processes, "
              f"{devices[gpu]['free']} MiB free", flush=True)
    if args.dry_run:
        return
    # Only one slot-filling manager may operate on this host.
    with Path("/tmp/unirank_mtl_shared_manager.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running, failed = {}, set()
        with ThreadPoolExecutor(max_workers=len(gpus) * args.max_per_gpu) as pool:
            while True:
                for future in list(running):
                    job, _ = running[future]
                    if future.done():
                        try:
                            success = future.result()
                        except Exception as error:
                            print(f"FAIL {job['expid']} dispatch: {error}", flush=True)
                            success = False
                        if not success:
                            failed.add((str(job["config"]), job["expid"]))
                        del running[future]
                pending = [j for j in jobs if not complete(j)
                           and (str(j["config"]), j["expid"]) not in failed]
                if not pending and not running:
                    break
                devices, active = gpu_inventory(), running_expids()
                for job in list(adopted):
                    if job["expid"] not in active:
                        adopt_finished(job)
                        adopted.remove(job)
                reserved_expids = {j["expid"] for j, _ in running.values()}
                def eligible(job):
                    if args.pilot_seed is None:
                        return True
                    arm = job["manifest"]["arms"][job["expid"]]
                    if arm["seed"] == args.pilot_seed:
                        return True
                    pilots = [p for p in jobs if p["config"] == job["config"]
                              and p["manifest"]["arms"][p["expid"]]["dataset"] == arm["dataset"]
                              and p["manifest"]["arms"][p["expid"]]["seed"] == args.pilot_seed]
                    return all(complete(p) for p in pilots)
                if not running and pending and not any(
                        eligible(j) and j["expid"] not in active for j in pending) and failed:
                    print("STOP: failed pilots block remaining seeds", flush=True)
                    break
                for gpu in gpus:
                    # Workers still starting may not yet appear in nvidia-smi.
                    occupancy = slot_occupancy(devices[gpu], running, gpu)
                    if occupancy >= args.max_per_gpu or devices[gpu]["free"] < args.min_free_mib:
                        continue
                    job = next((j for j in pending
                                if j["expid"] not in active | reserved_expids
                                and eligible(j)), None)
                    if job is None:
                        continue
                    future = pool.submit(execute, job, gpu)
                    running[future] = (job, gpu)
                    reserved_expids.add(job["expid"])
                time.sleep(15)
        if failed:
            raise SystemExit(f"{len(failed)} failed jobs; see log")


if __name__ == "__main__":
    main()
