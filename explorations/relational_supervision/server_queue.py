"""Bounded two-dataset Phase-1 queue; per-GPU slots and one pilot per dataset.

Uses a resource journal to pause only assigned holder UUIDs and restore their
previous pause state on normal exit, failure, deadline or SIGTERM. SIGKILL or
host failure requires manual recovery from resources.json. No automatic retries.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HOLDER = Path("/jizhi/jizhi2/worker/trainer/rsi_holder/rsi_holder.py")
STOP = False


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def holder(command, uuid=None):
    proc = subprocess.run([sys.executable, str(HOLDER), command] + ([uuid] if uuid else []),
                          capture_output=True, text=True, timeout=45, check=True)
    return json.loads(proc.stdout)


def resources(root, indices):
    journal = root / "resources.json"
    if journal.exists():
        state = json.loads(journal.read_text())
        if [g["index"] for g in state["gpus"]] != indices or state.get("restored"):
            raise ValueError("resource journal does not match this allocation")
    else:
        before = holder("status")
        paused_path = HOLDER.parent / "paused.json"
        paused = json.loads(paused_path.read_text()).get("uuids", []) if paused_path.exists() else []
        state = {"before": before, "paused_before": paused,
                 "gpus": [g for g in before["gpus"] if g["index"] in indices], "restored": False}
        if sorted(g["index"] for g in state["gpus"]) != indices:
            raise ValueError("holder inventory missing assigned GPUs")
        write_json(journal, state)
    for gpu in state["gpus"]:
        result = holder("pause", gpu["uuid"])
        if not result["paused"]:
            raise RuntimeError(f"holder did not release {gpu['uuid']}")
    # A holder pause only clears holder PIDs, not another workload.
    proc = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                           "--format=csv,noheader"], capture_output=True, text=True, check=True)
    assigned = {g["uuid"] for g in state["gpus"]}
    if any(line.split(",")[0].strip() in assigned for line in proc.stdout.splitlines()):
        raise RuntimeError("assigned GPU still has a compute process; refusing launch")
    return state


def restore(root):
    path = root / "resources.json"
    if not path.exists():
        return
    state = json.loads(path.read_text())
    errors = []
    for gpu in state["gpus"]:
        if gpu["uuid"] not in state["paused_before"]:
            try:
                holder("resume", gpu["uuid"])
            except Exception as exc:
                errors.append(str(exc))
    state.update(restored=not errors, restore_errors=errors)
    write_json(path, state)


def env_for(root, uuid, seed):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": uuid,
            "RELSUP_RUN_ROOT": str(root), "PYTHONHASHSEED": str(seed),
            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2", "POLARS_MAX_THREADS": "2",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor_cache"),
            "PYTHONUNBUFFERED": "1"}


def terminate(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--configs", nargs=2, type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--slots", type=int, default=2, choices=[1, 2])
    parser.add_argument("--max-runs", type=int, default=40)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--run-hours", type=float, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "scheduler.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.restore_only:
        restore(root)
        return
    if args.gpus != sorted(set(args.gpus)) or not set(args.gpus) <= set(range(6)):
        raise ValueError("this study is authorized only on unique GPU indices 0–5")
    try:
        allocation = resources(root, args.gpus)
    except Exception:
        restore(root)
        raise
    if args.prepare_only:
        print(json.dumps(allocation, indent=2))
        return
    if not args.configs or args.hours <= 0 or args.run_hours <= 0:
        raise ValueError("two configs and positive budgets required")
    if (root / "queue.json").exists():
        raise ValueError("queue already exists; inspect existing attempts before restarting")
    jobs, records = [], []
    for config in args.configs:
        config = config.resolve()
        record = json.loads((config / "protocol.json").read_text())
        if record["phase"] != "1":
            raise ValueError("only Phase 1 is enabled in this queue")
        records.append(record)
        for expid in (config / "expids.txt").read_text().splitlines():
            jobs.append({"dataset": record["dataset_id"], "expid": expid,
                         "config": str(config), "status": "pending",
                         "pilot": expid == "full_S20262027"})
    if len(jobs) != 40 or len(jobs) > args.max_runs:
        raise ValueError(f"expected 40 runs within budget, got {len(jobs)}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    state = {"pid": os.getpid(), "commit": commit, "started": time.time(),
             "max_runs": args.max_runs, "hours": args.hours, "run_hours": args.run_hours,
             "slots_per_gpu": args.slots, "allocation": allocation["gpus"], "jobs": jobs}
    running, blocked = {}, set()
    gpu_ids = {g["index"]: g["uuid"] for g in allocation["gpus"]}
    # Separate dataset pools; same algorithm settings and per-process resources.
    pools = {records[0]["dataset_id"]: args.gpus[:len(args.gpus)//2],
             records[1]["dataset_id"]: args.gpus[len(args.gpus)//2:]}
    if any(not p for p in pools.values()):
        raise ValueError("need at least two GPUs")
    def stop_signal(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        while True:
            now = time.time()
            if STOP or now - state["started"] > args.hours * 3600:
                state["stop_reason"] = "signal" if STOP else "wall-clock budget"
                break
            for key, (proc, handle, job) in list(running.items()):
                if now - job["started"] > args.run_hours * 3600:
                    terminate(proc)
                    job["error"] = "per-run timeout"
                code = proc.poll()
                if code is not None:
                    handle.close()
                    if code == 0:
                        from .collect_research import parse_log
                        record = next(r for r in records if r["dataset_id"] == job["dataset"])
                        model_log = Path(record["model_root"]) / job["dataset"] / (job["expid"] + ".log")
                        if parse_log(model_log, record["tasks"])["status"] != "complete":
                            code = 1
                            job["error"] = "exit zero without complete validation record"
                    job.update(status="completed" if code == 0 else "failed", returncode=code,
                               ended=now, seconds=now-job["started"])
                    if code != 0:
                        blocked.add(job["dataset"])
                    del running[key]
                    print(json.dumps({"event": "exit", **job}), flush=True)
            pilots_done = {j["dataset"] for j in jobs if j["pilot"] and j["status"] == "completed"}
            occupancy = Counter(j["gpu"] for _, _, j in running.values())
            for job in jobs:
                if job["status"] != "pending" or job["dataset"] in blocked:
                    continue
                if not job["pilot"] and job["dataset"] not in pilots_done:
                    continue
                candidates = [g for g in pools[job["dataset"]] if occupancy[g] < args.slots]
                if not candidates:
                    continue
                gpu = min(candidates, key=lambda g: (occupancy[g], g))
                seed = int(job["expid"].rsplit("_S", 1)[1])
                directory = root / "stdout" / job["dataset"]
                directory.mkdir(parents=True, exist_ok=True)
                logfile = directory / (job["expid"] + ".log")
                handle = logfile.open("x")
                proc = subprocess.Popen([sys.executable, "-m", "explorations.relational_supervision.run_research",
                    "--config", job["config"], "--expid", job["expid"], "--gpu", "0"],
                    cwd=ROOT, env=env_for(root, gpu_ids[gpu], seed), stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(status="running", gpu=gpu, uuid=gpu_ids[gpu], pid=proc.pid,
                           started=time.time(), stdout=str(logfile))
                running[(job["dataset"],job["expid"])] = proc, handle, job
                occupancy[gpu] += 1
                print(json.dumps({"event": "start", **job}), flush=True)
            for job in jobs:
                if job["status"] == "pending" and job["dataset"] in blocked:
                    job["status"] = "blocked_after_failure"
            state["updated"] = time.time()
            write_json(root / "queue.json", state)
            if not running and not any(j["status"] == "pending" for j in jobs):
                break
            time.sleep(5)
        for proc, handle, job in running.values():
            terminate(proc)
            handle.close()
            job.update(status="stopped", ended=time.time())
        state["finished"] = time.time()
        write_json(root / "queue.json", state)
        # No test parsing: generate only the protocol's validation report.
        for config in args.configs:
            record = json.loads((config / "protocol.json").read_text())
            if all(j["status"] == "completed" for j in jobs if j["dataset"] == record["dataset_id"]):
                subprocess.run([sys.executable, "-m", "explorations.relational_supervision.collect_research",
                    "--stage", "phase1", "--config", str(config.resolve()), "--output",
                    str(root / "validation" / record["dataset_id"])], cwd=ROOT, check=True)
    finally:
        for proc, handle, job in running.values():
            terminate(proc)
            handle.close()
        restore(root)


if __name__ == "__main__":
    main()
