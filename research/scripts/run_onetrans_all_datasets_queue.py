#!/usr/bin/env python3
"""Recoverable multi-GPU queue for all-dataset OneTrans experiments."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config_onetrans_all_datasets_multiseed"
RUN_DIR = ROOT / "runs_local/onetrans_all_datasets_multiseed"
QUEUE_FILE = RUN_DIR / "queue.json"
LOCK_FILE = RUN_DIR / "queue.lock"

DATASETS = (
    "Taobao_Action",
    "QK_Video_Action",
    "TencentGR_10M_Action",
    "MerRec_Action",
    "KuaiRand_Video_Action",
)
SEEDS = (20262027, 20262028, 20262029)
VARIANTS = ("Baseline", "FiLM", "QVLoRA")


def task_ids():
    # Keep paired comparisons close in wall-clock time: all three variants for
    # one dataset/seed are adjacent. Smaller datasets go first for early signal.
    return [
        f"OneTransAll_{dataset}_{variant}_S{seed}"
        for dataset in DATASETS
        for seed in SEEDS
        for variant in VARIANTS
    ]


def initialize_queue(reset=False):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if QUEUE_FILE.exists() and not reset:
        return
    state = {
        "pending": task_ids(),
        "running": {},
        "completed": [],
        "failed": [],
    }
    QUEUE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def with_locked_state(callback):
    LOCK_FILE.touch(exist_ok=True)
    with LOCK_FILE.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(QUEUE_FILE.read_text())
        result = callback(state)
        QUEUE_FILE.write_text(json.dumps(state, indent=2) + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)
        return result


def claim_task(gpu):
    def claim(state):
        # Recover stale same-GPU entries left by a killed worker.
        stale = [task for task, owner in state["running"].items()
                 if str(owner) == str(gpu)]
        for task in stale:
            state["running"].pop(task, None)
            if task not in state["pending"]:
                state["pending"].insert(0, task)
        while state["pending"]:
            task = state["pending"].pop(0)
            if (RUN_DIR / f"{task}.done").exists():
                if task not in state["completed"]:
                    state["completed"].append(task)
                continue
            state["running"][task] = str(gpu)
            return task
        return None
    return with_locked_state(claim)


def finish_task(task, success):
    def finish(state):
        state["running"].pop(task, None)
        target = "completed" if success else "failed"
        if task not in state[target]:
            state[target].append(task)
    with_locked_state(finish)


def gpu_memory_mib(gpu):
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=used_memory",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True,
    )
    if result.returncode != 0:
        return None
    values = []
    for line in result.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            pass
    return sum(values)


def wait_until_free(gpu, threshold_mib, poll_seconds):
    while True:
        memory = gpu_memory_mib(gpu)
        if memory is not None and memory <= threshold_mib:
            return
        time.sleep(poll_seconds)


def worker(gpu, threshold_mib, poll_seconds):
    worker_log = RUN_DIR / f"worker_gpu{gpu}.log"
    with worker_log.open("a", buffering=1) as status:
        status.write(f"worker start gpu={gpu}\n")
        while True:
            wait_until_free(gpu, threshold_mib, poll_seconds)
            task = claim_task(gpu)
            if task is None:
                status.write("queue empty; worker exit\n")
                return
            log_path = RUN_DIR / f"{task}.stdout.log"
            status.write(f"start {task}\n")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            with log_path.open("w") as output:
                result = subprocess.run(
                    [str(ROOT / ".venv-unirank/bin/python"), "run_expid.py",
                     "--config", str(CONFIG), "--expid", task, "--gpu", "0"],
                    cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                )
            success = result.returncode == 0
            marker = RUN_DIR / f"{task}.{'done' if success else 'failed'}"
            marker.write_text(f"returncode={result.returncode}\n")
            finish_task(task, success)
            status.write(f"finish {task} success={success}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--free_threshold_mib", type=int, default=1024)
    parser.add_argument("--poll_seconds", type=int, default=30)
    args = parser.parse_args()
    initialize_queue(reset=args.reset)
    if args.init:
        print(QUEUE_FILE)
        return
    if args.gpu is None:
        raise SystemExit("--gpu is required unless --init is used")
    worker(args.gpu, args.free_threshold_mib, args.poll_seconds)


if __name__ == "__main__":
    main()
