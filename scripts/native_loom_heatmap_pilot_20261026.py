"""Prepare isolated Native LOOM runs and clearly labeled synthetic plot data."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs_local/native_loom_heatmap_pilot_20261026"
CONFIG = OUT / "config"
SEEDS = (20262027, 20262028, 20262029, 20262030)
GROUPS = (
    ("Taobao", "structure_ablation_dag", "DAGTrue", ("Click", "Cart", "Fav", "Buy")),
    ("MerRec", "structure_merrec", "Struct_MerRec_True",
     ("Like", "Cart", "Offer", "Checkout", "Purchase")),
    ("KuaiRand", "structure_kuai", "Struct_Kuai_True",
     ("Click", "Follow", "Like", "Comment", "Forward", "Long View")),
    ("QK", "structure_qk", "Struct_QK_True", ("Click", "Follow", "Like", "Share")),
)
# Arbitrary layout placeholders, not fitted estimates or experimental evidence.
MOCK = {
    "Taobao": (
        (-.0004, -.0008, -.0006, -.0020),
        (-.0007, -.0012, -.0009, -.0010),
        (0., 0., 0., -.0025),
        (.0001, -.0004, .0002, -.0008),
    ),
    "MerRec": (
        (-.0010, -.0013, -.0010, -.0015, -.0020),
        (-.0010, -.0015, -.0012, -.0010, -.0015),
        (0., 0., 0., 0., -.0025),
        (.0002, -.0005, -.0003, .0001, -.0008),
    ),
    "KuaiRand": (
        (-.0007, -.0020, -.0008, -.0025, -.0020, -.0006),
        (-.0005, -.0025, -.0007, -.0020, -.0022, -.0005),
        (0., -.0015, 0., -.0020, -.0018, 0.),
        (.0001, -.0008, .0002, -.0010, -.0007, -.0002),
    ),
    "QK": (
        (-.0001, -.0002, -.0001, -.0002),
        (-.0001, -.0002, -.0001, -.0001),
        (None, None, None, None),
        (.0001, -.0001, .0001, -.0001),
    ),
}


def prepare():
    CONFIG.mkdir(parents=True, exist_ok=True)
    (OUT / "console").mkdir(exist_ok=True)
    models = {"Base": {}}
    datasets = {}
    jobs = []
    for dataset, suffix, control, _ in GROUPS:
        source = ROOT / ("config_rankmixer_" + suffix)
        cfg = yaml.safe_load((source / "model_config.yaml").read_text())
        datasets.update(yaml.safe_load((source / "dataset_config.yaml").read_text()))
        for seed in SEEDS:
            full = {**cfg.get("Base", {}), **cfg[f"{control}_S{seed}"]}
            full["model_root"] = str(OUT / "checkpoints")
            variants = ["Full", "NoBCU", "NoMAS", "Untied"]
            if dataset != "QK":
                variants.append("NoCBM")
            for variant in variants:
                params = copy.deepcopy(full)
                if variant == "NoBCU":
                    params["use_behavior_embedding"] = False
                if variant == "NoMAS":
                    params["supervision_mode"] = "progressive"
                    params["auxiliary_loss_weight"] = 0.0
                if variant == "Untied":
                    params["share_loop_parameters"] = False
                if variant == "NoCBM":
                    params["disable_branch_merge"] = True
                expid = f"Native_{dataset}_{variant}_S{seed}"
                models[expid] = params
                jobs.append({"expid": expid, "dataset": dataset,
                             "variant": variant, "seed": seed})
    (CONFIG / "model_config.yaml").write_text(yaml.safe_dump(models, sort_keys=False))
    (CONFIG / "dataset_config.yaml").write_text(yaml.safe_dump(datasets, sort_keys=False))
    (OUT / "jobs.json").write_text(json.dumps(jobs, indent=2))
    with (OUT / "SYNTHETIC_ONLY_NOT_EXPERIMENTAL_delta_auc.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "task", "variant", "synthetic_delta_auc", "status"))
        for dataset, _, _, tasks in GROUPS:
            for variant, values in zip(
                    ("w/o BCU", "w/o MAS", "w/o CBM", "Untied SRU"), MOCK[dataset]):
                for task, value in zip(tasks, values):
                    writer.writerow((dataset, task, variant, value,
                                     "NOT_APPLICABLE" if value is None else "SYNTHETIC_NOT_MEASURED"))
    (OUT / "README.txt").write_text(
        "Synthetic CSV is ONLY for plotting layout. Values are arbitrary, NOT "
        "measured, NOT seed-level AUC, and NOT suitable for scientific claims.\n"
        "Runs: original width-2 DAG structure controls; NOT latest width-4 recipes.\n"
        "76 jobs: four seeds of Full/NoBCU/NoMAS/Untied plus NoCBM except QK.\n"
        "NoMAS retains per-task progressive main losses and disables deep auxiliaries.\n"
        "NoCBM keeps all branch updates, selecting first declared parent at merges.\n"
        "Untied uses independent blocks per behavior and per preloop; QK merge is N/A.\n"
        "Raw per-task Test metrics will be under checkpoints/<dataset>/<expid>.log.\n"
    )
    return jobs


def run():
    jobs = json.loads((OUT / "jobs.json").read_text())
    active = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode().split("\0")
            if "--train" not in args or str(CONFIG) not in args:
                continue
            expid = args[args.index("--expid") + 1]
            gpu = int(args[args.index("--gpu") + 1])
            active[gpu] = (expid, int(proc.name))
        except (OSError, ValueError, IndexError):
            continue
    active_ids = {expid for expid, _ in active.values()}
    pending = []
    for job in jobs:
        logs = list((OUT / "checkpoints").rglob(job["expid"] + ".log"))
        if logs and "******** Test evaluation ********" in logs[0].read_text():
            continue
        if job["expid"] in active_ids:
            continue
        pending.append(job)
    queue = Queue()
    for job in pending:
        queue.put(job)
    print(f"Pending {len(pending)} queued; adopting {len(active)} active; "
          "two slots per GPU", flush=True)

    def worker(slot):
        gpu = slot % 8
        if slot < 8 and gpu in active:
            expid, pid = active[gpu]
            print(f"ADOPT gpu={gpu} expid={expid} pid={pid}", flush=True)
            while True:
                try:
                    stat = Path(f"/proc/{pid}/stat").read_text()
                    if stat.rsplit(")", 1)[1].split()[0] == "Z":
                        break
                except FileNotFoundError:
                    break
                time.sleep(5)
            print(f"ADOPT_END gpu={gpu} expid={expid}", flush=True)
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                return
            expid = job["expid"]
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--train",
                       "--config", str(CONFIG),
                       "--expid", expid, "--gpu", str(gpu)]
            print(f"START gpu={gpu} expid={expid}", flush=True)
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       CUDA_VISIBLE_DEVICES=str(gpu),
                       TORCHINDUCTOR_CACHE_DIR=str(OUT / "compile_cache" / expid))
            previous = OUT / "console" / f"{expid}.txt"
            if previous.exists():
                stamp = str(time.time_ns())
                previous.rename(previous.with_suffix(f".failed_{stamp}.txt"))
                for log in (OUT / "checkpoints").rglob(expid + ".log"):
                    log.rename(log.with_suffix(f".failed_{stamp}.log"))
            with previous.open("w") as handle:
                code = subprocess.call(command, cwd=ROOT, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            print(f"END gpu={gpu} expid={expid} returncode={code}", flush=True)
            if code:
                print(f"STOP gpu={gpu}: inspect failed job before continuing", flush=True)
                break

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(worker, range(16)))


if __name__ == "__main__":
    if "--train" in sys.argv:
        import runpy
        sys.path.insert(0, str(ROOT))
        import unirank.preprocess

        def existing_dataset(feature_encoder, train_data=None, valid_data=None,
                             test_data=None, **kwargs):
            feature_map = Path(feature_encoder.data_dir) / "feature_map.json"
            with feature_map.open() as handle:
                json.load(handle)
            return train_data, valid_data, test_data

        unirank.preprocess.build_dataset = existing_dataset
        sys.argv = ["run_expid.py"] + [arg for arg in sys.argv[1:] if arg != "--train"]
        runpy.run_path(str(ROOT / "run_expid.py"), run_name="__main__")
        sys.exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
    else:
        jobs = prepare()
        print(f"Prepared {len(jobs)} jobs at {OUT}")
        if args.launch:
            with (OUT / "scheduler.log").open("a") as handle:
                process = subprocess.Popen(
                    [sys.executable, "-B", str(Path(__file__).resolve()), "--run"],
                    cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            (OUT / "scheduler_pid.json").write_text(json.dumps({"pid": process.pid}))
            print(f"Launched scheduler PID={process.pid}")
