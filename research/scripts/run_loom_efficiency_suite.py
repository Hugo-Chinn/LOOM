#!/usr/bin/env python3
"""One microbenchmark per GPU, with archived model code and raw results."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from statistics import mean, stdev

import yaml

from summarize_all_local_experiments import parse_log

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/loom_efficiency_20261025"
RUNTIME = ROOT / "runs_local/source_snapshots/loop_strategy_20261021"
DATASETS = ["TencentGR_10M_Action", "QK_Video_Action", "KuaiRand_Video_Action",
            "MerRec_Action", "Taobao_Action"]


def jobs():
    result = []
    for backbone in ("rankmixer", "onetrans"):
        config = f"config_{backbone}_loop_strategy_20261021"
        for ds in DATASETS:
            for variant in ("Parallel", "LoopControl", "Wide", "WideWeight", "WideBypass"):
                result.append({
                    "backbone": backbone, "name": f"Native-{variant}", "study": "native_loop_v2",
                    "runtime": str(RUNTIME), "config": config,
                    "expid": f"StrategyV2_{backbone}_{variant}_{ds}_S20262027",
                })
        config = f"config_{backbone}_mtl_aligned"
        for ds in DATASETS:
            for variant in ("ParallelIndependentTower", "MMoE", "PLE", "STEM", "LoopMT", "UntiedDeep"):
                result.append({
                    "backbone": backbone, "name": f"Head-{variant}", "study": "fixed_backbone_head",
                    "runtime": str(RUNTIME), "config": config,
                    "expid": f"Head_{variant}_{ds}_S20262027",
                })
    return result


def historical_timing(job):
    config = ROOT / job["config"]
    manifest = json.loads((config / "manifest.json").read_text())
    snapshot = hashlib.sha256((config / "manifest.json").read_bytes()).hexdigest()
    logdir = ROOT / "logs/rankmixer_mtl_aligned" / snapshot[:12]
    expid = job["expid"]
    records = []
    for seed in (20262027, 20262028, 20262029, 20262030):
        path = logdir / expid.replace("S20262027", f"S{seed}")
        path = path.with_name(path.name + ".log")
        parsed = parse_log(path) if path.exists() else None
        if parsed:
            records.append({"seed": seed, "wall_seconds": parsed["wall_seconds"],
                            "log": str(path.relative_to(ROOT))})
    values = [r["wall_seconds"] for r in records if r["wall_seconds"] is not None]
    return {
        "runs": records, "n": len(values),
        "wall_minutes_mean": mean(values) / 60 if values else None,
        "wall_minutes_std": stdev(values) / 60 if len(values) > 1 else None,
        "scope": "full log wall time: includes startup, training, validation/test, concurrent-resource effects",
    }


def main():
    OUT.mkdir(exist_ok=True)
    all_jobs = jobs()
    (OUT / "jobs.json").write_text(json.dumps(all_jobs, indent=2) + "\n")
    # No two measurements share a GPU; CPU/data movement remain excluded from timing.
    slots = [[] for _ in range(8)]
    for index, job in enumerate(all_jobs):
        slots[index % 8].append((index, job))

    def worker(gpu):
        for index, job in slots[gpu]:
            output = OUT / f"{index:03d}.json"
            if output.exists():
                continue
            log = OUT / f"{index:03d}.log"
            while True:
                usage = subprocess.check_output([
                    "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"], text=True).strip().split(",")
                if int(usage[0]) < 64 and int(usage[1]) == 0:
                    break
                print(f"WAIT GPU={gpu} occupied; next={index}", flush=True)
                time.sleep(60)
            with log.open("w") as handle:
                proc = subprocess.run([
                    sys.executable, str(ROOT / "scripts/profile_loom_efficiency.py"),
                    "--job", json.dumps(job), "--out", str(output),
                ], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                    stdout=handle, stderr=subprocess.STDOUT)
            print(f"{'DONE' if proc.returncode == 0 else 'FAIL'} {index} {job['expid']}", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))
    results, failures = [], []
    for index, job in enumerate(all_jobs):
        path = OUT / f"{index:03d}.json"
        if path.exists():
            row = json.loads(path.read_text())
            row["historical_training_wall"] = historical_timing(job)
            results.append(row)
        else:
            failures.append(job)
    (OUT / "results.json").write_text(json.dumps({
        "results": results, "failures": failures,
        "protocol": "one job per GPU; eight GPUs run in parallel; shared host CPU can affect eager launch timings",
        "head_source_note": "Head methods reprofiled on V2 archived source; not asserted byte-identical to old aligned runtime.",
    }, indent=2) + "\n")
    lines = [
        "# LOOM Efficiency Comparison", "",
        "Hardware: H20. BF16 autocast / FP32 weights. Synthetic full-length 100-event history.",
        "Real dataset vocabulary sizes; randomly initialized model weights; no accuracy evaluation.",
        "Eager execution: Torch compile disabled for all methods. CPU inputs/data loader excluded.",
        "One benchmark per GPU, eight GPUs parallel; shared CPU can affect eager launch timing.",
        "FLOPs: registered PyTorch operators, MAC=2 FLOPs; lookup/activation/normalization may be omitted.",
        "Training step includes forward, FP32 loss, backward and AdamW/Adagrad update, batch=256.",
        "Historical wall time includes entire experiment and resource sharing; not isolated training speed.",
        "Head variants are reprofiled on V2 archived code, not an assertion of identical old source.",
        "SharedBottom is a parallel alias. OneRank/MoME/ResFlow lack verified faithful results and are excluded.",
        "",
    ]
    flat = []
    for backbone in ("rankmixer", "onetrans"):
        for ds in DATASETS:
            lines += [
                f"## {backbone} / {ds}", "",
                "| Method | Total M | Emb M | Dense M | Forward GFLOPs/sample | F+B GFLOPs/sample | Infer B1 ms | Infer B256 ms | Infer samples/s B256 | Train B256 ms | Train samples/s | Peak train MiB | Historical wall min |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in results:
                if r["backbone"] != backbone or r["dataset"] != ds:
                    continue
                a, b, t = r["inference"]["1"], r["inference"]["256"], r["training_256"]
                wall = r["historical_training_wall"]["wall_minutes_mean"]
                lines.append(
                    f"| {r['name']} | {r['counts']['total']/1e6:.3f} | "
                    f"{r['counts']['embedding']/1e6:.3f} | {r['counts']['dense']/1e6:.3f} | "
                    f"{r['forward_gflops_per_sample']:.6f} | "
                    f"{r['forward_backward_gflops_per_sample']:.6f} | "
                    f"{a['median_ms']:.3f} | {b['median_ms']:.3f} | {b['samples_per_second']:.0f} | "
                    f"{t['median_ms']:.3f} | {t['samples_per_second']:.0f} | "
                    f"{t['peak_allocated_mib']:.1f} | {wall:.1f} |"
                    if wall is not None else f"| {r['name']} | Missing historical timing |"
                )
                flat.append({
                    "backbone": backbone, "dataset": ds, "method": r["name"],
                    "study": r["study"], **r["counts"],
                    "forward_gflops_per_sample": r["forward_gflops_per_sample"],
                    "forward_backward_gflops_per_sample": r["forward_backward_gflops_per_sample"],
                    "inference_b1_median_ms": a["median_ms"],
                    "inference_b1_p95_ms": a["p95_ms"],
                    "inference_b256_median_ms": b["median_ms"],
                    "inference_b256_samples_per_second": b["samples_per_second"],
                    "train_b256_median_ms": t["median_ms"],
                    "train_b256_samples_per_second": t["samples_per_second"],
                    "train_peak_allocated_mib": t["peak_allocated_mib"],
                    "historical_wall_minutes_mean": wall,
                    "historical_wall_minutes_std": r["historical_training_wall"]["wall_minutes_std"],
                })
            lines.append("")
    lines += ["## Failures", "", json.dumps(failures, indent=2), "",
              "Raw files contain operator breakdown, parameter module groups and source hashes."]
    (OUT / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    if flat:
        with (OUT / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    print(f"Finished {len(results)}/{len(all_jobs)} benchmarks; failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
