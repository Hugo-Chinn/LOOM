#!/usr/bin/env python3
"""Actual parameter counts and experiment wall times from completed run logs."""
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, stdev

from summarize_all_local_experiments import parse_log

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "total_parameters": re.compile(r"Total number of parameters:\s*(\d+)"),
    "embedding_parameters": re.compile(r"Number of embedding parameters:\s*(\d+)"),
    "dense_parameters": re.compile(r"Number of dense parameters:\s*(\d+)"),
}


def main():
    records = []
    for backbone in ("rankmixer", "onetrans"):
        for version in ("mtl_aligned", "loop_strategy_20261021"):
            config = ROOT / f"config_{backbone}_{version}"
            manifest = json.loads((config / "manifest.json").read_text())
            snapshot = hashlib.sha256((config / "manifest.json").read_bytes()).hexdigest()
            logdir = ROOT / "logs/rankmixer_mtl_aligned" / snapshot[:12]
            groups = {}
            for expid, arm in manifest["arms"].items():
                if arm["status"] != "ready":
                    continue
                path = logdir / f"{expid}.log"
                parsed = parse_log(path) if path.exists() else None
                if not parsed:
                    continue
                text = path.read_text(errors="ignore")
                counts = {name: int(match.group(1)) if (match := regex.search(text)) else None
                          for name, regex in PATTERNS.items()}
                groups.setdefault((arm["study"], arm["dataset"], arm["variant"]), []).append(
                    {**counts, "seed": arm["seed"], "seconds": parsed["wall_seconds"]})
            for (study, ds, variant), runs in sorted(groups.items()):
                times = [r["seconds"]/60 for r in runs if r["seconds"] is not None]
                counts = {}
                for name in PATTERNS:
                    values = {r[name] for r in runs}
                    if len(values) != 1:
                        raise ValueError(f"Parameter count drift: {backbone}/{ds}/{variant}/{name}")
                    counts[name] = next(iter(values))
                records.append({
                    "backbone": backbone, "version": version, "study": study,
                    "dataset": ds, "variant": variant, "n": len(runs),
                    "seeds": ",".join(str(r["seed"]) for r in sorted(runs, key=lambda r:r["seed"])),
                    **counts, "wall_minutes_mean": mean(times),
                    "wall_minutes_std": stdev(times) if len(times)>1 else None,
                    "wall_minutes_min": min(times), "wall_minutes_max": max(times),
                    "snapshot": snapshot[:12],
                })
    out = ROOT / "reports/loom_efficiency_20261025"
    out.mkdir(exist_ok=True)
    with (out / "logged_parameters_walltime.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (out / "logged_parameters_walltime.json").write_text(json.dumps(records, indent=2)+"\n")
    lines = [
        "# Logged Parameters and Experiment Wall Time", "",
        "Actual completed experiment logs; parameter counts must agree across seeds.",
        "Wall time is NOT isolated training time: includes startup/compile, validation/test",
        "and resource contention from co-located runs. Do not infer architecture speed from it alone.",
        "Different study/source versions are separated; no cross-version result pooling.",
        "", "| Backbone | Version | Study | Dataset | Method | N | Total M | Emb M | Dense M | Wall min mean +/- SD |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        fmt = lambda name: f"{r[name]/1e6:.3f}" if r[name] is not None else "NA"
        sd = f"{r['wall_minutes_std']:.2f}" if r["wall_minutes_std"] is not None else "NA"
        lines.append(
            f"| {r['backbone']} | {r['version']} | {r['study']} | {r['dataset']} | "
            f"{r['variant']} | {r['n']} | {fmt('total_parameters')} | "
            f"{fmt('embedding_parameters')} | {fmt('dense_parameters')} | "
            f"{r['wall_minutes_mean']:.2f} +/- {sd} |"
        )
    (out / "LOGGED_PARAMETERS_WALLTIME.md").write_text("\n".join(lines)+"\n")
    print(f"Exported {len(records)} groups")


if __name__ == "__main__":
    main()
