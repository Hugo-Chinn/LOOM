#!/usr/bin/env python3
"""Analyze phase-3 margin calibration with reused phase-1/2 controls."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/rankmixer_preference_margin_phase3"
PHASE1 = ROOT / "reports/rankmixer_preference_margin_four_seed.json"
PHASE2 = ROOT / "reports/rankmixer_preference_margin_phase2.json"
OUTPUT = ROOT / "reports/rankmixer_preference_margin_phase3.json"
SEEDS = (20262027, 20262028, 20262029, 20262030)
VARIANTS = ("W001M000", "W001M005", "W001M010", "W00075M020")
TASKS = ("is_click", "cart", "fav", "buy")


def parse(path):
    text = path.read_text()
    if "******** Test evaluation ********" not in text:
        raise ValueError(f"Incomplete run: {path}")
    test = text.rsplit("******** Test evaluation ********", 1)[1]
    out = {}
    for task, payload in re.findall(r"\[Task: ([^]]+)\]\[Metrics\] (.+)", test):
        out[task] = {key: float(value) for key, value in
                     re.findall(r"(AUC|logloss): ([0-9.]+)", payload)}
    if set(out) != set(TASKS):
        raise ValueError(f"Incomplete metrics in {path}")
    return out


def stats(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def paired(runs, control):
    result = {}
    for variant, by_seed in runs.items():
        result[variant] = {}
        for task in TASKS:
            values = {
                str(seed): by_seed[str(seed)][task]["AUC"]
                - control[str(seed)][task]["AUC"] for seed in SEEDS
            }
            result[variant][task] = {
                "by_seed": values, **stats(list(values.values())),
                "positive_seeds": sum(value > 0 for value in values.values()),
            }
    return result


def main():
    phase1 = json.loads(PHASE1.read_text())
    phase2 = json.loads(PHASE2.read_text())
    baseline = phase1["runs"]["Baseline"]
    best_phase2 = phase2["runs"]["ProgressiveW001"]
    runs = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in SEEDS:
            path = RUN_DIR / f"RankMixer_Branch_{variant}_Seed{seed}.stdout.log"
            runs[variant][str(seed)] = parse(path)

    aggregate = {
        variant: {
            task: {
                metric: stats([runs[variant][str(seed)][task][metric]
                               for seed in SEEDS])
                for metric in ("AUC", "logloss")
            }
            for task in TASKS
        }
        for variant in VARIANTS
    }
    result = {
        "date": "2026-10-10",
        "dataset": "Taobao_Action",
        "seeds": list(SEEDS),
        "controls": {
            "baseline": "phase1 Baseline",
            "W001M020": "phase2 ProgressiveW001",
        },
        "runs": runs,
        "aggregate": aggregate,
        "paired_auc_vs_baseline": paired(runs, baseline),
        "paired_auc_vs_W001M020": paired(runs, best_phase2),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
