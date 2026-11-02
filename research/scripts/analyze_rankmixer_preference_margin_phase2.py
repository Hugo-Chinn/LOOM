#!/usr/bin/env python3
"""Analyze low-weight margin sweep against the frozen phase-1 baseline."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/rankmixer_preference_margin_phase2"
OUTPUT = ROOT / "reports/rankmixer_preference_margin_phase2.json"
BASELINE = ROOT / "reports/rankmixer_preference_margin_four_seed.json"
SEEDS = (20262027, 20262028, 20262029, 20262030)
VARIANTS = (
    "ProgressiveW0005", "ProgressiveW001", "ProgressiveW002",
    "ProgressiveW005", "ZeroMarginBPRW002",
)
TASKS = ("is_click", "cart", "fav", "buy")


def parse(path):
    text = path.read_text()
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


def main():
    baseline = json.loads(BASELINE.read_text())["runs"]["Baseline"]
    runs = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in SEEDS:
            path = RUN_DIR / f"RankMixer_Branch_{variant}_Seed{seed}.stdout.log"
            runs[variant][str(seed)] = parse(path)
    aggregate = {}
    paired = {}
    for variant in VARIANTS:
        aggregate[variant] = {}
        paired[variant] = {}
        for task in TASKS:
            aggregate[variant][task] = {
                metric: stats([runs[variant][str(seed)][task][metric] for seed in SEEDS])
                for metric in ("AUC", "logloss")
            }
            deltas = {
                str(seed): runs[variant][str(seed)][task]["AUC"]
                - baseline[str(seed)][task]["AUC"] for seed in SEEDS
            }
            paired[variant][task] = {
                "by_seed": deltas, **stats(list(deltas.values())),
                "positive_seeds": sum(value > 0 for value in deltas.values()),
            }
    result = {
        "date": "2026-10-10", "dataset": "Taobao_Action",
        "seeds": list(SEEDS), "runs": runs, "aggregate": aggregate,
        "paired_auc_vs_phase1_baseline": paired,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
