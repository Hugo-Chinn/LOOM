#!/usr/bin/env python3
"""Summarize the 4-seed RankMixer preference-margin experiment."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20262027, 20262028, 20262029, 20262030)
VARIANTS = ("Baseline", "FixedMargin", "ProgressiveMargin", "ReverseProgressiveMargin")
TASKS = ("is_click", "cart", "fav", "buy")


def parse_log(path):
    text = path.read_text()
    if "******** Test evaluation ********" not in text:
        raise ValueError(f"Incomplete run: {path}")
    test = text.rsplit("******** Test evaluation ********", 1)[1]
    metrics = {}
    for task, payload in re.findall(r"\[Task: ([^]]+)\]\[Metrics\] (.+)", test):
        metrics[task] = {
            key: float(value)
            for key, value in re.findall(r"(AUC|logloss): ([0-9.]+)", payload)
        }
    missing = set(TASKS) - set(metrics)
    if missing:
        raise ValueError(f"Missing tasks={sorted(missing)} in {path}")
    return metrics


def mean_std(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "runs_local/checkpoints_rankmixer_preference_margin/Taobao_Action",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/rankmixer_preference_margin_four_seed.json",
    )
    args = parser.parse_args()

    runs = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in SEEDS:
            path = args.log_dir / f"RankMixer_Branch_{variant}_Seed{seed}.log"
            runs[variant][str(seed)] = parse_log(path)

    aggregate = {}
    paired = {}
    for variant in VARIANTS:
        aggregate[variant] = {
            task: {
                metric: mean_std([
                    runs[variant][str(seed)][task][metric] for seed in SEEDS
                ])
                for metric in ("AUC", "logloss")
            }
            for task in TASKS
        }
        if variant == "Baseline":
            continue
        paired[variant] = {}
        for task in TASKS:
            values = {
                str(seed): runs[variant][str(seed)][task]["AUC"]
                - runs["Baseline"][str(seed)][task]["AUC"]
                for seed in SEEDS
            }
            paired[variant][task] = {
                "by_seed": values,
                **mean_std(list(values.values())),
                "positive_seeds": sum(value > 0 for value in values.values()),
            }

    progressive_vs_fixed = {
        task: {
            str(seed): runs["ProgressiveMargin"][str(seed)][task]["AUC"]
            - runs["FixedMargin"][str(seed)][task]["AUC"]
            for seed in SEEDS
        }
        for task in TASKS
    }
    progressive_vs_reverse = {
        task: {
            str(seed): runs["ProgressiveMargin"][str(seed)][task]["AUC"]
            - runs["ReverseProgressiveMargin"][str(seed)][task]["AUC"]
            for seed in SEEDS
        }
        for task in TASKS
    }
    result = {
        "date": "2026-10-09",
        "dataset": "Taobao_Action",
        "seeds": list(SEEDS),
        "protocol": {
            "backbone": "RankMixer branch Loop + deep progressive BCE",
            "batching": "user-group-aware for every variant",
            "positive": "buy",
            "negative": "click-only, no cart/fav/buy",
            "pairing": "same user only",
            "depths": ["click", "mean(cart,fav)", "buy"],
            "margin_weight": 0.1,
            "margin_target": 0.2,
        },
        "runs": runs,
        "aggregate": aggregate,
        "paired_auc_vs_baseline": paired,
        "progressive_minus_fixed_auc": progressive_vs_fixed,
        "progressive_minus_reverse_auc": progressive_vs_reverse,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
