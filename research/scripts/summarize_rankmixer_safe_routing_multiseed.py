#!/usr/bin/env python3
"""Aggregate paired KuaiRand safe-routing confirmations across three seeds."""
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20262028, 20262029, 20262030)
TASKS = ("is_click", "is_follow", "is_like", "is_comment", "is_forward", "long_view")
METRIC = re.compile(r"\[Task: ([^]]+)\]\[Metrics\].*?AUC: ([0-9.]+)")


def parse(path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    if "******** Test evaluation ********" not in text:
        return None
    section = text.rsplit("******** Test evaluation ********", 1)[1]
    values = {task: float(auc) for task, auc in METRIC.findall(section)}
    return values if set(values) == set(TASKS) else None


def candidate_path(name, seed):
    if seed == 20262028:
        return ROOT / (
            "runs_local/checkpoints_rankmixer_safe_routing/KuaiRand_Video_Action/"
            f"RankMixer_Kuai_{name}_S{seed}.log"
        )
    return ROOT / (
        "runs_local/checkpoints_rankmixer_safe_routing_multiseed/KuaiRand_Video_Action/"
        f"RankMixer_Kuai_{name}_S{seed}.log"
    )


def control_path(seed):
    return ROOT / (
        "runs_local/rankmixer_multiseed_confirmation/"
        f"RankMixer_Kuai_DAGMeanPre2_S{seed}.stdout.log"
    )


def stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def main():
    runs = {
        "Control": {str(seed): parse(control_path(seed)) for seed in SEEDS},
        "AdapterOnly": {
            str(seed): parse(candidate_path("AdapterOnly", seed)) for seed in SEEDS
        },
        "AdapterResidualBypass": {
            str(seed): parse(candidate_path("AdapterResidualBypass", seed))
            for seed in SEEDS
        },
    }
    result = {"runs": runs, "aggregate": {}}
    for name, by_seed in runs.items():
        complete = {seed: row for seed, row in by_seed.items() if row}
        aggregate = {"completed_seeds": sorted(complete)}
        if complete:
            aggregate["tasks"] = {
                task: stats([row[task] for row in complete.values()]) for task in TASKS
            }
            macro = [statistics.mean(row.values()) for row in complete.values()]
            aggregate["macro_auc"] = stats(macro)
        result["aggregate"][name] = aggregate

    controls = runs["Control"]
    result["paired_deltas_vs_control"] = {}
    for name in ("AdapterOnly", "AdapterResidualBypass"):
        deltas = {}
        for seed in map(str, SEEDS):
            if runs[name][seed] and controls[seed]:
                task_delta = {
                    task: runs[name][seed][task] - controls[seed][task] for task in TASKS
                }
                deltas[seed] = {
                    "tasks": task_delta,
                    "macro_auc": statistics.mean(task_delta.values()),
                }
        paired = {"by_seed": deltas}
        if len(deltas) >= 2:
            paired["tasks"] = {
                task: stats([row["tasks"][task] for row in deltas.values()])
                for task in TASKS
            }
            paired["macro_auc"] = stats([
                row["macro_auc"] for row in deltas.values()
            ])
            paired["positive_macro_seeds"] = sum(
                row["macro_auc"] > 0 for row in deltas.values()
            )
        result["paired_deltas_vs_control"][name] = paired
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
