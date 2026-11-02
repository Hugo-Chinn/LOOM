"""Export measured per-seed/task metrics and paired ablation statistics."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def summarize(directory):
    runs = {}
    raw = []
    for path in sorted(directory.glob("*/S*/metrics.json")):
        variant = path.parent.parent.name
        result = json.loads(path.read_text())
        seed = result["seed"]
        if (variant, seed) in runs:
            raise ValueError("duplicate variant/seed result")
        runs[variant, seed] = result["test"]
        for task, metric in result["test"].items():
            raw.append({"variant": variant, "seed": seed, "task": task,
                        "test_auc": metric["auc"], "test_logloss": metric["logloss"],
                        "observed": metric["observed"], "source": str(path)})
    if not raw:
        raise ValueError("no */S*/metrics.json results found")
    with (directory / "per_seed_task.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    statistics = []
    for variant, task in sorted({(r["variant"], r["task"]) for r in raw}):
        available = {seed: metrics[task]["auc"] for (arm, seed), metrics in runs.items()
                     if arm == variant and task in metrics and metrics[task]["auc"] is not None}
        common = sorted(s for s in available if ("full", s) in runs
                        and runs["full", s].get(task, {}).get("auc") is not None)
        deltas = [available[s] - runs["full", s][task]["auc"] for s in common]
        values = list(available.values())
        statistics.append({
            "variant": variant, "task": task, "n_seeds": len(values),
            "auc_mean": mean(values) if values else None,
            "auc_sample_std": stdev(values) if len(values) > 1 else None,
            "paired_seeds": ";".join(map(str, common)), "n_paired": len(common),
            "paired_delta_mean": mean(deltas) if deltas else None,
            "paired_delta_sample_std": stdev(deltas) if len(deltas) > 1 else None,
        })
    with (directory / "per_task_summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
    return statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input)
    print(f"Per-seed and paired per-task CSVs written to {args.input}")


if __name__ == "__main__":
    main()
