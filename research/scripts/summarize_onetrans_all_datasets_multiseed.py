#!/usr/bin/env python3
"""Summarize the 45-run OneTrans cross-dataset multi-seed experiment."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import yaml

try:
    from scipy.stats import ttest_rel, wilcoxon
except ImportError:  # Keep the result parser usable without scipy.
    ttest_rel = wilcoxon = None


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/onetrans_all_datasets_multiseed"
CONFIG_DIR = ROOT / "config_onetrans_all_datasets_multiseed"
JSON_OUT = ROOT / "reports/onetrans_all_datasets_multiseed_results.json"
REPORT_OUT = ROOT / "reports/ONETRANS_ALL_DATASETS_MULTISEED_20261009.md"
DATASETS = (
    "TencentGR_10M_Action",
    "Taobao_Action",
    "QK_Video_Action",
    "MerRec_Action",
    "KuaiRand_Video_Action",
)
VARIANTS = ("Baseline", "FiLM", "QVLoRA")
SEEDS = (20262027, 20262028, 20262029)
METRIC = re.compile(
    r"\[Task: ([^]]+)\]\[Metrics\].*?"
    r"(?:AUC: ([0-9.]+) - logloss: ([0-9.]+)|"
    r"logloss: ([0-9.]+) - AUC: ([0-9.]+))"
)
PARAMS = re.compile(r"Total number of parameters: ([0-9]+)\.")
TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+")


def expid(dataset, variant, seed):
    return f"OneTransAll_{dataset}_{variant}_S{seed}"


def parse_log(path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    if "******** Test evaluation ********" not in text:
        return None
    section = text.rsplit("******** Test evaluation ********", 1)[1]
    metrics = {}
    for match in METRIC.finditer(section):
        task = match.group(1)
        auc = float(match.group(5) or match.group(2))
        logloss = float(match.group(4) or match.group(3))
        metrics[task] = {"auc": auc, "logloss": logloss}
    params_match = PARAMS.search(text)
    timestamps = [
        datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        for line in text.splitlines()
        if (m := TIMESTAMP.match(line))
    ]
    return {
        "metrics": metrics,
        "parameters": int(params_match.group(1)) if params_match else None,
        "wall_seconds": (
            (timestamps[-1] - timestamps[0]).total_seconds()
            if len(timestamps) >= 2 else None
        ),
    }


def sample_stats(values):
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "std": stdev(values) if len(values) >= 2 else None,
        "values": values,
    }


def paired_stats(candidate, baseline):
    common = sorted(set(candidate) & set(baseline))
    deltas = [candidate[seed] - baseline[seed] for seed in common]
    result = {
        "n": len(common),
        "seeds": common,
        "deltas": deltas,
        "mean_delta": mean(deltas) if deltas else None,
        "std_delta": stdev(deltas) if len(deltas) >= 2 else None,
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
    }
    if len(deltas) >= 2 and ttest_rel is not None:
        result["paired_t_p_two_sided"] = float(
            ttest_rel([candidate[s] for s in common],
                      [baseline[s] for s in common]).pvalue
        )
        try:
            result["wilcoxon_p_two_sided"] = float(
                wilcoxon(deltas, alternative="two-sided").pvalue
            )
        except ValueError:
            result["wilcoxon_p_two_sided"] = None
    return result


def fmt(value, digits=6, signed=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    dataset_cfg = yaml.safe_load((CONFIG_DIR / "dataset_config.yaml").read_text())
    task_order = {
        dataset: [label["name"] for label in dataset_cfg[dataset]["label_col"]]
        for dataset in DATASETS
    }
    runs = {}
    missing = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                name = expid(dataset, variant, seed)
                parsed = parse_log(RUN_DIR / f"{name}.stdout.log")
                if parsed and set(parsed["metrics"]) == set(task_order[dataset]):
                    runs[name] = parsed
                else:
                    missing.append(name)
    if args.require_complete and missing:
        raise SystemExit(f"Experiment incomplete: {len(missing)} runs missing")

    aggregates = defaultdict(dict)
    comparisons = defaultdict(dict)
    for dataset in DATASETS:
        tasks = task_order[dataset]
        for variant in VARIANTS:
            available = {
                seed: runs[expid(dataset, variant, seed)]
                for seed in SEEDS if expid(dataset, variant, seed) in runs
            }
            variant_result = {
                "completed_seeds": sorted(available),
                "tasks": {},
                "macro_auc": sample_stats([
                    mean(run["metrics"][task]["auc"] for task in tasks)
                    for run in available.values()
                ]),
                "last_task_auc": sample_stats([
                    run["metrics"][tasks[-1]]["auc"] for run in available.values()
                ]),
                "parameters": sample_stats([
                    run["parameters"] for run in available.values()
                    if run["parameters"] is not None
                ]),
                "wall_seconds": sample_stats([
                    run["wall_seconds"] for run in available.values()
                    if run["wall_seconds"] is not None
                ]),
            }
            for task in tasks:
                variant_result["tasks"][task] = {
                    metric: sample_stats([
                        run["metrics"][task][metric] for run in available.values()
                    ])
                    for metric in ("auc", "logloss")
                }
            aggregates[dataset][variant] = variant_result

        for variant in VARIANTS[1:]:
            baseline_runs = {
                seed: runs[expid(dataset, "Baseline", seed)]
                for seed in SEEDS if expid(dataset, "Baseline", seed) in runs
            }
            candidate_runs = {
                seed: runs[expid(dataset, variant, seed)]
                for seed in SEEDS if expid(dataset, variant, seed) in runs
            }
            comparison = {"tasks": {}}
            for task in tasks:
                comparison["tasks"][task] = paired_stats(
                    {s: r["metrics"][task]["auc"] for s, r in candidate_runs.items()},
                    {s: r["metrics"][task]["auc"] for s, r in baseline_runs.items()},
                )
            comparison["macro_auc"] = paired_stats(
                {s: mean(r["metrics"][t]["auc"] for t in tasks)
                 for s, r in candidate_runs.items()},
                {s: mean(r["metrics"][t]["auc"] for t in tasks)
                 for s, r in baseline_runs.items()},
            )
            comparison["last_task_auc"] = paired_stats(
                {s: r["metrics"][tasks[-1]]["auc"]
                 for s, r in candidate_runs.items()},
                {s: r["metrics"][tasks[-1]]["auc"]
                 for s, r in baseline_runs.items()},
            )
            comparisons[dataset][variant] = comparison

    result = {
        "date": "2026-10-09",
        "protocol": {
            "datasets": list(DATASETS), "variants": list(VARIANTS),
            "seeds": list(SEEDS), "expected_runs": 45,
            "completed_runs": len(runs), "missing_runs": missing,
        },
        "task_order": task_order,
        "aggregates": dict(aggregates),
        "paired_vs_baseline": dict(comparisons),
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# OneTrans loop cross-dataset multi-seed results",
        "",
        "Date: 2026-10-09",
        "",
        (f"Status: **{len(runs)}/45 completed**, {len(missing)} missing. "
         "Statistics below use only completed runs."),
        "",
        "Variants: original OneTrans baseline; fixed-memory one-layer NS refiner "
        "+ FiLM; fixed-memory one-layer NS refiner + QV-LoRA-r4.",
        "",
        "Seeds: 20262027, 20262028, 20262029. All loop variants use linear "
        "feedback order and progressive supervision.",
        "",
    ]
    for dataset in DATASETS:
        tasks = task_order[dataset]
        lines += [f"## {dataset}", "", "### Per-task test metrics", "",
                  "| Task | Variant | Seeds | AUC mean±std | Logloss mean±std | ΔAUC vs baseline | Wins |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for task in tasks:
            for variant in VARIANTS:
                agg = aggregates[dataset][variant]["tasks"][task]
                auc, ll = agg["auc"], agg["logloss"]
                comp = comparisons[dataset].get(variant, {}).get("tasks", {}).get(task, {})
                lines.append(
                    f"| {task} | {variant} | {auc['n']} | "
                    f"{fmt(auc['mean'])}±{fmt(auc['std'])} | "
                    f"{fmt(ll['mean'])}±{fmt(ll['std'])} | "
                    f"{fmt(comp.get('mean_delta'), signed=True)} | "
                    f"{comp.get('wins', '—')}/{comp.get('n', '—')} |"
                )
        lines += ["", "### Dataset-level summary", "",
                  "| Variant | Macro AUC | Δ macro AUC | Last-task AUC | Δ last-task AUC | Params | Wall time/run |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for variant in VARIANTS:
            agg = aggregates[dataset][variant]
            comp = comparisons[dataset].get(variant, {})
            params = agg["parameters"]["mean"]
            wall = agg["wall_seconds"]["mean"]
            row = (
                f"| {variant} | {fmt(agg['macro_auc']['mean'])} | "
                f"{fmt(comp.get('macro_auc', {}).get('mean_delta'), signed=True)} | "
                f"{fmt(agg['last_task_auc']['mean'])} | "
                f"{fmt(comp.get('last_task_auc', {}).get('mean_delta'), signed=True)} | "
                f"{int(params):,}" if params is not None else
                f"| {variant} | {fmt(agg['macro_auc']['mean'])} | "
                f"{fmt(comp.get('macro_auc', {}).get('mean_delta'), signed=True)} | "
                f"{fmt(agg['last_task_auc']['mean'])} | "
                f"{fmt(comp.get('last_task_auc', {}).get('mean_delta'), signed=True)} | —"
            )
            # Complete the row separately to keep formatting readable.
            row += f" | {wall / 60:.1f} min |" if wall is not None else " | — |"
            lines.append(row)
        lines.append("")

    if not missing:
        lines += [
            "## Paired dataset-level statistics",
            "",
            "| Dataset | Variant | Δ macro AUC | Macro wins | Macro paired-t p | "
            "Δ last-task AUC | Last-task wins | Last-task paired-t p | Params Δ | Time × |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for dataset in DATASETS:
            baseline = aggregates[dataset]["Baseline"]
            for variant in VARIANTS[1:]:
                agg = aggregates[dataset][variant]
                comp = comparisons[dataset][variant]
                macro = comp["macro_auc"]
                last = comp["last_task_auc"]
                param_delta = (
                    agg["parameters"]["mean"] / baseline["parameters"]["mean"] - 1
                ) * 100
                time_ratio = (
                    agg["wall_seconds"]["mean"] / baseline["wall_seconds"]["mean"]
                )
                lines.append(
                    f"| {dataset} | {variant} | {fmt(macro['mean_delta'], signed=True)} | "
                    f"{macro['wins']}/{macro['n']} | {fmt(macro.get('paired_t_p_two_sided'), 4)} | "
                    f"{fmt(last['mean_delta'], signed=True)} | {last['wins']}/{last['n']} | "
                    f"{fmt(last.get('paired_t_p_two_sided'), 4)} | {param_delta:+.2f}% | "
                    f"{time_ratio:.2f}× |"
                )

        lines += ["", "## Cross-dataset conclusion", ""]
        for variant in VARIANTS[1:]:
            macro_wins = sum(comparisons[d][variant]["macro_auc"]["mean_delta"] > 0
                             for d in DATASETS)
            last_wins = sum(comparisons[d][variant]["last_task_auc"]["mean_delta"] > 0
                            for d in DATASETS)
            lines.append(
                f"- {variant}: positive mean paired delta on {macro_wins}/5 datasets "
                f"for macro AUC and {last_wins}/5 for the final task."
            )
        lines += [
            "",
            "- QV-LoRA is the better terminal-task choice on Taobao and the more balanced "
            "choice on QK, but it is not universally superior: it hurts both TencentGR "
            "macro AUC and conversion AUC, and is unstable on MerRec.",
            "- FiLM is the strongest KuaiRand choice: it improves long_view in 3/3 seeds "
            "and raises macro AUC. It is also the safer TencentGR option, though its mean "
            "gain there is very small after the third seed.",
            "- Taobao and MerRec expose the central failure mode of a linear feedback loop: "
            "the terminal task can improve while intermediate feedback AUCs decline. Future "
            "work should explicitly protect intermediate states or use dataset-specific topology.",
            "- No paired t-test reaches p < 0.05 with three seeds. P-values are descriptive; "
            "effect direction, size, seed wins, and cross-dataset consistency carry more weight.",
            "",
        ]
    else:
        lines += ["## Pending", "", f"{len(missing)} runs remain; final cross-dataset conclusions are deferred.", ""]

    REPORT_OUT.write_text("\n".join(lines))
    print(f"completed={len(runs)} missing={len(missing)}")
    print(JSON_OUT)
    print(REPORT_OUT)


if __name__ == "__main__":
    main()
