#!/usr/bin/env python3
"""Summarize targeted OneTrans DAG/protection/lightweight follow-ups."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy.stats import ttest_rel, wilcoxon
except ImportError:
    ttest_rel = wilcoxon = None


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/onetrans_targeted_followup_multiseed"
OLD_JSON = ROOT / "reports/onetrans_ns_s_state_multiseed_results.json"
JSON_OUT = ROOT / "reports/onetrans_targeted_followup_multiseed_results.json"
REPORT_OUT = ROOT / "reports/ONETRANS_TARGETED_FOLLOWUP_MULTISEED_20261009.md"
SEEDS = (20262027, 20262028, 20262029, 20262030)
DATASETS = {
    "Taobao_Action": {
        "control": "IndependentNS",
        "variants": ("ContextDAGMean", "ContextDAGLearned"),
    },
    "KuaiRand_Video_Action": {
        "control": "DecoupledScratchNS",
        "variants": (
            "DecProtect05", "DecProtect10", "CrossOnly",
            "CrossOnlyProtect05",
        ),
    },
}
METRIC = re.compile(
    r"\[Task: ([^]]+)\]\[Metrics\].*?"
    r"(?:AUC: ([0-9.]+) - logloss: ([0-9.]+)|"
    r"logloss: ([0-9.]+) - AUC: ([0-9.]+))"
)
PARAMS = re.compile(r"Total number of parameters: ([0-9]+)\.")
TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+")


def expid(dataset, variant, seed):
    return f"OneTransTarget_{dataset}_{variant}_S{seed}"


def parse_log(path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    if "******** Test evaluation ********" not in text:
        return None
    section = text.rsplit("******** Test evaluation ********", 1)[1]
    metrics = {}
    for match in METRIC.finditer(section):
        metrics[match.group(1)] = {
            "auc": float(match.group(5) or match.group(2)),
            "logloss": float(match.group(4) or match.group(3)),
        }
    param_match = PARAMS.search(text)
    times = [
        datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        for line in text.splitlines() if (m := TIMESTAMP.match(line))
    ]
    return {
        "metrics": metrics,
        "parameters": int(param_match.group(1)) if param_match else None,
        "wall_seconds": (
            (times[-1] - times[0]).total_seconds() if len(times) >= 2 else None
        ),
    }


def stat(values):
    return {
        "n": len(values), "mean": mean(values) if values else None,
        "std": stdev(values) if len(values) >= 2 else None, "values": values,
    }


def paired(candidate, reference):
    seeds = sorted(set(candidate) & set(reference))
    deltas = [candidate[s] - reference[s] for s in seeds]
    result = {
        "n": len(seeds), "seeds": seeds, "deltas": deltas,
        "mean_delta": mean(deltas) if deltas else None,
        "std_delta": stdev(deltas) if len(deltas) >= 2 else None,
        "wins": sum(x > 0 for x in deltas),
        "ties": sum(x == 0 for x in deltas),
    }
    if len(deltas) >= 2 and ttest_rel is not None:
        result["paired_t_p_two_sided"] = float(ttest_rel(
            [candidate[s] for s in seeds], [reference[s] for s in seeds]).pvalue)
        try:
            result["wilcoxon_p_two_sided"] = float(
                wilcoxon(deltas, alternative="two-sided").pvalue)
        except ValueError:
            result["wilcoxon_p_two_sided"] = None
    return result


def fmt(value, digits=6, signed=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def old_seed_values(old, dataset, variant, task, metric="auc"):
    values = old["aggregates"][dataset][variant]["tasks"][task][metric]["values"]
    completed = old["aggregates"][dataset][variant]["completed_seeds"]
    return {int(seed): value for seed, value in zip(completed, values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    old = json.loads(OLD_JSON.read_text())
    task_order = old["task_order"]
    runs, missing = {}, []
    for dataset, spec in DATASETS.items():
        for variant in spec["variants"]:
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
    for dataset, spec in DATASETS.items():
        tasks = task_order[dataset]
        baseline = {
            task: old_seed_values(old, dataset, "Baseline", task) for task in tasks
        }
        control_name = spec["control"]
        control = {
            task: old_seed_values(old, dataset, control_name, task) for task in tasks
        }
        for variant in spec["variants"]:
            available = {
                seed: runs[expid(dataset, variant, seed)]
                for seed in SEEDS if expid(dataset, variant, seed) in runs
            }
            aggregates[dataset][variant] = {
                "completed_seeds": sorted(available),
                "tasks": {
                    task: {
                        metric: stat([
                            run["metrics"][task][metric]
                            for run in available.values()
                        ]) for metric in ("auc", "logloss")
                    } for task in tasks
                },
                "macro_auc": stat([
                    mean(run["metrics"][task]["auc"] for task in tasks)
                    for run in available.values()
                ]),
                "last_task_auc": stat([
                    run["metrics"][tasks[-1]]["auc"]
                    for run in available.values()
                ]),
                "parameters": stat([
                    run["parameters"] for run in available.values()
                    if run["parameters"] is not None
                ]),
                "wall_seconds": stat([
                    run["wall_seconds"] for run in available.values()
                    if run["wall_seconds"] is not None
                ]),
            }
            candidate = {
                task: {s: r["metrics"][task]["auc"] for s, r in available.items()}
                for task in tasks
            }
            comp = {"vs_baseline": {"tasks": {}}, "vs_control": {"tasks": {}}}
            for task in tasks:
                comp["vs_baseline"]["tasks"][task] = paired(
                    candidate[task], baseline[task])
                comp["vs_control"]["tasks"][task] = paired(
                    candidate[task], control[task])
            candidate_macro = {
                s: mean(r["metrics"][t]["auc"] for t in tasks)
                for s, r in available.items()
            }
            baseline_macro = {
                s: mean(baseline[t][s] for t in tasks) for s in SEEDS
            }
            control_macro = {
                s: mean(control[t][s] for t in tasks) for s in SEEDS
            }
            candidate_last = {s: r["metrics"][tasks[-1]]["auc"]
                              for s, r in available.items()}
            for label, reference_tasks, reference_macro in (
                    ("vs_baseline", baseline, baseline_macro),
                    ("vs_control", control, control_macro)):
                comp[label]["macro_auc"] = paired(candidate_macro, reference_macro)
                comp[label]["last_task_auc"] = paired(
                    candidate_last, reference_tasks[tasks[-1]])
                deltas = [
                    comp[label]["tasks"][task]["mean_delta"] for task in tasks
                    if comp[label]["tasks"][task]["mean_delta"] is not None
                ]
                comp[label]["worst_task_mean_delta"] = min(deltas) if deltas else None
            comparisons[dataset][variant] = comp

    result = {
        "date": "2026-10-09",
        "protocol": {
            "seeds": list(SEEDS), "expected_new_runs": 24,
            "completed_new_runs": len(runs), "missing_runs": missing,
            "controls": {dataset: spec["control"] for dataset, spec in DATASETS.items()},
        },
        "task_order": task_order,
        "aggregates": dict(aggregates),
        "comparisons": dict(comparisons),
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# OneTrans targeted follow-up multi-seed results", "",
        "Date: 2026-10-09", "",
        f"Status: **{len(runs)}/24 new runs completed**, {len(missing)} missing.", "",
        "Controls are reused from the paired NS/S state experiment with identical seeds and shared initialization.", "",
    ]
    for dataset, spec in DATASETS.items():
        tasks = task_order[dataset]
        lines += [f"## {dataset}", "", f"Control: `{spec['control']}`", "",
                  "### Per-task metrics", "",
                  "| Task | Variant | Seeds | AUC mean±std | Logloss mean±std | Δ vs baseline | Δ vs control |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for task in tasks:
            for variant in spec["variants"]:
                agg = aggregates[dataset][variant]["tasks"][task]
                comp = comparisons[dataset][variant]
                lines.append(
                    f"| {task} | {variant} | {agg['auc']['n']} | "
                    f"{fmt(agg['auc']['mean'])}±{fmt(agg['auc']['std'])} | "
                    f"{fmt(agg['logloss']['mean'])}±{fmt(agg['logloss']['std'])} | "
                    f"{fmt(comp['vs_baseline']['tasks'][task]['mean_delta'], signed=True)} | "
                    f"{fmt(comp['vs_control']['tasks'][task]['mean_delta'], signed=True)} |"
                )
        lines += ["", "### Dataset-level summary", "",
                  "| Variant | Macro AUC | Δ macro vs base | Δ macro vs control | Last AUC | Δ last vs base | Δ last vs control | Worst Δ vs base | Time/run |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for variant in spec["variants"]:
            agg = aggregates[dataset][variant]
            comp = comparisons[dataset][variant]
            wall = agg["wall_seconds"]["mean"]
            lines.append(
                f"| {variant} | {fmt(agg['macro_auc']['mean'])} | "
                f"{fmt(comp['vs_baseline']['macro_auc']['mean_delta'], signed=True)} | "
                f"{fmt(comp['vs_control']['macro_auc']['mean_delta'], signed=True)} | "
                f"{fmt(agg['last_task_auc']['mean'])} | "
                f"{fmt(comp['vs_baseline']['last_task_auc']['mean_delta'], signed=True)} | "
                f"{fmt(comp['vs_control']['last_task_auc']['mean_delta'], signed=True)} | "
                f"{fmt(comp['vs_baseline']['worst_task_mean_delta'], signed=True)} | "
                f"{wall / 60:.1f} min |" if wall is not None else "| — |"
            )
        lines.append("")

    if not missing:
        lines += ["## Paired statistics", "",
                  "| Dataset | Variant | Reference | Δ macro | Macro wins | Macro p | Δ last | Last wins | Last p |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for dataset, spec in DATASETS.items():
            for variant in spec["variants"]:
                for label, reference in (("vs_baseline", "Baseline"),
                                         ("vs_control", spec["control"])):
                    comp = comparisons[dataset][variant][label]
                    macro, last = comp["macro_auc"], comp["last_task_auc"]
                    lines.append(
                        f"| {dataset} | {variant} | {reference} | "
                        f"{fmt(macro['mean_delta'], signed=True)} | {macro['wins']}/{macro['n']} | "
                        f"{fmt(macro.get('paired_t_p_two_sided'), 4)} | "
                        f"{fmt(last['mean_delta'], signed=True)} | {last['wins']}/{last['n']} | "
                        f"{fmt(last.get('paired_t_p_two_sided'), 4)} |"
                    )
        lines.append("")
    REPORT_OUT.write_text("\n".join(lines))
    print(f"completed={len(runs)} missing={len(missing)}")
    print(JSON_OUT)
    print(REPORT_OUT)


if __name__ == "__main__":
    main()
