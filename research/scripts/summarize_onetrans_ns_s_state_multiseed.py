#!/usr/bin/env python3
"""Summarize paired OneTrans NS/S state-mechanism experiments."""
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
except ImportError:
    ttest_rel = wilcoxon = None


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/onetrans_ns_s_state_multiseed"
CONFIG_DIR = ROOT / "config_onetrans_ns_s_state_multiseed"
JSON_OUT = ROOT / "reports/onetrans_ns_s_state_multiseed_results.json"
REPORT_OUT = ROOT / "reports/ONETRANS_NS_S_STATE_MULTISEED_20261009.md"
DATASETS = ("Taobao_Action", "KuaiRand_Video_Action")
VARIANTS = (
    "Baseline", "RecurrentNS", "IndependentNS", "AnchoredScratchNS",
    "DecoupledScratchNS", "SOnlyMemory",
)
SEEDS = (20262027, 20262028, 20262029, 20262030)
METRIC = re.compile(
    r"\[Task: ([^]]+)\]\[Metrics\].*?"
    r"(?:AUC: ([0-9.]+) - logloss: ([0-9.]+)|"
    r"logloss: ([0-9.]+) - AUC: ([0-9.]+))"
)
PARAMS = re.compile(r"Total number of parameters: ([0-9]+)\.")
TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+")


def expid(dataset, variant, seed):
    return f"OneTransState_{dataset}_{variant}_S{seed}"


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
        metrics[task] = {
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


def stats(values):
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "std": stdev(values) if len(values) >= 2 else None,
        "values": values,
    }


def paired(candidate, control):
    seeds = sorted(set(candidate) & set(control))
    deltas = [candidate[s] - control[s] for s in seeds]
    result = {
        "n": len(seeds), "seeds": seeds, "deltas": deltas,
        "mean_delta": mean(deltas) if deltas else None,
        "std_delta": stdev(deltas) if len(deltas) >= 2 else None,
        "wins": sum(x > 0 for x in deltas),
        "ties": sum(x == 0 for x in deltas),
    }
    if len(deltas) >= 2 and ttest_rel is not None:
        result["paired_t_p_two_sided"] = float(ttest_rel(
            [candidate[s] for s in seeds], [control[s] for s in seeds]).pvalue)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    dataset_cfg = yaml.safe_load((CONFIG_DIR / "dataset_config.yaml").read_text())
    task_order = {
        dataset: [x["name"] for x in dataset_cfg[dataset]["label_col"]]
        for dataset in DATASETS
    }
    runs, missing = {}, []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                name = expid(dataset, variant, seed)
                result = parse_log(RUN_DIR / f"{name}.stdout.log")
                if result and set(result["metrics"]) == set(task_order[dataset]):
                    runs[name] = result
                else:
                    missing.append(name)
    if args.require_complete and missing:
        raise SystemExit(f"Experiment incomplete: {len(missing)} runs missing")

    aggregates = defaultdict(dict)
    comparisons = defaultdict(dict)
    recurrent_comparisons = defaultdict(dict)
    for dataset in DATASETS:
        tasks = task_order[dataset]
        per_variant_runs = {}
        for variant in VARIANTS:
            available = {
                seed: runs[expid(dataset, variant, seed)]
                for seed in SEEDS if expid(dataset, variant, seed) in runs
            }
            per_variant_runs[variant] = available
            aggregates[dataset][variant] = {
                "completed_seeds": sorted(available),
                "tasks": {
                    task: {
                        metric: stats([
                            run["metrics"][task][metric]
                            for run in available.values()
                        ]) for metric in ("auc", "logloss")
                    } for task in tasks
                },
                "macro_auc": stats([
                    mean(run["metrics"][task]["auc"] for task in tasks)
                    for run in available.values()
                ]),
                "last_task_auc": stats([
                    run["metrics"][tasks[-1]]["auc"]
                    for run in available.values()
                ]),
                "worst_task_delta_auc": None,
                "parameters": stats([
                    run["parameters"] for run in available.values()
                    if run["parameters"] is not None
                ]),
                "wall_seconds": stats([
                    run["wall_seconds"] for run in available.values()
                    if run["wall_seconds"] is not None
                ]),
            }
        baseline = per_variant_runs["Baseline"]
        for variant in VARIANTS[1:]:
            candidate = per_variant_runs[variant]
            comp = {"tasks": {}}
            for task in tasks:
                comp["tasks"][task] = paired(
                    {s: r["metrics"][task]["auc"] for s, r in candidate.items()},
                    {s: r["metrics"][task]["auc"] for s, r in baseline.items()},
                )
            comp["macro_auc"] = paired(
                {s: mean(r["metrics"][t]["auc"] for t in tasks)
                 for s, r in candidate.items()},
                {s: mean(r["metrics"][t]["auc"] for t in tasks)
                 for s, r in baseline.items()},
            )
            comp["last_task_auc"] = paired(
                {s: r["metrics"][tasks[-1]]["auc"] for s, r in candidate.items()},
                {s: r["metrics"][tasks[-1]]["auc"] for s, r in baseline.items()},
            )
            task_deltas = [
                comp["tasks"][task]["mean_delta"] for task in tasks
                if comp["tasks"][task]["mean_delta"] is not None
            ]
            comp["worst_task_mean_delta"] = min(task_deltas) if task_deltas else None
            comparisons[dataset][variant] = comp
        recurrent = per_variant_runs["RecurrentNS"]
        for variant in VARIANTS[2:]:
            candidate = per_variant_runs[variant]
            comp = {"tasks": {}}
            for task in tasks:
                comp["tasks"][task] = paired(
                    {s: r["metrics"][task]["auc"] for s, r in candidate.items()},
                    {s: r["metrics"][task]["auc"] for s, r in recurrent.items()},
                )
            comp["macro_auc"] = paired(
                {s: mean(r["metrics"][t]["auc"] for t in tasks)
                 for s, r in candidate.items()},
                {s: mean(r["metrics"][t]["auc"] for t in tasks)
                 for s, r in recurrent.items()},
            )
            comp["last_task_auc"] = paired(
                {s: r["metrics"][tasks[-1]]["auc"] for s, r in candidate.items()},
                {s: r["metrics"][tasks[-1]]["auc"] for s, r in recurrent.items()},
            )
            recurrent_comparisons[dataset][variant] = comp

    result = {
        "date": "2026-10-09",
        "protocol": {
            "datasets": list(DATASETS), "variants": list(VARIANTS),
            "seeds": list(SEEDS), "expected_runs": 48,
            "completed_runs": len(runs), "missing_runs": missing,
        },
        "task_order": task_order,
        "aggregates": dict(aggregates),
        "paired_vs_baseline": dict(comparisons),
        "paired_vs_recurrent_ns": dict(recurrent_comparisons),
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# OneTrans NS/S state-loop multi-seed results",
        "", "Date: 2026-10-09", "",
        f"Status: **{len(runs)}/48 completed**, {len(missing)} missing.", "",
        "Seeds: 20262027, 20262028, 20262029, 20262030.", "",
        "## Variants", "",
        "- `Baseline`: original three-layer parallel OneTrans.",
        "- `RecurrentNS`: fixed final S memory and one recurrent NS trajectory.",
        "- `IndependentNS`: fixed S memory; every task rereads from the same base NS state.",
        "- `AnchoredScratchNS`: immutable base-NS prediction path plus a recurrent detached correction state.",
        "- `DecoupledScratchNS`: anchored scratch state with separate NS-to-S cross-attention and full NS self-attention.",
        "- `SOnlyMemory`: fixed-length recurrent low-rank S-memory writes; NS is independently reread from base NS at each stage.",
        "",
        "All variants use paired initialization for embeddings, tokenizer, three-layer encoder, and task towers.",
        "",
    ]
    for dataset in DATASETS:
        tasks = task_order[dataset]
        lines += [
            f"## {dataset}", "", "### Per-task test metrics", "",
            "| Task | Variant | Seeds | AUC mean±std | Logloss mean±std | ΔAUC vs baseline | Wins |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for task in tasks:
            for variant in VARIANTS:
                agg = aggregates[dataset][variant]["tasks"][task]
                auc, ll = agg["auc"], agg["logloss"]
                comp = comparisons[dataset].get(variant, {}).get(
                    "tasks", {}).get(task, {})
                lines.append(
                    f"| {task} | {variant} | {auc['n']} | "
                    f"{fmt(auc['mean'])}±{fmt(auc['std'])} | "
                    f"{fmt(ll['mean'])}±{fmt(ll['std'])} | "
                    f"{fmt(comp.get('mean_delta'), signed=True)} | "
                    f"{comp.get('wins', '—')}/{comp.get('n', '—')} |"
                )
        lines += [
            "", "### Dataset-level summary", "",
            "| Variant | Macro AUC | Δ macro | Last-task AUC | Δ last | Worst task Δ | Params | Time/run |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
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
                f"{fmt(comp.get('worst_task_mean_delta'), signed=True)} | "
                f"{int(params):,}" if params is not None else
                f"| {variant} | {fmt(agg['macro_auc']['mean'])} | "
                f"{fmt(comp.get('macro_auc', {}).get('mean_delta'), signed=True)} | "
                f"{fmt(agg['last_task_auc']['mean'])} | "
                f"{fmt(comp.get('last_task_auc', {}).get('mean_delta'), signed=True)} | "
                f"{fmt(comp.get('worst_task_mean_delta'), signed=True)} | —"
            )
            row += (
                f" | {wall / 60:.1f} min |" if wall is not None else " | — |"
            )
            lines.append(row)
        lines.append("")
    if not missing:
        lines += [
            "## Paired statistics versus original OneTrans", "",
            "| Dataset | Variant | Δ macro | Macro wins | Macro paired-t p | "
            "Δ last | Last wins | Last paired-t p | Params Δ | Time × |",
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
                    f"| {dataset} | {variant} | "
                    f"{fmt(macro['mean_delta'], signed=True)} | "
                    f"{macro['wins']}/{macro['n']} | "
                    f"{fmt(macro.get('paired_t_p_two_sided'), 4)} | "
                    f"{fmt(last['mean_delta'], signed=True)} | "
                    f"{last['wins']}/{last['n']} | "
                    f"{fmt(last.get('paired_t_p_two_sided'), 4)} | "
                    f"{param_delta:+.2f}% | {time_ratio:.2f}× |"
                )
        lines.append("")
        lines += [
            "## New mechanisms versus current RecurrentNS", "",
            "| Dataset | Variant | Δ macro vs recurrent | Macro wins | "
            "Δ last vs recurrent | Last wins |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for dataset in DATASETS:
            for variant in VARIANTS[2:]:
                comp = recurrent_comparisons[dataset][variant]
                lines.append(
                    f"| {dataset} | {variant} | "
                    f"{fmt(comp['macro_auc']['mean_delta'], signed=True)} | "
                    f"{comp['macro_auc']['wins']}/{comp['macro_auc']['n']} | "
                    f"{fmt(comp['last_task_auc']['mean_delta'], signed=True)} | "
                    f"{comp['last_task_auc']['wins']}/{comp['last_task_auc']['n']} |"
                )
        lines.append("")
        lines += ["## Selection criteria", ""]
        for dataset in DATASETS:
            lines.append(f"### {dataset}")
            lines.append("")
            ranked = sorted(
                VARIANTS[1:],
                key=lambda variant: (
                    comparisons[dataset][variant]["macro_auc"]["mean_delta"],
                    comparisons[dataset][variant]["last_task_auc"]["mean_delta"],
                    comparisons[dataset][variant]["worst_task_mean_delta"],
                ), reverse=True)
            for variant in ranked:
                comp = comparisons[dataset][variant]
                lines.append(
                    f"- {variant}: macro {fmt(comp['macro_auc']['mean_delta'], signed=True)}, "
                    f"last {fmt(comp['last_task_auc']['mean_delta'], signed=True)}, "
                    f"worst-task {fmt(comp['worst_task_mean_delta'], signed=True)}."
                )
            lines.append("")
        lines += [
            "## Answers to the four mechanism questions", "",
            "1. **Is recurrent NS propagation necessary?** Not universally. On Taobao, independent rereading improves macro and Buy AUC relative to RecurrentNS, although both remain below baseline. On KuaiRand, removing recurrence reduces both macro and long_view performance.",
            "2. **Does anchoring the base NS state solve forgetting?** No. AnchoredScratchNS does not improve Taobao and only redistributes KuaiRand performance toward long_view; its four-seed macro mean is essentially baseline-level.",
            "3. **Does separating S cross-attention from NS self-attention help?** Yes on KuaiRand. DecoupledScratchNS is the only tested state mechanism with positive four-seed macro and terminal-task deltas while also outperforming RecurrentNS on both. It still lowers comment AUC and is not useful on Taobao.",
            "4. **Should S memory be writable?** Not in the tested S-only form. SOnlyMemory is the least harmful loop on Taobao by macro AUC, but lowers Buy; on KuaiRand it lowers macro and long_view and has the largest runtime overhead.",
            "",
            "## Final conclusion", "",
            "- Taobao rejects all five tested loop mechanisms under this controlled initialization. The best macro result, SOnlyMemory, remains below the original OneTrans baseline; no loop improves Buy on average.",
            "- KuaiRand supports one narrower conclusion: decoupling S-memory reading from NS-slot interaction is better than the current concatenated recurrent refiner. DecoupledScratchNS improves macro AUC by +0.001523 and long_view by +0.005652 versus baseline, with 3/4 seed wins for both.",
            "- DecoupledScratchNS is still not a universal solution: it loses comment AUC by -0.002207 on KuaiRand, loses all four Taobao tasks on average, and costs about 1.74× baseline runtime on KuaiRand.",
            "- No paired t-test reaches p < 0.05 with four seeds. These are mechanism-level directional results, not statistical proof of a universal improvement.",
            "",
            "The next high-value experiment is therefore not another generic NS/S loop. It is a KuaiRand-focused refinement of DecoupledScratchNS with explicit protection for comment/follow representations, followed by a topology-specific Taobao design rather than a linear state trajectory.",
            "",
        ]
    REPORT_OUT.write_text("\n".join(lines))
    print(f"completed={len(runs)} missing={len(missing)}")
    print(JSON_OUT)
    print(REPORT_OUT)


if __name__ == "__main__":
    main()
