#!/usr/bin/env python3
"""Summarize the four-seed multi-task baseline sweep.

Per (dataset, model) the script reports the macro test AUC (mean over tasks)
and the per-task AUCs, averaged over the four canonical seeds, plus the paired
delta against the ``ParallelIndependentTower`` control arm of the same dataset.

Usage:
    python scripts/summarize_mtl_baselines_four_seed.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy.stats import ttest_rel
except ImportError:  # pragma: no cover - scipy absent in minimal envs
    ttest_rel = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_all_local_experiments import parse_log  # noqa: E402

CONFIG = ROOT / "config_mtl_baselines_four_seed"
LOG_DIR = ROOT / "logs" / "mtl_baselines_four_seed"
OUT_DIR = ROOT / "reports"
CONTROL = "ParallelIndependentTower"


def load_arms() -> list[str]:
    return (CONFIG / "expids.txt").read_text().split()


def main() -> None:
    arms = load_arms()
    by_dataset_model: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    pending = []
    for expid in arms:
        log = LOG_DIR / f"{expid}.log"
        if not log.exists():
            pending.append(expid)
            continue
        parsed = parse_log(log)
        if parsed is None:
            pending.append(expid)
            continue
        model, dataset, seed = split_expid(expid)
        aucs = {task: value["auc"] for task, value in parsed["metrics"].items()}
        by_dataset_model[(dataset, model)][seed] = {
            "macro": mean(aucs.values()) if aucs else None,
            "per_task": aucs,
            "parameters": parsed.get("parameters"),
            "wall_seconds": parsed.get("wall_seconds"),
        }

    rows = []
    datasets = sorted({ds for ds, _ in by_dataset_model})
    for dataset in datasets:
        control = by_dataset_model.get((dataset, CONTROL), {})
        for (ds, model), seeds in sorted(by_dataset_model.items()):
            if ds != dataset:
                continue
            ordered = [seeds[s] for s in sorted(seeds) if seeds[s]["macro"] is not None]
            if not ordered:
                continue
            macro = [item["macro"] for item in ordered]
            row = {
                "dataset": dataset,
                "model": model,
                "n_seeds": len(macro),
                "macro_mean": mean(macro),
                "macro_std": stdev(macro) if len(macro) > 1 else 0.0,
                "macro_per_seed": macro,
                "per_task_mean": {
                    task: mean([item["per_task"][task] for item in ordered
                                if task in item["per_task"]])
                    for task in ordered[0]["per_task"]
                },
            }
            paired = [control[s]["macro"] for s in sorted(seeds)
                      if s in control and control[s]["macro"] is not None
                      and seeds[s]["macro"] is not None]
            if model != CONTROL and len(paired) == len(macro) and len(macro) > 1:
                delta = [a - b for a, b in zip(macro, paired)]
                row["delta_vs_control"] = mean(delta)
                row["wins"] = sum(1 for d in delta if d > 0)
                if ttest_rel is not None:
                    row["p_value"] = float(ttest_rel(macro, paired).pvalue)
            rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = "mtl_baselines_four_seed"
    (OUT_DIR / f"{stamp}_results.json").write_text(
        json.dumps({"rows": rows, "pending": pending}, indent=2))

    lines = ["# Multi-Task Baselines: four-seed sweep", ""]
    lines.append("Control arm: `ParallelIndependentTower` (UniRank backbone, "
                 "no task routing). Macro AUC is the mean over dataset tasks.")
    lines.append("")
    lines.append("| Dataset | Model | Macro AUC (mean ± std) | Δ vs control | Wins | p |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        delta = row.get("delta_vs_control")
        p = row.get("p_value")
        lines.append(
            "| {dataset} | {model} | {mean:.6f} ± {std:.6f} | {delta} | {wins} | {p} |".format(
                dataset=row["dataset"], model=row["model"],
                mean=row["macro_mean"], std=row["macro_std"],
                delta="—" if delta is None else f"{delta:+.6f}",
                wins="—" if delta is None else f"{row['wins']}/{row['n_seeds']}",
                p="—" if p is None else f"{p:.4f}"))
    lines.append("")
    if pending:
        lines.append(f"Pending / not finished yet: {len(pending)} arms.")
    (OUT_DIR / f"{stamp.upper()}.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} rows, {len(pending)} pending -> {OUT_DIR}")


def split_expid(expid: str) -> tuple[str, str, int]:
    seed_tag = expid.rsplit("_", 1)[1]
    remainder = expid[: -(len(seed_tag) + 1)]
    dataset = remainder
    model = ""
    for candidate in ("TencentGR_10M_Action", "QK_Video_Action",
                      "KuaiRand_Video_Action", "MerRec_Action", "Taobao_Action"):
        if remainder.endswith("_" + candidate):
            model = remainder[: -(len(candidate) + 1)]
            dataset = candidate
            break
    return model, dataset, int(seed_tag.replace("S", ""))


if __name__ == "__main__":
    main()
