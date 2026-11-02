#!/usr/bin/env python3
"""Summarize loop results for RankMixer, OneTrans, EST, HeMix, UltraHSTU."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("is_click", "cart", "fav", "buy")


def parse(path: Path):
    text = path.read_bytes().replace(b"\r", b"\n").replace(b"\0", b"").decode("utf-8", "ignore")
    params = re.search(r"Total number of parameters: (\d+)", text)
    found = {}
    pattern = re.compile(
        r"\[Task: ([^]]+)\]\[Metrics\] "
        r"(?:AUC: ([0-9.]+) - logloss: ([0-9.]+)|logloss: ([0-9.]+) - AUC: ([0-9.]+))"
    )
    for match in pattern.finditer(text):
        found[match.group(1)] = {
            "AUC": float(match.group(5) or match.group(2)),
            "LogLoss": float(match.group(4) or match.group(3)),
        }
    return {"parameters": int(params.group(1)), "test": {task: found[task] for task in TASKS}}


def main():
    all_base = json.loads((ROOT.parent / "reports/unirank_taobao_all_baselines_results.json").read_text())
    base = {row["model"]: {"parameters": row["parameters"], "test": row["test"]} for row in all_base}
    variants = {
        "RankMixer": {
            "looped": ROOT / "runs_local/taobao_phase1/RankMixer_Taobao_P1_Looped.stdout.log",
            "deep": ROOT / "runs_local/taobao_phase1/RankMixer_Taobao_P1_Looped_DeepSupervision.stdout.log",
        },
        "OneTrans": {
            "looped": ROOT / "runs_local/taobao_onetrans_multigpu/OneTrans_Taobao_P1_Looped.stdout.log",
            "deep": ROOT / "runs_local/taobao_onetrans_multigpu/OneTrans_Taobao_P1_Looped_DeepSupervision.stdout.log",
        },
        "EST": {
            "looped": ROOT / "runs_local/taobao_top_backbone_loops/EST_Taobao_Looped.stdout.log",
            "deep": ROOT / "runs_local/taobao_top_backbone_loops/EST_Taobao_Looped_Deep.stdout.log",
        },
        "HeMix": {
            "looped": ROOT / "runs_local/taobao_top_backbone_loops/HeMix_Taobao_Looped.stdout.log",
            "deep": ROOT / "runs_local/taobao_top_backbone_loops/HeMix_Taobao_Looped_Deep.stdout.log",
        },
        "UltraHSTU": {
            "looped": ROOT / "runs_local/taobao_top_backbone_loops/UltraHSTU_Taobao_Looped.stdout.log",
            "deep": ROOT / "runs_local/taobao_top_backbone_loops/UltraHSTU_Taobao_Looped_Deep.stdout.log",
        },
    }
    rows = []
    for model, paths in variants.items():
        baseline = base[model]
        for variant, path in paths.items():
            result = parse(path)
            rows.append({
                "backbone": model,
                "variant": variant,
                **result,
                "baseline_parameters": baseline["parameters"],
                "baseline_test": baseline["test"],
                "buy_auc_delta": result["test"]["buy"]["AUC"] - baseline["test"]["buy"]["AUC"],
                "parameter_reduction": 1.0 - result["parameters"] / baseline["parameters"],
            })
    (ROOT.parent / "reports/unirank_taobao_cross_backbone_loop_results.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    lines = [
        "# Taobao cross-backbone loop results — 2026-09-23",
        "",
        "All results use the same local single-GPU protocol: batch size 8192, one epoch, seed 20262027, and the same public Taobao split and evaluation code.",
        "",
        "| Backbone | Variant | Params | Param change | Click AUC | Cart AUC | Fav AUC | Buy AUC | Buy delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in variants:
        b = base[model]
        lines.append(
            f"| {model} | baseline | {b['parameters']:,} | — | "
            f"{b['test']['is_click']['AUC']:.6f} | {b['test']['cart']['AUC']:.6f} | "
            f"{b['test']['fav']['AUC']:.6f} | {b['test']['buy']['AUC']:.6f} | — |"
        )
        for variant in ("looped", "deep"):
            row = next(r for r in rows if r["backbone"] == model and r["variant"] == variant)
            t = row["test"]
            lines.append(
                f"| {model} | {variant} | {row['parameters']:,} | "
                f"{-100 * row['parameter_reduction']:+.1f}% | "
                f"{t['is_click']['AUC']:.6f} | {t['cart']['AUC']:.6f} | "
                f"{t['fav']['AUC']:.6f} | {t['buy']['AUC']:.6f} | "
                f"{row['buy_auc_delta']:+.6f} |"
            )
    output = ROOT.parent / "reports/UNIRANK_TAOBAO_CROSS_BACKBONE_LOOP_20260923.md"
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
