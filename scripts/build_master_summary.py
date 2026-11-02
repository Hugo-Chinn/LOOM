#!/usr/bin/env python3
"""Emit the curated per-dataset summary section of the master results table."""
from __future__ import annotations

import json
from pathlib import Path


OUTER = Path("/home/HugoChinn/LoopMT")
KEY_TASK = {
    "Taobao_Action": "buy",
    "MerRec_Action": "Purchase",
    "KuaiRand_Video_Action": "is_forward",
    "QK_Video_Action": "share",
    "TencentGR_10M_Action": "is_conversion",
}
MAIN_STUDIES = {
    "RankMixer": ["taobao_four_seed", "rankmixer_multiseed_confirmation", "taac_multiseed"],
    "OneTrans": ["onetrans_ns_s_state_multiseed", "onetrans_targeted_followup_multiseed",
                 "onetrans_all_datasets_multiseed"],
}
NONLOOP = {"Parallel", "Sequential", "Baseline"}


def fmt(v, d=6):
    return "-" if v is None else f"{v:.{d}f}"


def main():
    data = json.loads((OUTER / "reports/master_results_table_20261013.json").read_text())
    rows = data["rows"]
    lines = []
    for backbone in ("RankMixer", "OneTrans"):
        lines.append(f"### {backbone}")
        lines.append("")
        lines.append("| 数据集 | 基线（非 Loop） | 最佳 Loop 变体 | Macro（基线） | Macro（Loop） | ΔMacro | wins | p | 主转化任务 Δ |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for dataset in ["Taobao_Action", "MerRec_Action", "KuaiRand_Video_Action",
                        "QK_Video_Action", "TencentGR_10M_Action"]:
            pool = [r for r in rows if r["backbone"] == backbone and r["dataset"] == dataset
                    and r["study"] in MAIN_STUDIES[backbone] and r["n_seeds"] >= 3]
            if not pool:
                continue
            baselines = [r for r in pool if r["arm"] in NONLOOP]
            if baselines:
                base = max(baselines, key=lambda r: r["macro_mean"])
            else:
                base = None
            loops = [r for r in pool if r is not base and r["arm"] not in NONLOOP
                     and r.get("delta_macro_mean") is not None]
            if base is None or not loops:
                continue
            best = max(loops, key=lambda r: r["macro_mean"])
            task = KEY_TASK[dataset]
            task_delta = None
            if task in best["per_task"] and task in base["per_task"]:
                task_delta = best["per_task"][task]["mean"] - base["per_task"][task]["mean"]
            lines.append(
                "| {ds} | {base} | {arm} | {bmacro} ± {bstd} | {lmacro} ± {lstd} | "
                "{delta} ± {dstd} | {wins} | {p} | {task}: {td} |".format(
                    ds=dataset, base=base["arm"], arm=f"{best['arm']} ({best['study']})",
                    bmacro=fmt(base["macro_mean"]), bstd=fmt(base["macro_std"]),
                    lmacro=fmt(best["macro_mean"]), lstd=fmt(best["macro_std"]),
                    delta=fmt(best.get("delta_macro_mean")), dstd=fmt(best.get("delta_macro_std")),
                    wins="-" if best.get("wins") is None else f"{best['wins']}/{best.get('paired_seeds')}",
                    p=fmt(best.get("paired_p"), 4), task=task,
                    td=fmt(task_delta),
                )
            )
        lines.append("")
    (OUTER / "reports/master_summary_section.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
