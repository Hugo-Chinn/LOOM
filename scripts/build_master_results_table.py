#!/usr/bin/env python3
"""Build the master results table for RankMixer and OneTrans loop studies.

Rows are (backbone, dataset, study, arm). Per-seed test AUCs are re-parsed
from run logs (via summarize_all_local_experiments.parse_log) and merged with
dedicated multi-seed result JSONs, then aggregated to mean +- std with paired
deltas against the study baseline.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy.stats import ttest_rel
except ImportError:
    ttest_rel = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_all_local_experiments import parse_log, split_name  # noqa: E402

OUTER = Path("/home/HugoChinn/LoopMT")
RUNS = ROOT / "runs_local"

DATASETS = (
    "Taobao_Action", "KuaiRand_Video_Action", "MerRec_Action",
    "QK_Video_Action", "TencentGR_10M_Action",
)
DATASET_ALIAS = {
    "KuaiRand_Video": "KuaiRand_Video_Action",
    "TencentGR_10M": "TencentGR_10M_Action",
}
EXCLUDED_FAMILIES = {
    "all_baselines", "hidden_analysis", "cross_dataset_hidden", "loop_top_backbones",
    # covered by the dedicated 4-seed delta table below
    "rankmixer_depth_channel_multiseed",
}
# (dataset, study) remaps: merge replicated seed blocks into one study
STUDY_MERGE = {
    ("TencentGR_10M_Action", "rankmixer_multiseed_confirmation"): "taac_multiseed",
    ("TencentGR_10M_Action", "rankmixer_taac_six_seed"): "taac_multiseed",
}
# studies whose control lives in a sibling study (same protocol, paired init)
CROSS_STUDY_BASELINE = {
    ("OneTrans", "KuaiRand_Video_Action", "onetrans_targeted_followup_multiseed"):
        ("onetrans_ns_s_state_multiseed", "Baseline"),
    ("OneTrans", "Taobao_Action", "onetrans_targeted_followup_multiseed"):
        ("onetrans_ns_s_state_multiseed", "Baseline"),
}
STUDY_ORDER = [
    "taobao_four_seed", "taac_multiseed", "rankmixer_multiseed_confirmation",
    "head_sharing_4seed", "preference_margin_4seed",
    "onetrans_ns_s_state_multiseed", "onetrans_targeted_followup_multiseed",
    "onetrans_all_datasets_multiseed", "branch_gate_ablation", "rankmixer_multiseed",
    "rankmixer_taobao_raw_multiseed", "onetrans_refiner_multiseed",
    "onetrans_residual_multiseed", "onetrans_peft_multiseed",
    "onetrans_peft_pair_multiseed", "onetrans_warmstart",
]
ARM_ALIAS = {
    "LinearDeep": "Linear",
    "LinearGated": "Linear+Gate",
    "RawParallel": "Parallel",
    "Branch": "Branch",
    "DAGGated": "DAG+Gate",
    "DAGMeanPre2": "DAG+Mean+Pre2",
    "LoopDeep": "Loop+Deep",
    "Loop": "Loop",
    "GatedLoopDeep": "DAG+Gate+Deep",
}
BASELINE_OVERRIDE = {
    ("RankMixer", "Taobao_Action", "taobao_four_seed"): "Parallel",
}
STUDY_LABEL = {
    "taobao_four_seed": "Taobao 四 seed 主对比（含 raw Parallel 基线）",
    "head_sharing_4seed": "Head 共享（Taobao，4 seeds）",
    "preference_margin_4seed": "Preference margin（Taobao，4 seeds）",
    "depth_channel_4seed": "Depth channel scale（跨数据集，4 seeds，仅 delta）",
    "branch_gate_ablation": "Branch gate 消融（Taobao，3 seeds）",
    "rankmixer_multiseed": "Taobao Linear/Branch（3 seeds）",
    "rankmixer_multiseed_confirmation": "跨数据集 3 seed 确认",
    "taac_multiseed": "TAAC 多 seed 合并（S28-S30 + S31-S32；S33 训练中）",
    "rankmixer_taac_loop": "TAAC Loop 单 seed 筛查",
    "rankmixer_generalization": "跨数据集单 seed 泛化筛查",
    "rankmixer_hierarchical": "层级 DAG 单 seed 筛查",
    "rankmixer_hierarchical_phase2": "层级 DAG 二阶段单 seed 筛查",
    "rankmixer_safe_routing": "Safe routing 单 seed 筛查",
    "rankmixer_safe_routing_multiseed": "Safe routing 多 seed",
    "rankmixer_sample_trust": "Sample trust gate 筛查",
    "rankmixer_recurrence_conditioning": "Recurrence conditioning 筛查",
    "rankmixer_optimized": "Taobao 优化版筛查",
    "rankmixer_taobao_raw_multiseed": "Taobao raw/parallel 3 seed",
    "rankmixer_preference_margin": "Preference margin 筛查",
    "rankmixer_preference_margin_phase2": "Preference margin 二阶段",
    "rankmixer_preference_margin_phase3": "Preference margin 三阶段",
    "rankmixer_iteration_depth": "Iteration depth 筛查",
    "rankmixer_depth_gate": "Depth gate 筛查",
    "kuairand_routing_multiseed": "KuaiRand routing 多 seed",
    "kuairand_routing_screen": "KuaiRand routing 单 seed 筛查",
    "onetrans_all_datasets_multiseed": "OneTrans FiLM/QVLoRA（跨数据集，3 seeds）",
    "onetrans_ns_s_state_multiseed": "OneTrans NS/S 状态循环（4 seeds）",
    "onetrans_targeted_followup_multiseed": "OneTrans 定向 follow-up（4 seeds）",
    "onetrans_refiner_multiseed": "OneTrans refiner 多 seed",
    "onetrans_residual_multiseed": "OneTrans residual 多 seed",
    "onetrans_peft_multiseed": "OneTrans PEFT 多 seed",
    "onetrans_peft_pair_multiseed": "OneTrans PEFT pair 多 seed",
    "onetrans_warmstart": "OneTrans warmstart",
    "onetrans_loss_screen": "OneTrans loss 筛查",
    "onetrans_pattern_pair": "OneTrans pattern pair 筛查",
    "onetrans_peft_pair": "OneTrans PEFT pair 筛查",
    "onetrans_peft_screen": "OneTrans PEFT 筛查",
    "onetrans_transition_screen": "OneTrans transition 筛查",
}
TASK_ORDER = {
    "Taobao_Action": ["is_click", "cart", "fav", "buy"],
    "MerRec_Action": ["Like", "Cart", "Offer", "Checkout", "Purchase"],
    "KuaiRand_Video_Action": ["is_click", "is_like", "long_view", "is_follow", "is_comment", "is_forward"],
    "QK_Video_Action": ["click", "like", "follow", "share"],
    "TencentGR_10M_Action": ["is_click", "is_conversion"],
}


def norm_arm(name: str) -> str:
    arm = name
    for token in ("TAAC_", "Taobao_", "MerRec_", "Kuai_", "QKVideo_", "QK_",
                  "OneTransState_", "OneTrans_", "KuaiRand_"):
        if arm.startswith(token):
            arm = arm[len(token):]
            break
    return ARM_ALIAS.get(arm, arm)


def add_record(store, backbone, dataset, study, arm, seed, tasks):
    store[(backbone, dataset, study, arm)][str(seed)] = {
        "tasks": tasks,
        "macro": mean(tasks.values()),
    }


def collect_logs(store):
    for ckpt_dir in sorted(RUNS.glob("checkpoints_*")):
        family = ckpt_dir.name[len("checkpoints_"):]
        if family in EXCLUDED_FAMILIES:
            continue
        backbone = "OneTrans" if family.startswith("onetrans") else "RankMixer"
        for log in sorted(ckpt_dir.rglob("*.log")):
            parsed = parse_log(log)
            if parsed is None:
                continue
            dataset = log.parent.name if log.parent != ckpt_dir else "unknown"
            dataset = DATASET_ALIAS.get(dataset, dataset)
            if dataset not in DATASETS:
                continue
            variant, seed = split_name(log.name.split(".log")[0])
            if seed == "single":
                seed = "s1"
            tasks = {t: v["auc"] for t, v in parsed["metrics"].items()}
            study = STUDY_MERGE.get((dataset, family), family)
            add_record(store, backbone, dataset, study, norm_arm(variant), seed, tasks)


def collect_four_seed(store):
    path = OUTER / "reports/unirank_taobao_rankmixer_four_seed_full.json"
    data = json.loads(path.read_text())
    for seed, arms in data["runs"].items():
        for arm, payload in arms.items():
            if not payload.get("training_finished"):
                continue
            tasks = {t: v["AUC"] for t, v in payload["metrics"].items()}
            add_record(store, "RankMixer", "Taobao_Action", "taobao_four_seed",
                       norm_arm(arm), seed, tasks)

    path = OUTER / "reports/unirank_taobao_raw_rankmixer_four_seed.json"
    data = json.loads(path.read_text())
    for seed, payload in data["runs"].items():
        tasks = {t: v["AUC"] for t, v in payload["metrics"].items()}
        add_record(store, "RankMixer", "Taobao_Action", "taobao_four_seed",
                   "Parallel", seed, tasks)


def collect_json_runs(store, filename, backbone, dataset, study):
    path = OUTER / "reports" / filename
    data = json.loads(path.read_text())
    runs = data.get("runs", {})
    for arm, by_seed in runs.items():
        for seed, payload in by_seed.items():
            metrics = payload.get("metrics", payload)
            tasks = {t: v["AUC"] for t, v in metrics.items() if isinstance(v, dict) and "AUC" in v}
            if not tasks:
                continue
            add_record(store, backbone, dataset, study, norm_arm(arm), seed, tasks)


def collect_depth_channel(store):
    path = OUTER / "reports/rankmixer_depth_channel_multiseed_results.json"
    data = json.loads(path.read_text())
    out = {}
    for key, payload in data.items():
        dataset = DATASET_ALIAS.get(key, key + "_Action")
        per_task = {}
        for task, entry in payload["summary"].items():
            per_task[task] = {
                "control_mean": entry["control_auc"]["mean"],
                "control_std": entry["control_auc"]["std"],
                "delta_mean": entry["auc_delta"]["mean"],
                "delta_std": entry["auc_delta"]["std"],
                "positive_seeds": entry["auc_positive_seeds"],
            }
        out[dataset] = {
            "seeds": payload["per_seed"],
            "macro_delta_mean": payload["macro_auc_delta"]["mean"],
            "macro_delta_std": payload["macro_auc_delta"]["std"],
            "macro_positive_seeds": payload["macro_positive_seeds"],
            "per_task": per_task,
        }
    return out


def find_baseline(arms, backbone, dataset, study):
    override = BASELINE_OVERRIDE.get((backbone, dataset, study))
    if override and override in arms:
        return override
    for suffix in ("_Baseline",):
        for arm in arms:
            if arm.endswith(suffix):
                return arm
    for candidate in ("Parallel", "Sequential", "Baseline", "Independent", "Branch"):
        if candidate in arms:
            return candidate
    return None


def aggregate(store, depth_channel):
    rows = []
    for (backbone, dataset, study, arm), by_seed in store.items():
        seeds = sorted(by_seed)
        tasks = sorted({t for s in by_seed.values() for t in s["tasks"]})
        tasks = [t for t in TASK_ORDER.get(dataset, tasks) if t in tasks] or tasks
        per_task = {}
        for task in tasks:
            values = [by_seed[s]["tasks"][task] for s in seeds if task in by_seed[s]["tasks"]]
            per_task[task] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        macros = [by_seed[s]["macro"] for s in seeds]
        rows.append({
            "backbone": backbone,
            "dataset": dataset,
            "study": study,
            "arm": arm,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "macro_mean": mean(macros),
            "macro_std": stdev(macros) if len(macros) > 1 else 0.0,
            "macro_by_seed": {s: by_seed[s]["macro"] for s in seeds},
            "per_task": per_task,
        })

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["backbone"], row["dataset"], row["study"])].append(row)
    for key, group in grouped.items():
        backbone, dataset, study = key
        baseline = find_baseline([r["arm"] for r in group], backbone, dataset, study)
        cross = CROSS_STUDY_BASELINE.get((backbone, dataset, study))
        if cross is not None:
            src_study, src_arm = cross
            base_row = next((r for r in rows if r["backbone"] == backbone
                             and r["dataset"] == dataset and r["study"] == src_study
                             and r["arm"] == src_arm), None)
            if base_row is not None:
                baseline = f"{src_arm}（{src_study}）"
        else:
            base_row = next((r for r in group if r["arm"] == baseline), None)
        for row in group:
            row["baseline"] = baseline
            if base_row is None or row is base_row:
                row["delta_macro_mean"] = None
                row["delta_macro_std"] = None
                row["wins"] = None
                row["paired_p"] = None
                continue
            common = sorted(set(row["seeds"]) & set(base_row["seeds"]))
            if len(common) < 2:
                row["delta_macro_mean"] = None
                row["delta_macro_std"] = None
                row["wins"] = None
                row["paired_p"] = None
                continue
            deltas = [row["macro_by_seed"][s] - base_row["macro_by_seed"][s] for s in common]
            row["delta_macro_mean"] = mean(deltas)
            row["delta_macro_std"] = stdev(deltas)
            row["wins"] = sum(d > 0 for d in deltas)
            row["paired_seeds"] = len(common)
            if ttest_rel is not None:
                row["paired_p"] = float(ttest_rel(
                    [row["macro_by_seed"][s] for s in common],
                    [base_row["macro_by_seed"][s] for s in common],
                ).pvalue)
            else:
                row["paired_p"] = None
    rows.sort(key=lambda r: (r["backbone"], r["dataset"], r["study"],
                             r["arm"] != (r.get("baseline") or ""), r["arm"]))
    return rows, depth_channel


def fmt(value, digits=6):
    return "-" if value is None else f"{value:.{digits}f}"


def render_md(rows, depth):
    out = []
    for backbone in ("RankMixer", "OneTrans"):
        out.append(f"## {'1' if backbone == 'RankMixer' else '2'}. {backbone} backbone")
        out.append("")
        datasets = [d for d in DATASETS if any(r["backbone"] == backbone and r["dataset"] == d for r in rows)]
        for dataset in datasets:
            out.append(f"### {dataset}")
            out.append("")
            studies = []
            for row in rows:
                if row["backbone"] == backbone and row["dataset"] == dataset and row["study"] not in studies:
                    studies.append(row["study"])

            def study_key(name):
                if name in STUDY_ORDER:
                    return (0, STUDY_ORDER.index(name), name)
                max_n = max(r["n_seeds"] for r in rows
                            if r["backbone"] == backbone and r["dataset"] == dataset
                            and r["study"] == name)
                return (1, -max_n, name)

            studies.sort(key=study_key)
            for study in studies:
                group = [r for r in rows if r["backbone"] == backbone
                         and r["dataset"] == dataset and r["study"] == study]
                label = STUDY_LABEL.get(study, study)
                out.append(f"**{label}** (`{study}`，基线 = `{group[0].get('baseline')}`)")
                out.append("")
                out.append("| 变体 | seeds | Macro AUC (mean ± std) | ΔMacro vs 基线 | wins | p (paired, 2-sided) |")
                out.append("|---|---:|---:|---:|---:|---:|")
                for row in group:
                    if row["wins"] is None:
                        wins_cell = "-"
                    else:
                        wins_cell = "{}/{}".format(row["wins"], row["paired_seeds"])
                    out.append(
                        "| {arm} | {n} | {macro} ± {macro_std} | {delta} ± {delta_std} | {wins} | {p} |".format(
                            arm=row["arm"], n=row["n_seeds"],
                            macro=fmt(row["macro_mean"]), macro_std=fmt(row["macro_std"]),
                            delta=fmt(row["delta_macro_mean"]), delta_std=fmt(row["delta_macro_std"]),
                            wins=wins_cell, p=fmt(row["paired_p"], 4),
                        )
                    )
                out.append("")
                out.append("<details><summary>各任务 AUC (mean ± std)</summary>")
                out.append("")
                tasks = list(group[0]["per_task"].keys())
                header = "| 变体 | " + " | ".join(tasks) + " |"
                out.append(header)
                out.append("|---|" + "|".join(["---:" for _ in tasks]) + "|")
                for row in group:
                    cells = " | ".join(
                        f"{fmt(row['per_task'][t]['mean'])} ± {fmt(row['per_task'][t]['std'])}"
                        if t in row["per_task"] else "-"
                        for t in tasks
                    )
                    out.append(f"| {row['arm']} | {cells} |")
                out.append("")
                out.append("</details>")
                out.append("")
        if backbone == "RankMixer":
            out.append("### Depth channel scale（仅报告 delta，4 seeds）")
            out.append("")
            out.append("| 数据集 | ΔMacro | positive seeds | 备注 |")
            out.append("|---|---:|---:|---|")
            for dataset, payload in depth.items():
                out.append(
                    f"| {dataset} | {fmt(payload['macro_delta_mean'])} ± {fmt(payload['macro_delta_std'])} "
                    f"| {payload['macro_positive_seeds']}/{len(payload['seeds'])} | 静态 depth 参数化 |"
                )
            out.append("")
    return "\n".join(out)


def main():
    store = defaultdict(dict)
    collect_logs(store)
    collect_four_seed(store)
    collect_json_runs(store, "rankmixer_head_sharing_taobao_results.json",
                      "RankMixer", "Taobao_Action", "head_sharing_4seed")
    collect_json_runs(store, "rankmixer_preference_margin_four_seed.json",
                      "RankMixer", "Taobao_Action", "preference_margin_4seed")
    depth = collect_depth_channel(store)
    rows, depth = aggregate(store, depth)

    md = render_md(rows, depth)
    (OUTER / "reports/master_results_tables_20261013.md").write_text(md)
    (OUTER / "reports/master_results_table_20261013.json").write_text(
        json.dumps({"rows": rows, "depth_channel": depth}, indent=2)
    )
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
