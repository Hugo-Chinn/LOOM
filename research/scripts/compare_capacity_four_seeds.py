#!/usr/bin/env python3
"""Four-seed (20262027-20262030) comparison of the widening experiments.

Candidate/control pairs per dataset; extra seeds (20262031+) are reported
separately as a robustness check when available.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_all_local_experiments import parse_log, split_name  # noqa: E402
from scipy.stats import ttest_rel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/home/HugoChinn/LoopMT/reports/rankmixer_capacity_four_seed_results.json")
FOUR = {"20262027", "20262028", "20262029", "20262030"}

FAMILIES = [
    "rankmixer_loop_capacity", "rankmixer_capacity_general", "rankmixer_capacity_taobao",
    "rankmixer_multiseed_confirmation", "branch_gate_ablation", "rankmixer_bypass_rare",
    "rankmixer_task_weight_rare", "rankmixer_loop_depth",
]

PAIRS = [
    ("KuaiRand_Video_Action", "Kuai_Wide4_WeightRare4", "Kuai_DAGMeanPre2", "加宽+加权 vs DAG+Mean+Pre2"),
    ("KuaiRand_Video_Action", "Kuai_Wide4", "Kuai_DAGMeanPre2", "只加宽 vs DAG+Mean+Pre2"),
    ("KuaiRand_Video_Action", "Kuai_Wide4_WeightRare4", "Kuai_Parallel", "加宽+加权 vs Parallel"),
    ("KuaiRand_Video_Action", "Kuai_Depth2_WeightRare4", "Kuai_DAGMeanPre2", "加深2层+加权 vs DAG+Mean+Pre2"),
    ("KuaiRand_Video_Action", "Kuai_Depth2", "Kuai_DAGMeanPre2", "只加深2层 vs DAG+Mean+Pre2"),
    ("KuaiRand_Video_Action", "Kuai_Wide8_WeightRare4", "Kuai_Wide4_WeightRare4", "宽度8 vs 宽度4（都加权）"),
    ("TencentGR_10M_Action", "TAAC_Wide4_BypassConv", "TAAC_Loop_GatedDeep_Pre2", "加宽+旁路 vs Loop_GatedDeep_Pre2"),
    ("TencentGR_10M_Action", "TAAC_Wide4_BypassConv", "TAAC_Parallel", "加宽+旁路 vs Parallel"),
    ("TencentGR_10M_Action", "TAAC_BypassRare", "TAAC_Loop_GatedDeep_Pre2", "只旁路 vs Loop_GatedDeep_Pre2"),
    ("Taobao_Action", "Wide4", "Branch", "只加宽 vs Branch"),
    ("Taobao_Action", "Wide4_WeightDeep2", "Branch", "加宽+加权 vs Branch"),
    ("MerRec_Action", "MerRec_Wide4", "MerRec_DAGGated", "只加宽 vs DAG+Gate"),
    ("MerRec_Action", "MerRec_Wide4_WeightLow2", "MerRec_DAGGated", "加宽+加权 vs DAG+Gate"),
    ("QK_Video_Action", "QK_Wide4", "QK_DAGGated", "只加宽 vs DAG+Gate"),
    ("QK_Video_Action", "QK_Wide4", "QK_Parallel", "只加宽 vs Parallel"),
]


def collect():
    table = defaultdict(lambda: defaultdict(dict))
    for family in FAMILIES:
        for log in sorted((ROOT / "runs_local" / f"checkpoints_{family}").rglob("*.log")):
            parsed = parse_log(log)
            if parsed is None:
                continue
            dataset = log.parent.name
            variant, seed = split_name(log.name.split(".log")[0])
            # capacity_taobao names the control "Taobao_Branch"; branch_gate_ablation
            # names it "Branch". Normalise so the same arm merges across families.
            if variant.startswith("Taobao_"):
                variant = variant[len("Taobao_"):]
            tasks = {t: v["auc"] for t, v in parsed["metrics"].items()}
            table[dataset][variant][seed] = {
                "tasks": tasks,
                "macro": mean(tasks.values()),
            }
    return table


def stats(cand, ctrl, seeds):
    common = sorted(set(cand) & set(ctrl) & seeds)
    if len(common) < 2:
        return None
    deltas = [cand[s]["macro"] - ctrl[s]["macro"] for s in common]
    p = float(ttest_rel([cand[s]["macro"] for s in common],
                        [ctrl[s]["macro"] for s in common]).pvalue)
    per_task = {}
    for task in sorted(cand[common[0]]["tasks"]):
        dd = [cand[s]["tasks"][task] - ctrl[s]["tasks"][task] for s in common]
        per_task[task] = {
            "control": mean(ctrl[s]["tasks"][task] for s in common),
            "candidate": mean(cand[s]["tasks"][task] for s in common),
            "delta": mean(dd),
            "wins": sum(d > 0 for d in dd),
        }
    return {
        "seeds": common,
        "n": len(common),
        "candidate_macro": mean(cand[s]["macro"] for s in common),
        "candidate_macro_std": stdev([cand[s]["macro"] for s in common]),
        "control_macro": mean(ctrl[s]["macro"] for s in common),
        "delta_macro_mean": mean(deltas),
        "delta_macro_std": stdev(deltas),
        "wins": sum(d > 0 for d in deltas),
        "paired_t_p_two_sided": p,
        "per_task": per_task,
    }


def main():
    table = collect()
    report = []
    for dataset, cand_name, ctrl_name, label in PAIRS:
        cand = table[dataset].get(cand_name, {})
        ctrl = table[dataset].get(ctrl_name, {})
        if not cand or not ctrl:
            print(f"skip {dataset} {cand_name} vs {ctrl_name}: missing")
            continue
        entry = {
            "dataset": dataset,
            "candidate": cand_name,
            "control": ctrl_name,
            "label": label,
            "four_seed": stats(cand, ctrl, FOUR),
            "all_seeds": stats(cand, ctrl, set(cand) | set(ctrl)),
            "available_seeds": {
                "candidate": sorted(cand),
                "control": sorted(ctrl),
            },
        }
        report.append(entry)
    OUT.write_text(json.dumps(report, indent=2))
    for e in report:
        f = e["four_seed"]
        if f is None:
            print(f"{e['dataset']:22s} {e['label']:34s} four-seed n/a (seeds {e['available_seeds']})")
            continue
        print(f"{e['dataset']:22s} {e['label']:34s} n={f['n']} "
              f"cand={f['candidate_macro']:.6f} ctrl={f['control_macro']:.6f} "
              f"delta={f['delta_macro_mean']:+.6f}±{f['delta_macro_std']:.6f} "
              f"wins={f['wins']}/{f['n']} p={f['paired_t_p_two_sided']:.4f}")


if __name__ == "__main__":
    main()
