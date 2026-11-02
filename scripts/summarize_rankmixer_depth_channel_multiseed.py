#!/usr/bin/env python3
"""Summarize paired four-seed depth-channel RankMixer experiments."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT.parent / "reports"
SEEDS = (20262027, 20262028, 20262029, 20262030)
METRIC = re.compile(
    r"\[Task: ([^]]+)\]\[Metrics\].*?logloss: ([0-9.]+) - AUC: ([0-9.]+)"
)


CONTROL_PATHS = {
    "Taobao": {
        20262027: ROOT / "runs_local/checkpoints_rankmixer_head_sharing/Taobao_Action/RankMixer_Branch_Independent_Seed20262027.log",
        20262028: ROOT / "runs_local/rankmixer_branch_gate_ablation/RankMixer_Branch_Seed20262028.stdout.log",
        20262029: ROOT / "runs_local/checkpoints_rankmixer_head_sharing/Taobao_Action/RankMixer_Branch_Independent_Seed20262029.log",
        20262030: ROOT / "runs_local/checkpoints_rankmixer_head_sharing/Taobao_Action/RankMixer_Branch_Independent_Seed20262030.log",
    },
    "MerRec": {
        20262027: ROOT / "runs_local/checkpoints_rankmixer_hierarchical/MerRec_Action/RankMixer_MerRec_DAG_Gated.log",
        20262028: ROOT / "runs_local/rankmixer_multiseed_confirmation/RankMixer_MerRec_DAGGated_S20262028.stdout.log",
        20262029: ROOT / "runs_local/rankmixer_multiseed_confirmation/RankMixer_MerRec_DAGGated_S20262029.stdout.log",
        20262030: ROOT / "runs_local/rankmixer_multiseed_confirmation/RankMixer_MerRec_DAGGated_S20262030.stdout.log",
    },
    "KuaiRand": {
        20262027: ROOT / "runs_local/checkpoints_rankmixer_hierarchical_phase2/KuaiRand_Video_Action/RankMixer_KuaiRand_DAG_Gated_Mean_Pre2.log",
        20262028: ROOT / "runs_local/rankmixer_multiseed_confirmation/RankMixer_Kuai_DAGMeanPre2_S20262028.stdout.log",
        20262029: ROOT / "runs_local/rankmixer_multiseed_confirmation/RankMixer_Kuai_DAGMeanPre2_S20262029.stdout.log",
        20262030: ROOT / "runs_local/rankmixer_multiseed_confirmation/RankMixer_Kuai_DAGMeanPre2_S20262030.stdout.log",
    },
}


def variant_path(dataset, seed):
    short = "Kuai" if dataset == "KuaiRand" else dataset
    if seed == 20262028:
        return ROOT / (
            "runs_local/rankmixer_recurrence_conditioning/"
            f"RankMixer_{short}_DepthChannelScale_S{seed}.stdout.log"
        )
    return ROOT / (
        "runs_local/rankmixer_depth_channel_multiseed/"
        f"RankMixer_{short}_DepthChannelScale_S{seed}.stdout.log"
    )


def read_metrics(path):
    text = path.read_text(errors="ignore")
    section = text.rsplit("******** Test evaluation ********", 1)[1]
    return {
        task: {"logloss": float(logloss), "auc": float(auc)}
        for task, logloss, auc in METRIC.findall(section)
    }


def mean_std(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
    }


def main():
    output = {}
    for dataset in ("Taobao", "MerRec", "KuaiRand"):
        per_seed = {}
        for seed in SEEDS:
            control = read_metrics(CONTROL_PATHS[dataset][seed])
            variant = read_metrics(variant_path(dataset, seed))
            tasks = list(control)
            per_seed[str(seed)] = {
                "control": control,
                "depth_channel": variant,
                "auc_delta": {
                    task: variant[task]["auc"] - control[task]["auc"]
                    for task in tasks
                },
                "logloss_delta": {
                    task: variant[task]["logloss"] - control[task]["logloss"]
                    for task in tasks
                },
                "macro_auc_delta": statistics.mean(
                    variant[task]["auc"] - control[task]["auc"]
                    for task in tasks
                ),
            }
        tasks = list(per_seed[str(SEEDS[0])]["control"])
        summary = {}
        for task in tasks:
            control_auc = [per_seed[str(seed)]["control"][task]["auc"] for seed in SEEDS]
            variant_auc = [per_seed[str(seed)]["depth_channel"][task]["auc"] for seed in SEEDS]
            auc_delta = [per_seed[str(seed)]["auc_delta"][task] for seed in SEEDS]
            logloss_delta = [per_seed[str(seed)]["logloss_delta"][task] for seed in SEEDS]
            summary[task] = {
                "control_auc": mean_std(control_auc),
                "depth_channel_auc": mean_std(variant_auc),
                "auc_delta": mean_std(auc_delta),
                "auc_positive_seeds": sum(value > 0 for value in auc_delta),
                "auc_deltas": auc_delta,
                "logloss_delta": mean_std(logloss_delta),
                "logloss_improved_seeds": sum(value < 0 for value in logloss_delta),
            }
        macro_delta = [per_seed[str(seed)]["macro_auc_delta"] for seed in SEEDS]
        output[dataset] = {
            "per_seed": per_seed,
            "summary": summary,
            "macro_auc_delta": mean_std(macro_delta),
            "macro_positive_seeds": sum(value > 0 for value in macro_delta),
            "macro_deltas": macro_delta,
        }
    path = REPORT_ROOT / "rankmixer_depth_channel_multiseed_results.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

