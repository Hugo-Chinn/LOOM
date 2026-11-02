#!/usr/bin/env python3
"""Summarize zero-perturbation routing screens against same-seed controls."""
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRIC = re.compile(r"\[Task: ([^]]+)\]\[Metrics\].*?AUC: ([0-9.]+)")

EXPERIMENTS = {
    "Taobao": {
        "Control": ROOT / "runs_local/rankmixer_branch_gate_ablation/RankMixer_Branch_Seed20262028.stdout.log",
        "DynamicBranch": ROOT / "runs_local/checkpoints_rankmixer_safe_routing/Taobao_Action/RankMixer_Taobao_DynamicBranch_S20262028.log",
    },
    "KuaiRand": {
        "Control": ROOT / "runs_local/rankmixer_multiseed_confirmation/RankMixer_Kuai_DAGMeanPre2_S20262028.stdout.log",
        "AdapterOnly": ROOT / "runs_local/checkpoints_rankmixer_safe_routing/KuaiRand_Video_Action/RankMixer_Kuai_AdapterOnly_S20262028.log",
        "DynamicSoftmax": ROOT / "runs_local/checkpoints_rankmixer_safe_routing/KuaiRand_Video_Action/RankMixer_Kuai_DynamicSoftmax_S20262028.log",
        "DynamicSoftmaxAdapter": ROOT / "runs_local/checkpoints_rankmixer_safe_routing/KuaiRand_Video_Action/RankMixer_Kuai_DynamicSoftmaxAdapter_S20262028.log",
        "ResidualBypass": ROOT / "runs_local/checkpoints_rankmixer_safe_routing/KuaiRand_Video_Action/RankMixer_Kuai_ResidualBypass_S20262028.log",
        "AdapterResidualBypass": ROOT / "runs_local/checkpoints_rankmixer_safe_routing/KuaiRand_Video_Action/RankMixer_Kuai_AdapterResidualBypass_S20262028.log",
    },
}


def parse(path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    if "******** Test evaluation ********" not in text:
        return None
    section = text.rsplit("******** Test evaluation ********", 1)[1]
    values = {task: float(auc) for task, auc in METRIC.findall(section)}
    return values or None


def main():
    summary = {}
    for dataset, experiments in EXPERIMENTS.items():
        parsed = {name: parse(path) for name, path in experiments.items()}
        control = parsed["Control"]
        rows = {}
        for name, values in parsed.items():
            row = {"tasks": values}
            if values:
                row["mean_auc"] = sum(values.values()) / len(values)
            if name != "Control" and control and values and values.keys() == control.keys():
                row["task_delta"] = {
                    task: values[task] - control[task] for task in control
                }
                row["mean_auc_delta"] = (
                    sum(values.values()) / len(values)
                    - sum(control.values()) / len(control)
                )
            rows[name] = row
        summary[dataset] = rows
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
