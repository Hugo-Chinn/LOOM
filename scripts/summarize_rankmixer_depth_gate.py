#!/usr/bin/env python3
"""Compare monotonic depth-gate runs with exact same-seed controls."""
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    "Taobao": (
        ROOT / "runs_local/rankmixer_branch_gate_ablation/RankMixer_Branch_Seed20262028.stdout.log",
        ROOT / "runs_local/checkpoints_rankmixer_depth_gate/Taobao_Action/RankMixer_Taobao_Branch_DepthGate.log",
    ),
    "KuaiRand": (
        ROOT / "runs_local/rankmixer_multiseed_confirmation/RankMixer_Kuai_DAGMeanPre2_S20262028.stdout.log",
        ROOT / "runs_local/checkpoints_rankmixer_depth_gate/KuaiRand_Video_Action/RankMixer_Kuai_DAGMeanPre2_DepthGate.log",
    ),
}
METRIC = re.compile(r"\[Task: ([^]]+)\]\[Metrics\].*?AUC: ([0-9.]+)")


def test_aucs(path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    if "******** Test evaluation ********" not in text:
        return None
    section = text.rsplit("******** Test evaluation ********", 1)[1]
    return {task: float(auc) for task, auc in METRIC.findall(section)}


def main():
    result = {}
    for dataset, (control_path, depth_path) in PAIRS.items():
        control, depth = test_aucs(control_path), test_aucs(depth_path)
        row = {"control": control, "depth_gate": depth}
        if control and depth and control.keys() == depth.keys():
            row["delta"] = {task: depth[task] - control[task] for task in control}
            row["control_mean_auc"] = sum(control.values()) / len(control)
            row["depth_gate_mean_auc"] = sum(depth.values()) / len(depth)
            row["mean_auc_delta"] = row["depth_gate_mean_auc"] - row["control_mean_auc"]
        result[dataset] = row
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
