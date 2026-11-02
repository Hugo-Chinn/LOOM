#!/usr/bin/env python3
"""Compare iteration-depth runs with their exact same-seed controls."""
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    "KuaiRand": (
        ROOT / "runs_local/rankmixer_multiseed_confirmation/RankMixer_Kuai_DAGMeanPre2_S20262028.stdout.log",
        ROOT / "runs_local/checkpoints_rankmixer_iteration_depth/KuaiRand_Video_Action/RankMixer_Kuai_DAGMeanPre2_IterDepth.log",
    ),
    "TAAC": (
        ROOT / "runs_local/rankmixer_multiseed_confirmation/RankMixer_TAAC_Loop_GatedDeep_Pre2_S20262028.stdout.log",
        ROOT / "runs_local/checkpoints_rankmixer_iteration_depth/TencentGR_10M_Action/RankMixer_TAAC_GatedDeepPre2_IterDepth.log",
    ),
}
METRIC = re.compile(r"\[Task: ([^]]+)\]\[Metrics\].*?AUC: ([0-9.]+)")


def test_aucs(path):
    text = path.read_text(errors="ignore")
    if "******** Test evaluation ********" not in text:
        return None
    section = text.rsplit("******** Test evaluation ********", 1)[1]
    return {task: float(auc) for task, auc in METRIC.findall(section)}


def main():
    result = {}
    for dataset, (control_path, depth_path) in PAIRS.items():
        control = test_aucs(control_path)
        depth = test_aucs(depth_path) if depth_path.exists() else None
        row = {"control": control, "iteration_depth": depth}
        if control and depth and control.keys() == depth.keys():
            row["delta"] = {task: depth[task] - control[task] for task in control}
            row["control_mean_auc"] = sum(control.values()) / len(control)
            row["iteration_depth_mean_auc"] = sum(depth.values()) / len(depth)
            row["mean_auc_delta"] = row["iteration_depth_mean_auc"] - row["control_mean_auc"]
        result[dataset] = row
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
