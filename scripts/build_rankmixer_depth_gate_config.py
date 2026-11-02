#!/usr/bin/env python3
"""Build Taobao and KuaiRand ablations for monotonic depth-controlled gates."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_depth_gate"


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    taobao = yaml.safe_load(
        (ROOT / "config_rankmixer_branch_gate_ablation/model_config.yaml").read_text()
    )
    confirmation = yaml.safe_load(
        (ROOT / "config_rankmixer_multiseed_confirmation/model_config.yaml").read_text()
    )
    templates = {
        # Best same-seed Taobao branch control uses no per-stage residual gate.
        "Taobao_Branch": taobao["RankMixer_Branch_Seed20262028"],
        # Best confirmed KuaiRand Loop uses per-stage gates and two pre-loops.
        "Kuai_DAGMeanPre2": confirmation["RankMixer_Kuai_DAGMeanPre2_S20262028"],
    }

    config = {"Base": clone(confirmation["Base"])}
    config["Base"]["model_root"] = "./runs_local/checkpoints_rankmixer_depth_gate/"
    for short, template in templates.items():
        control = clone(template)
        control["use_depth_loop_gate"] = False
        depth_gate = clone(template)
        depth_gate.update({
            "use_loop_gate": False,
            "use_depth_loop_gate": True,
            "depth_gate_init": 0.5,
            "depth_gate_slope_init": 0.1,
            "use_iteration_depth_embedding": False,
        })
        config[f"RankMixer_{short}_Control"] = control
        config[f"RankMixer_{short}_DepthGate"] = depth_gate

    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    dataset_ids = sorted({item["dataset_id"] for item in templates.values()})
    datasets = {dataset_id: local_data[dataset_id] for dataset_id in dataset_ids}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(datasets, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
