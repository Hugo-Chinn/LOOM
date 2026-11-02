#!/usr/bin/env python3
"""Build matched single-seed RankMixer recurrence-conditioning screens."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_recurrence_conditioning"
SEED = 20262028


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    taobao = yaml.safe_load(
        (ROOT / "config_rankmixer_branch_gate_ablation/model_config.yaml").read_text()
    )
    confirmed = yaml.safe_load(
        (ROOT / "config_rankmixer_multiseed_confirmation/model_config.yaml").read_text()
    )
    templates = {
        "Taobao": taobao[f"RankMixer_Branch_Seed{SEED}"],
        "MerRec": confirmed[f"RankMixer_MerRec_DAGGated_S{SEED}"],
        "Kuai": confirmed[f"RankMixer_Kuai_DAGMeanPre2_S{SEED}"],
    }
    variants = {
        "BehaviorZero": {"behavior_embedding_init": "zero"},
        "BehaviorNormal002": {
            "behavior_embedding_init": "small_normal",
            "behavior_embedding_init_std": 0.02,
        },
        "BehaviorScaled01": {
            "behavior_embedding_init": "scaled_default",
            "behavior_embedding_init_std": 0.1,
        },
        "DepthChannelScale": {
            "loop_depth_conditioning": "channel_scale",
        },
        "DepthSinusoidal": {
            "loop_depth_conditioning": "sinusoidal",
            "depth_conditioning_dim": 16,
        },
        "BehaviorDepthMod": {
            "loop_depth_conditioning": "behavior_sinusoidal",
            "depth_conditioning_dim": 16,
        },
        "BehaviorAdapter16": {
            "behavior_adapter_dim": 16,
        },
    }

    config = {"Base": clone(confirmed["Base"])}
    config["Base"]["model_root"] = (
        "./runs_local/checkpoints_rankmixer_recurrence_conditioning/"
    )
    for dataset, template in templates.items():
        for variant, overrides in variants.items():
            expid = f"RankMixer_{dataset}_{variant}_S{SEED}"
            config[expid] = dict(clone(template), **overrides)

    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    dataset_ids = sorted({template["dataset_id"] for template in templates.values()})
    datasets = {dataset_id: local_data[dataset_id] for dataset_id in dataset_ids}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(datasets, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()

