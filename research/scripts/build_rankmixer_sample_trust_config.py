#!/usr/bin/env python3
"""Build matched RankMixer sample-wise trust-gate screens."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_sample_trust"
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
        "Stage": {
            "sample_trust_gate": "stage",
            "sample_trust_hidden_dim": 16,
            "sample_trust_init": 0.9,
        },
        "Shared": {
            "sample_trust_gate": "shared",
            "sample_trust_hidden_dim": 16,
            "sample_trust_init": 0.9,
        },
        "StageKeep": {
            "sample_trust_gate": "stage",
            "sample_trust_hidden_dim": 16,
            "sample_trust_init": 0.9,
            "sample_trust_reg_weight": 1.0e-4,
        },
    }
    config = {"Base": clone(confirmed["Base"])}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_rankmixer_sample_trust/",
        "keep_checkpoint": True,
    })
    for dataset, template in templates.items():
        for variant, overrides in variants.items():
            config[f"RankMixer_{dataset}_SampleTrust{variant}_S{SEED}"] = dict(
                clone(template), **overrides
            )

    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    dataset_ids = sorted({template["dataset_id"] for template in templates.values()})
    datasets = {dataset_id: local_data[dataset_id] for dataset_id in dataset_ids}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(datasets, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()

