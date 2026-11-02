#!/usr/bin/env python3
"""Build paired linear-vs-branch RankMixer configs for multi-seed validation."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_multiseed"
SEEDS = (20262028, 20262029, 20262030)


def main():
    source = yaml.safe_load((ROOT / "config_rankmixer_optimized/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_rankmixer_optimized/dataset_config.yaml").read_text())
    config = {"Base": dict(source["Base"])}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_rankmixer_multiseed/",
        "hidden_analysis_batches": 1,
        "hidden_analysis_samples": 8192,
        "hidden_analysis_projection_dim": 64,
        "hidden_analysis_dir": "./runs_local/hidden_analysis_multiseed",
    })
    linear = yaml.safe_load(yaml.safe_dump(source["RankMixer_Gated_Linear_Deep"]))
    # Use the previous best non-gated linear Loop+Deep as the paired control.
    linear["use_loop_gate"] = False
    branch = yaml.safe_load(yaml.safe_dump(source["RankMixer_Branch_Gated"]))
    for seed in SEEDS:
        linear_id = f"RankMixer_LinearDeep_Seed{seed}"
        branch_id = f"RankMixer_BranchGated_Seed{seed}"
        config[linear_id] = dict(linear, seed=seed)
        config[branch_id] = dict(branch, seed=seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
