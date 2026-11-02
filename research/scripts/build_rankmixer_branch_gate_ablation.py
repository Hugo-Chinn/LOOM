#!/usr/bin/env python3
"""Build paired branch-only versus branch+gate multi-seed configs."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_branch_gate_ablation"
SEEDS = (20262028, 20262029, 20262030)


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    source = yaml.safe_load((ROOT / "config_rankmixer_optimized/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_rankmixer_optimized/dataset_config.yaml").read_text())
    config = {"Base": clone(source["Base"])}
    config["Base"]["model_root"] = "./runs_local/checkpoints_branch_gate_ablation/"
    ungated = clone(source["RankMixer_Branch"])
    gated = clone(source["RankMixer_Branch_Gated"])
    for seed in SEEDS:
        config[f"RankMixer_Branch_Seed{seed}"] = dict(ungated, seed=seed)
        config[f"RankMixer_BranchGated_Seed{seed}"] = dict(gated, seed=seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
