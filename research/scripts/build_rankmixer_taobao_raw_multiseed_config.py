#!/usr/bin/env python3
"""Build matched raw/Parallel RankMixer Taobao configs for missing seeds."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_taobao_raw_multiseed"
SEEDS = (20262028, 20262029, 20262030)


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    local_models = yaml.safe_load((ROOT / "config_local/model_config.yaml").read_text())
    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    template = local_models["RankMixer_Taobao_P1_Parallel"]
    config = {"Base": clone(local_models["Base"])}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_rankmixer_taobao_raw_multiseed/",
        "num_workers": 4,
    })
    for seed in SEEDS:
        item = clone(template)
        item["seed"] = seed
        config[f"RankMixer_Taobao_Raw_Seed{seed}"] = item
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump({
        "Taobao_Action": local_data["Taobao_Action"]
    }, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
