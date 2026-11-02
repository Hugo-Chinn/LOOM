#!/usr/bin/env python3
"""Build the TencentGR six-seed confirmation config for the two loop finalists.

Reuses the frozen templates from `config_rankmixer_multiseed_confirmation`
(3 seeds: 20262028-20262030) and adds three fresh base seeds so that
`TAAC_Loop_GatedDeep_Pre2`, `TAAC_Loop_Deep` and the `TAAC_Sequential` control
have six exactly paired seeds.
"""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "config_rankmixer_multiseed_confirmation"
OUT = ROOT / "config_rankmixer_taac_six_seed"
NEW_SEEDS = (20262031, 20262032, 20262033)
ARMS = ("Sequential", "Loop_Deep", "Loop_GatedDeep_Pre2")
TEMPLATE_SEED = 20262028


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    src = yaml.safe_load((SRC / "model_config.yaml").read_text())
    config = {"Base": clone(src["Base"])}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_rankmixer_taac_six_seed/",
        "num_workers": 4,
    })
    for arm in ARMS:
        template = clone(src[f"RankMixer_TAAC_{arm}_S{TEMPLATE_SEED}"])
        for seed in NEW_SEEDS:
            item = clone(template)
            item["seed"] = seed
            config[f"RankMixer_TAAC_{arm}_S{seed}"] = item

    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    dataset_ids = sorted({item["dataset_id"] for key, item in config.items() if key != "Base"})
    datasets = {dataset_id: local_data[dataset_id] for dataset_id in dataset_ids}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(datasets, sort_keys=False))
    print(OUT, "expids:", len(config) - 1)


if __name__ == "__main__":
    main()
