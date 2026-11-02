#!/usr/bin/env python3
"""Build paired multi-seed confirmation for the safe KuaiRand routing finalists."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_safe_routing_multiseed"
SEEDS = (20262028, 20262029, 20262030)


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    screen = yaml.safe_load(
        (ROOT / "config_rankmixer_safe_routing/model_config.yaml").read_text()
    )
    data = yaml.safe_load(
        (ROOT / "config_rankmixer_safe_routing/dataset_config.yaml").read_text()
    )
    templates = {
        "AdapterOnly": screen["RankMixer_Kuai_AdapterOnly_S20262028"],
        "AdapterResidualBypass": screen[
            "RankMixer_Kuai_AdapterResidualBypass_S20262028"
        ],
    }

    config = {"Base": clone(screen["Base"])}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_rankmixer_safe_routing_multiseed/",
        "keep_checkpoint": False,
    })
    for seed in SEEDS:
        for name, template in templates.items():
            item = clone(template)
            item["seed"] = seed
            config[f"RankMixer_Kuai_{name}_S{seed}"] = item

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump({
        "KuaiRand_Video_Action": data["KuaiRand_Video_Action"]
    }, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
