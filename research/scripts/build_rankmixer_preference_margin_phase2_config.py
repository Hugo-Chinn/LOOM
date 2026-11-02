#!/usr/bin/env python3
"""Build phase-2 low-weight preference-margin experiments."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_preference_margin_phase2"
SEEDS = (20262027, 20262028, 20262029, 20262030)
VARIANTS = {
    "ProgressiveW0005": {"preference_margin_weight": 0.005,
                          "preference_margin_target": 0.2},
    "ProgressiveW001": {"preference_margin_weight": 0.01,
                         "preference_margin_target": 0.2},
    "ProgressiveW002": {"preference_margin_weight": 0.02,
                         "preference_margin_target": 0.2},
    "ProgressiveW005": {"preference_margin_weight": 0.05,
                         "preference_margin_target": 0.2},
    "ZeroMarginBPRW002": {"preference_margin_weight": 0.02,
                           "preference_margin_target": 0.0},
}


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    source = yaml.safe_load(
        (ROOT / "config_rankmixer_preference_margin/model_config.yaml").read_text()
    )
    dataset = yaml.safe_load(
        (ROOT / "config_rankmixer_preference_margin/dataset_config.yaml").read_text()
    )
    config = {"Base": clone(source["Base"])}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_rankmixer_preference_margin_phase2/",
        "keep_checkpoint": False,
    })
    for seed in SEEDS:
        template = clone(source[f"RankMixer_Branch_ProgressiveMargin_Seed{seed}"])
        template["preference_margin_mode"] = "progressive"
        template["preference_margin_start"] = 0.0
        for name, updates in VARIANTS.items():
            item = clone(template)
            item.update(updates)
            config[f"RankMixer_Branch_{name}_Seed{seed}"] = item

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
