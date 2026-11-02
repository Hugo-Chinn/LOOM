#!/usr/bin/env python3
"""Build 4-seed RankMixer preference-margin experiments on Taobao."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_preference_margin"
SEEDS = (20262027, 20262028, 20262029, 20262030)


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    source_models = yaml.safe_load(
        (ROOT / "config_rankmixer_head_sharing/model_config.yaml").read_text()
    )
    source_data = yaml.safe_load(
        (ROOT / "config_rankmixer_head_sharing/dataset_config.yaml").read_text()
    )
    config = {"Base": clone(source_models["Base"])}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_rankmixer_preference_margin/",
        "hidden_analysis_batches": 0,
        "keep_checkpoint": False,
        "group_batch_col": "user_id",
    })
    variants = {
        "Baseline": {
            "preference_margin_mode": "none",
            "preference_margin_weight": 0.0,
        },
        "FixedMargin": {
            "preference_margin_mode": "fixed",
            "preference_margin_start": 0.0,
            "preference_margin_target": 0.2,
        },
        "ProgressiveMargin": {
            "preference_margin_mode": "progressive",
            "preference_margin_start": 0.0,
            "preference_margin_target": 0.2,
        },
        "ReverseProgressiveMargin": {
            "preference_margin_mode": "reverse_progressive",
            "preference_margin_start": 0.0,
            "preference_margin_target": 0.2,
        },
    }
    for seed in SEEDS:
        template_id = f"RankMixer_Branch_Independent_Seed{seed}"
        template = clone(source_models[template_id])
        template.update({
            "preference_positive_label": "buy",
            "preference_negative_mode": "click_only",
        })
        for suffix, updates in variants.items():
            item = clone(template)
            item.update(updates)
            if suffix != "Baseline":
                item["preference_margin_weight"] = 0.1
            config[f"RankMixer_Branch_{suffix}_Seed{seed}"] = item

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (OUT / "dataset_config.yaml").write_text(
        yaml.safe_dump({"Taobao_Action": source_data["Taobao_Action"]}, sort_keys=False)
    )
    print(OUT)


if __name__ == "__main__":
    main()
