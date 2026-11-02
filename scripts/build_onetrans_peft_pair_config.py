#!/usr/bin/env python3
"""Build explicit matched-pair, PEFT-only OneTrans supervision ablations."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_peft_pair"


def main():
    source = yaml.safe_load((ROOT / "config_onetrans_peft_screen/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_onetrans_peft_screen/dataset_config.yaml").read_text())
    base = dict(source.get("Base", {}))
    base["model_root"] = "./runs_local/checkpoints_onetrans_peft_pair/"
    config = {"Base": base}
    template = yaml.safe_load(yaml.safe_dump(
        source["OneTrans_Peft_QVLoRA_R4_Seed20262027"]))
    template.update({
        "seed": 20262027,
        "enable_torch_compile": False,
        "explicit_pair_loss_weight": 0.0,
        "explicit_pair_score_bins": 10,
        "explicit_pair_stage_weights": [1.0, 1.0, 1.0, 1.0],
    })
    variants = {
        "OneTrans_PeftPair_Control_Seed20262027": {},
        "OneTrans_PeftPair_Exact_W005_Seed20262027": {
            "explicit_pair_loss_weight": 0.005,
            "explicit_pair_score_bins": 1,
        },
        "OneTrans_PeftPair_Score10_W005_Seed20262027": {
            "explicit_pair_loss_weight": 0.005,
            "explicit_pair_score_bins": 10,
        },
        "OneTrans_PeftPair_Score10_W001_Seed20262027": {
            "explicit_pair_loss_weight": 0.001,
            "explicit_pair_score_bins": 10,
        },
        "OneTrans_PeftPair_Score10_StageWeighted_Seed20262027": {
            "explicit_pair_loss_weight": 0.005,
            "explicit_pair_score_bins": 10,
            "explicit_pair_stage_weights": [0.25, 1.0, 1.0, 0.5],
        },
    }
    for expid, overrides in variants.items():
        config[expid] = dict(template, **overrides)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
