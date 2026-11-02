#!/usr/bin/env python3
"""Build topology-retention and exact-pattern pair supervision ablations."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_pattern_pair"


def main():
    source = yaml.safe_load((ROOT / "config_onetrans_peft_screen/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_onetrans_peft_screen/dataset_config.yaml").read_text())
    base = dict(source.get("Base", {}))
    base["model_root"] = "./runs_local/checkpoints_onetrans_pattern_pair/"
    config = {"Base": base}

    template = yaml.safe_load(yaml.safe_dump(
        source["OneTrans_Peft_QVLoRA_R4_Seed20262027"]))
    template.update({
        "seed": 20262027,
        "enable_torch_compile": False,
        "pattern_pair_loss_weight": 0.0,
        "pattern_no_regret_weight": 0.0,
        "topology_retention_weight": 0.0,
    })
    variants = {
        "OneTrans_Pattern_Control_Seed20262027": {},
        "OneTrans_Pattern_Retention_Seed20262027": {
            "topology_retention_weight": 1.0,
        },
        "OneTrans_Pattern_Pair05_Seed20262027": {
            "pattern_pair_loss_weight": 0.05,
        },
        "OneTrans_Pattern_Pair05NoRegret05_Seed20262027": {
            "pattern_pair_loss_weight": 0.05,
            "pattern_no_regret_weight": 0.05,
        },
        "OneTrans_Pattern_Pair05Retention_Seed20262027": {
            "pattern_pair_loss_weight": 0.05,
            "topology_retention_weight": 1.0,
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
