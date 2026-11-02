#!/usr/bin/env python3
"""Build loss-only ablations for warm-started OneTrans QV-LoRA refinement."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_loss_screen"


def main():
    source = yaml.safe_load((ROOT / "config_onetrans_warmstart/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_onetrans_warmstart/dataset_config.yaml").read_text())
    base = dict(source.get("Base", {}))
    base["model_root"] = "./runs_local/checkpoints_onetrans_loss_screen/"
    config = {"Base": base}

    template = yaml.safe_load(yaml.safe_dump(
        source["OneTrans_QVLoRA_Conservative_LR1e5_Seed20262027"]))
    template.update({
        "seed": 20262027,
        "keep_checkpoint": False,
        "enable_torch_compile": False,
    })
    balanced = [0.364536, 1.004543, 1.146432, 1.484489]
    variants = {
        "OneTrans_Loss_Balanced_Seed20262027": {
            "task_loss_weights": balanced,
        },
        "OneTrans_Loss_Distill1_Seed20262027": {
            "distillation_loss_weight": 1.0,
        },
        "OneTrans_Loss_BalancedDistill1_Seed20262027": {
            "task_loss_weights": balanced,
            "distillation_loss_weight": 1.0,
        },
        "OneTrans_Loss_BalancedDistill5State1_Seed20262027": {
            "task_loss_weights": balanced,
            "distillation_loss_weight": 5.0,
            "state_preservation_weight": 1.0,
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
