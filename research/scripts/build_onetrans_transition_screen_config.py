#!/usr/bin/env python3
"""Build loop-aligned transition-supervision ablations for OneTrans."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_transition_screen"


def main():
    source = yaml.safe_load((ROOT / "config_onetrans_warmstart/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_onetrans_warmstart/dataset_config.yaml").read_text())
    base = dict(source.get("Base", {}))
    base["model_root"] = "./runs_local/checkpoints_onetrans_transition_screen/"
    config = {"Base": base}

    template = yaml.safe_load(yaml.safe_dump(
        source["OneTrans_QVLoRA_Conservative_LR1e5_Seed20262027"]))
    template.update({
        "seed": 20262027,
        "keep_checkpoint": False,
        "enable_torch_compile": False,
        "prediction_mode": "residual_logit",
        "residual_logit_scale": 1.0,
        "task_loss_weights": [1.0, 1.0, 1.0, 1.0],
        "distillation_loss_weight": 0.0,
        "state_preservation_weight": 0.0,
    })
    variants = {
        "OneTrans_ResidualLogit_Seed20262027": {},
        "OneTrans_ResidualLogit_Direction01_Seed20262027": {
            "transition_loss_weight": 0.1,
            "transition_margin": 0.0,
        },
        "OneTrans_ResidualLogit_Risk01_Seed20262027": {
            "risk_set_loss_weight": 0.1,
        },
        "OneTrans_ResidualLogit_Direction05Risk05_Seed20262027": {
            "transition_loss_weight": 0.05,
            "transition_margin": 0.0,
            "risk_set_loss_weight": 0.05,
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
