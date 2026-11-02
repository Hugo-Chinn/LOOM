#!/usr/bin/env python3
"""Build seed-20262027 OneTrans branch-refiner PEFT screening configs."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_peft_screen"


def main():
    source = yaml.safe_load((ROOT / "config_local/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    base = dict(source.get("Base", {}))
    base["model_root"] = "./runs_local/checkpoints_onetrans_peft_screen/"
    config = {"Base": base}
    template = yaml.safe_load(yaml.safe_dump(
        source["OneTrans_Taobao_P3_BaseRefiner_Branch"]))
    template.update({
        "seed": 20262027,
        "enable_loop_diagnostics": False,
        "enable_torch_compile": True,
    })
    variants = {
        "OneTrans_Peft_FiLM_Seed20262027": {
            "peft_type": "film",
        },
        "OneTrans_Peft_Adapter16_Seed20262027": {
            "peft_type": "adapter",
            "adapter_dim": 16,
        },
        "OneTrans_Peft_QLoRA_R4_Seed20262027": {
            "peft_type": "lora_q",
            "peft_rank": 4,
            "peft_alpha": 4.0,
            "peft_dropout": 0.0,
        },
        "OneTrans_Peft_QVLoRA_R4_Seed20262027": {
            "peft_type": "lora_qv",
            "peft_rank": 4,
            "peft_alpha": 4.0,
            "peft_dropout": 0.0,
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
