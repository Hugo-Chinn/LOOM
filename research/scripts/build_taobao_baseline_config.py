#!/usr/bin/env python3
"""Build a local config containing all official Taobao model-zoo baselines."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "config_taobao_all_baselines"


def main():
    official_models = yaml.safe_load((ROOT / "config/model_config.yaml").read_text())
    local_datasets = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())

    model_config = dict(official_models)
    model_config["Base"] = dict(official_models["Base"])
    model_config["Base"]["model_root"] = "./runs_local/checkpoints_all_baselines/"

    # PyTorch 2.6 Inductor has no valid lowering choice for the
    # flex_attention_backward graph used by TokenFormer on this H20 runtime.
    # Eager mode preserves the model/training semantics and avoids a compiler-
    # backend-only failure.
    model_config["TokenFormer_Taobao_Action"] = dict(
        official_models["TokenFormer_Taobao_Action"]
    )
    model_config["TokenFormer_Taobao_Action"]["enable_torch_compile"] = False

    output_models = OUTPUT / "model_config.yaml"
    output_datasets = OUTPUT / "dataset_config.yaml"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_models.write_text(yaml.safe_dump(model_config, sort_keys=False))
    output_datasets.write_text(
        yaml.safe_dump({"Taobao_Action": local_datasets["Taobao_Action"]}, sort_keys=False)
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
