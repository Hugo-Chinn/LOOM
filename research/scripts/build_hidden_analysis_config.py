#!/usr/bin/env python3
"""Build representative baseline/loop configs for trained hidden-state analysis."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "config_hidden_analysis"


def clean_copy(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    official = yaml.safe_load((ROOT / "config/model_config.yaml").read_text())
    local = yaml.safe_load((ROOT / "config_local/model_config.yaml").read_text())
    loops = yaml.safe_load((ROOT / "config_loop_top_backbones/model_config.yaml").read_text())
    optimized = yaml.safe_load(
        (ROOT / "config_rankmixer_optimized/model_config.yaml").read_text()
    )
    datasets = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())

    configs = {
        "Base": clean_copy(official["Base"]),
        "Hidden_RankMixer_Baseline": clean_copy(local["RankMixer_Taobao_P1_Parallel"]),
        "Hidden_RankMixer_Loop": clean_copy(local["RankMixer_Taobao_P1_Looped_DeepSupervision"]),
        "Hidden_OneTrans_Baseline": clean_copy(official["OneTrans_Taobao_Action"]),
        "Hidden_OneTrans_Loop": clean_copy(local["OneTrans_Taobao_P1_Looped_DeepSupervision"]),
        "Hidden_EST_Baseline": clean_copy(official["EST_Taobao_Action"]),
        "Hidden_EST_Loop": clean_copy(loops["EST_Taobao_Looped_Deep"]),
        "Hidden_HeMix_Baseline": clean_copy(official["HeMix_Taobao_Action"]),
        "Hidden_HeMix_Loop": clean_copy(loops["HeMix_Taobao_Looped_Deep"]),
        "Hidden_UltraHSTU_Baseline": clean_copy(official["UltraHSTU_Taobao_Action"]),
        "Hidden_UltraHSTU_Loop": clean_copy(loops["UltraHSTU_Taobao_Looped_Deep"]),
        "Hidden_RankMixer_Branch": clean_copy(optimized["RankMixer_Branch"]),
    }
    configs["Base"]["model_root"] = "./runs_local/checkpoints_hidden_analysis/"
    configs["Base"]["hidden_analysis_batches"] = 1
    configs["Base"]["hidden_analysis_samples"] = 8192
    configs["Base"]["hidden_analysis_projection_dim"] = 64
    configs["Base"]["hidden_analysis_dir"] = "./runs_local/hidden_analysis"

    # UltraHSTU uses flex_attention and remains in eager mode on this runtime.
    configs["Hidden_UltraHSTU_Baseline"]["enable_torch_compile"] = False
    configs["Hidden_UltraHSTU_Loop"]["enable_torch_compile"] = False

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "model_config.yaml").write_text(yaml.safe_dump(configs, sort_keys=False))
    (OUTPUT / "dataset_config.yaml").write_text(
        yaml.safe_dump({"Taobao_Action": datasets["Taobao_Action"]}, sort_keys=False)
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
