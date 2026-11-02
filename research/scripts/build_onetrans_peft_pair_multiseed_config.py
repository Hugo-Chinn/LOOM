#!/usr/bin/env python3
"""Build multi-seed explicit PEFT-only pair supervision configs."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_peft_pair_multiseed"
SEEDS = (20262028, 20262029, 20262030)


def main():
    source = yaml.safe_load((ROOT / "config_onetrans_peft_pair/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_onetrans_peft_pair/dataset_config.yaml").read_text())
    base = dict(source.get("Base", {}))
    base["model_root"] = "./runs_local/checkpoints_onetrans_peft_pair_multiseed/"
    config = {"Base": base}

    exact = yaml.safe_load(yaml.safe_dump(
        source["OneTrans_PeftPair_Exact_W005_Seed20262027"]))
    stage_weighted = yaml.safe_load(yaml.safe_dump(
        source["OneTrans_PeftPair_Score10_StageWeighted_Seed20262027"]))
    for seed in SEEDS:
        config[f"OneTrans_PeftPair_Exact_W005_Seed{seed}"] = dict(
            exact, seed=seed)
        config[f"OneTrans_PeftPair_Score10_StageWeighted_Seed{seed}"] = dict(
            stage_weighted, seed=seed)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
