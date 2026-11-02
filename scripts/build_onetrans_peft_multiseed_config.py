#!/usr/bin/env python3
"""Build paired FiLM-vs-QV-LoRA OneTrans PEFT configs for remaining seeds."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_peft_multiseed"
SEEDS = (20262028, 20262029, 20262030)


def main():
    source = yaml.safe_load((ROOT / "config_onetrans_peft_screen/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_onetrans_peft_screen/dataset_config.yaml").read_text())
    base = dict(source.get("Base", {}))
    base["model_root"] = "./runs_local/checkpoints_onetrans_peft_multiseed/"
    config = {"Base": base}
    film = yaml.safe_load(yaml.safe_dump(source["OneTrans_Peft_FiLM_Seed20262027"]))
    qvlora = yaml.safe_load(yaml.safe_dump(source["OneTrans_Peft_QVLoRA_R4_Seed20262027"]))
    for seed in SEEDS:
        config[f"OneTrans_Peft_FiLM_Seed{seed}"] = dict(film, seed=seed)
        config[f"OneTrans_Peft_QVLoRA_R4_Seed{seed}"] = dict(qvlora, seed=seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
