#!/usr/bin/env python3
"""Build multi-seed residual-logit OneTrans configs."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_residual_multiseed"
SEEDS = (20262028, 20262029, 20262030)


def main():
    source = yaml.safe_load((ROOT / "config_onetrans_transition_screen/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_onetrans_transition_screen/dataset_config.yaml").read_text())
    base = dict(source.get("Base", {}))
    base["model_root"] = "./runs_local/checkpoints_onetrans_residual_multiseed/"
    config = {"Base": base}
    template = yaml.safe_load(yaml.safe_dump(
        source["OneTrans_ResidualLogit_Seed20262027"]))
    checkpoint_root = ROOT / "runs_local/checkpoints_onetrans_warmstart/Taobao_Action"
    for seed in SEEDS:
        config[f"OneTrans_ResidualLogit_Seed{seed}"] = dict(
            template,
            seed=seed,
            warm_start_checkpoint=str(
                checkpoint_root / f"OneTrans_WarmSource_Seed{seed}.model"),
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
