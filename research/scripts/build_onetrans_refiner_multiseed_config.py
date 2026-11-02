#!/usr/bin/env python3
"""Build paired linear-vs-branch OneTrans refiner configs on Taobao."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_refiner_multiseed"
SEEDS = (20262027, 20262028, 20262029, 20262030)


def main():
    source = yaml.safe_load((ROOT / "config_local/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    base = dict(source.get("Base", {}))
    base["model_root"] = "./runs_local/checkpoints_onetrans_refiner_multiseed/"
    config = {"Base": base}

    template = yaml.safe_load(yaml.safe_dump(
        source["OneTrans_Taobao_P2_BaseRefiner_L1"]))
    template.update({
        "enable_loop_diagnostics": False,
        "enable_torch_compile": True,
        "model_type": "base_refiner",
        "refiner_num_layers": 1,
        "supervision_mode": "progressive",
    })
    baseline = yaml.safe_load(yaml.safe_dump(
        source["OneTrans_Taobao_P1_Parallel"]))
    for seed in SEEDS:
        baseline_id = f"OneTrans_Parallel_Seed{seed}"
        linear_id = f"OneTrans_BaseRefiner_Linear_Seed{seed}"
        branch_id = f"OneTrans_BaseRefiner_Branch_Seed{seed}"
        config[baseline_id] = dict(baseline, seed=seed)
        config[linear_id] = dict(template, seed=seed, loop_topology="linear")
        config[branch_id] = dict(template, seed=seed, loop_topology="branch")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
