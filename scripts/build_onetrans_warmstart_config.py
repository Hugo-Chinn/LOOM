#!/usr/bin/env python3
"""Build retained-baseline and warm-start QV-LoRA configs for Taobao."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_warmstart"
SEEDS = (20262027, 20262028, 20262029, 20262030)


def main():
    source = yaml.safe_load((ROOT / "config_onetrans_refiner_multiseed/model_config.yaml").read_text())
    peft = yaml.safe_load((ROOT / "config_onetrans_peft_screen/model_config.yaml").read_text())
    dataset = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    base = dict(source.get("Base", {}))
    base.update({
        "model_root": "./runs_local/checkpoints_onetrans_warmstart/",
        "enable_torch_compile": False,
    })
    config = {"Base": base}

    baseline = yaml.safe_load(yaml.safe_dump(source["OneTrans_Parallel_Seed20262027"]))
    baseline.update({"enable_torch_compile": False, "keep_checkpoint": True})
    qvlora = yaml.safe_load(yaml.safe_dump(peft["OneTrans_Peft_QVLoRA_R4_Seed20262027"]))
    qvlora.update({"enable_torch_compile": False, "keep_checkpoint": False})

    checkpoint_root = ROOT / "runs_local/checkpoints_onetrans_warmstart/Taobao_Action"
    for seed in SEEDS:
        baseline_id = f"OneTrans_WarmSource_Seed{seed}"
        checkpoint = checkpoint_root / f"{baseline_id}.model"
        config[baseline_id] = dict(baseline, seed=seed)
        config[f"OneTrans_QVLoRA_FrozenBase_Seed{seed}"] = dict(
            qvlora,
            seed=seed,
            warm_start_mode="frozen_base",
            warm_start_checkpoint=str(checkpoint),
        )
        config[f"OneTrans_QVLoRA_LowLRBase_Seed{seed}"] = dict(
            qvlora,
            seed=seed,
            warm_start_mode="low_lr_base",
            warm_start_checkpoint=str(checkpoint),
            base_lr_scale=0.1,
        )
        config[f"OneTrans_QVLoRA_ZeroInitFrozen_Seed{seed}"] = dict(
            qvlora,
            seed=seed,
            warm_start_mode="frozen_base",
            warm_start_checkpoint=str(checkpoint),
            zero_init_refiner_output=True,
        )
        config[f"OneTrans_QVLoRA_ZeroInitLowLR_Seed{seed}"] = dict(
            qvlora,
            seed=seed,
            warm_start_mode="low_lr_base",
            warm_start_checkpoint=str(checkpoint),
            base_lr_scale=0.1,
            zero_init_refiner_output=True,
        )
        for suffix, dense_lr, sparse_lr in (
            ("LR1e4", 1.0e-4, 1.0e-3),
            ("LR3e5", 3.0e-5, 3.0e-4),
            ("LR1e5", 1.0e-5, 1.0e-4),
        ):
            config[f"OneTrans_QVLoRA_Conservative_{suffix}_Seed{seed}"] = dict(
                qvlora,
                seed=seed,
                warm_start_mode="frozen_base",
                warm_start_checkpoint=str(checkpoint),
                zero_init_refiner_output=True,
                freeze_warm_start_towers=True,
                dense_learning_rate=dense_lr,
                sparse_learning_rate=sparse_lr,
            )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
