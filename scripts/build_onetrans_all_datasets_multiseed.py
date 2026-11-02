#!/usr/bin/env python3
"""Build cross-dataset, multi-seed OneTrans baseline/FiLM/QV-LoRA configs."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_all_datasets_multiseed"
SEEDS = (20262027, 20262028, 20262029)
DATASETS = (
    "TencentGR_10M_Action",
    "Taobao_Action",
    "QK_Video_Action",
    "MerRec_Action",
    "KuaiRand_Video_Action",
)


def main():
    official = yaml.safe_load((ROOT / "config/model_config.yaml").read_text())
    local_datasets = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    local_base = yaml.safe_load((ROOT / "config_local/model_config.yaml").read_text())["Base"]
    config = {"Base": dict(local_base)}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_onetrans_all_datasets_multiseed/",
        "enable_torch_compile": True,
        "num_workers": 4,
    })

    for dataset_id in DATASETS:
        source_id = f"OneTrans_{dataset_id}"
        baseline = yaml.safe_load(yaml.safe_dump(official[source_id]))
        baseline.update({
            "model_type": "parallel",
            "enable_loop_diagnostics": False,
        })
        common_loop = dict(
            model_type="base_refiner",
            num_loops=int(baseline["num_tasks"]),
            share_loop_parameters=True,
            use_behavior_embedding=True,
            supervision_mode="progressive",
            refiner_num_layers=1,
            refiner_gate_init=0.1,
            loop_topology="linear",
            enable_loop_diagnostics=False,
        )
        film = dict(baseline, **common_loop, peft_type="film")
        qvlora = dict(
            baseline,
            **common_loop,
            peft_type="lora_qv",
            peft_rank=4,
            peft_alpha=4.0,
            peft_dropout=0.0,
        )
        for seed in SEEDS:
            config[f"OneTransAll_{dataset_id}_Baseline_S{seed}"] = dict(
                baseline, seed=seed)
            config[f"OneTransAll_{dataset_id}_FiLM_S{seed}"] = dict(
                film, seed=seed)
            config[f"OneTransAll_{dataset_id}_QVLoRA_S{seed}"] = dict(
                qvlora, seed=seed)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    selected_datasets = {dataset_id: local_datasets[dataset_id] for dataset_id in DATASETS}
    (OUT / "dataset_config.yaml").write_text(
        yaml.safe_dump(selected_datasets, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
