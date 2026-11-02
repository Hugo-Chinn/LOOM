#!/usr/bin/env python3
"""Build targeted OneTrans follow-up experiments for Taobao and KuaiRand."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_targeted_followup_multiseed"
SEEDS = (20262027, 20262028, 20262029, 20262030)
VARIANTS = {
    "Taobao_Action": {
        "ContextDAGMean": {
            "refiner_mode": "independent",
            "loop_topology": "context_dag_mean",
        },
        "ContextDAGLearned": {
            "refiner_mode": "independent",
            "loop_topology": "context_dag_learned",
        },
    },
    "KuaiRand_Video_Action": {
        "DecProtect05": {
            "refiner_mode": "decoupled_scratch",
            "protected_task_indices": [1, 3],
            "protected_state_loss_weight": 0.05,
        },
        "DecProtect10": {
            "refiner_mode": "decoupled_scratch",
            "protected_task_indices": [1, 3],
            "protected_state_loss_weight": 0.10,
        },
        "CrossOnly": {
            "refiner_mode": "cross_only_scratch",
        },
        "CrossOnlyProtect05": {
            "refiner_mode": "cross_only_scratch",
            "protected_task_indices": [1, 3],
            "protected_state_loss_weight": 0.05,
        },
    },
}


def main():
    official = yaml.safe_load((ROOT / "config/model_config.yaml").read_text())
    local_datasets = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    local_base = yaml.safe_load((ROOT / "config_local/model_config.yaml").read_text())["Base"]
    config = {"Base": dict(local_base)}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_onetrans_targeted_followup_multiseed/",
        "enable_torch_compile": True,
        "num_workers": 4,
    })

    for dataset_id, variants in VARIANTS.items():
        source = yaml.safe_load(yaml.safe_dump(
            official[f"OneTrans_{dataset_id}"]))
        source["paired_fair_init"] = True
        common = dict(
            model_type="base_refiner",
            num_loops=int(source["num_tasks"]),
            share_loop_parameters=True,
            use_behavior_embedding=True,
            supervision_mode="progressive",
            refiner_num_layers=1,
            refiner_gate_init=0.1,
            loop_topology="linear",
            peft_type="none",
            enable_loop_diagnostics=False,
        )
        for variant, changes in variants.items():
            template = dict(source)
            template.update(common)
            template.update(changes)
            for seed in SEEDS:
                expid = f"OneTransTarget_{dataset_id}_{variant}_S{seed}"
                config[expid] = dict(template, seed=seed)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump({
        dataset: local_datasets[dataset] for dataset in VARIANTS
    }, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
