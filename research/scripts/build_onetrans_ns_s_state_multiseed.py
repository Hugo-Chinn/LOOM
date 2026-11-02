#!/usr/bin/env python3
"""Build paired OneTrans NS/S state-mechanism experiments."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_ns_s_state_multiseed"
SEEDS = (20262027, 20262028, 20262029, 20262030)
DATASETS = ("Taobao_Action", "KuaiRand_Video_Action")
VARIANTS = {
    "Baseline": {"model_type": "parallel"},
    "RecurrentNS": {"refiner_mode": "recurrent"},
    "IndependentNS": {"refiner_mode": "independent"},
    "AnchoredScratchNS": {"refiner_mode": "anchored_scratch"},
    "DecoupledScratchNS": {"refiner_mode": "decoupled_scratch"},
    "SOnlyMemory": {
        "refiner_mode": "s_only",
        "memory_writer_rank": 8,
        "memory_gate_init": 0.01,
    },
}


def main():
    official = yaml.safe_load((ROOT / "config/model_config.yaml").read_text())
    local_datasets = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    local_base = yaml.safe_load((ROOT / "config_local/model_config.yaml").read_text())["Base"]
    config = {"Base": dict(local_base)}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_onetrans_ns_s_state_multiseed/",
        "enable_torch_compile": True,
        "num_workers": 4,
    })

    for dataset_id in DATASETS:
        source = yaml.safe_load(yaml.safe_dump(
            official[f"OneTrans_{dataset_id}"]))
        source["paired_fair_init"] = True
        common_loop = dict(
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
        for variant, changes in VARIANTS.items():
            if variant == "Baseline":
                template = dict(source, **changes)
            else:
                template = dict(source, **common_loop, **changes)
            for seed in SEEDS:
                expid = f"OneTransState_{dataset_id}_{variant}_S{seed}"
                config[expid] = dict(template, seed=seed)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False))
    selected = {dataset: local_datasets[dataset] for dataset in DATASETS}
    (OUT / "dataset_config.yaml").write_text(
        yaml.safe_dump(selected, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
