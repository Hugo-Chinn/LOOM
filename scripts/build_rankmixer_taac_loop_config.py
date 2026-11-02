#!/usr/bin/env python3
"""Build matched TAAC/TencentGR-10M RankMixer loop experiments."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_taac_loop"


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    official = yaml.safe_load((ROOT / "config/model_config.yaml").read_text())
    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    base = clone(official["RankMixer_TencentGR_10M_Action"])
    common = {
        "batch_size": 8192,
        "epochs": 1,
        "seed": 20262027,
        "task_token_mode": "none",
        "use_behavior_embedding": True,
    }
    config = {"Base": clone(official["Base"])}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_rankmixer_taac_loop/",
        "num_workers": 4,
    })

    variants = {
        # Re-run Parallel on the same one-GPU protocol used by every variant.
        "RankMixer_TAAC_Parallel": {
            "model_type": "parallel",
        },
        # Independent one-layer blocks: controls recurrent sharing vs depth.
        "RankMixer_TAAC_Sequential": {
            "model_type": "sequential",
            "num_loops": 2,
            "share_loop_parameters": False,
            "loop_scope": "refinement",
            "supervision_mode": "progressive",
        },
        "RankMixer_TAAC_Loop_Progressive": {
            "model_type": "looped",
            "num_loops": 2,
            "share_loop_parameters": True,
            "loop_scope": "refinement",
            "loop_topology": "linear",
            "supervision_mode": "progressive",
            "use_loop_gate": False,
        },
        # Click is also supervised after conversion refinement.
        "RankMixer_TAAC_Loop_Deep": {
            "model_type": "looped",
            "num_loops": 2,
            "share_loop_parameters": True,
            "loop_scope": "refinement",
            "loop_topology": "linear",
            "supervision_mode": "deep_progressive",
            "auxiliary_loss_weight": 0.5,
            "use_loop_gate": False,
        },
        "RankMixer_TAAC_Loop_GatedDeep": {
            "model_type": "looped",
            "num_loops": 2,
            "share_loop_parameters": True,
            "loop_scope": "refinement",
            "loop_topology": "linear",
            "supervision_mode": "deep_progressive",
            "auxiliary_loss_weight": 0.5,
            "use_loop_gate": True,
            "loop_gate_init": 0.5,
        },
        # Same shared F is applied twice before behavior-specific refinement.
        "RankMixer_TAAC_Loop_GatedDeep_Pre2": {
            "model_type": "looped",
            "num_loops": 2,
            "share_loop_parameters": True,
            "loop_scope": "refinement",
            "loop_topology": "linear",
            "supervision_mode": "deep_progressive",
            "auxiliary_loss_weight": 0.5,
            "use_loop_gate": True,
            "loop_gate_init": 0.5,
            "num_preloops": 2,
        },
    }
    for expid, update in variants.items():
        item = clone(base)
        item.update(common)
        item.update(update)
        config[expid] = item

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump({
        "TencentGR_10M_Action": local_data["TencentGR_10M_Action"]
    }, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
