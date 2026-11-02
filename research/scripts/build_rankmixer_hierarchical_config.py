#!/usr/bin/env python3
"""Build hierarchical behavior-DAG RankMixer configs."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_hierarchical"
SPECS = {
    "MerRec": {
        "official": "RankMixer_MerRec_Action",
        "dataset": "MerRec_Action",
        "graph": {
            "Like": [],
            "Cart": ["Like"],
            "Offer": ["Like"],
            "Checkout": ["Like"],
            "Purchase": ["Cart", "Offer", "Checkout"],
        },
    },
    "QKVideo": {
        "official": "RankMixer_QK_Video_Action",
        "dataset": "QK_Video_Action",
        "graph": {
            "click": [],
            "follow": ["like"],
            "like": ["click"],
            "share": ["like"],
        },
    },
    "KuaiRand": {
        "official": "RankMixer_KuaiRand_Video_Action",
        "dataset": "KuaiRand_Video_Action",
        "graph": {
            "is_click": [],
            "is_follow": ["is_like", "long_view"],
            "is_like": ["is_click"],
            "is_comment": ["is_like", "long_view"],
            "is_forward": ["is_like", "long_view"],
            "long_view": ["is_click"],
        },
    },
}


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    official = yaml.safe_load((ROOT / "config/model_config.yaml").read_text())
    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    config = {"Base": clone(official["Base"])}
    config["Base"]["model_root"] = "./runs_local/checkpoints_rankmixer_hierarchical/"
    datasets = {}
    for short, spec in SPECS.items():
        base = clone(official[spec["official"]])
        base.update({
            "model_type": "looped",
            "num_loops": len(spec["graph"]),
            "share_loop_parameters": True,
            "use_behavior_embedding": True,
            "task_token_mode": "none",
            "loop_scope": "refinement",
            "loop_topology": "dag",
            "behavior_graph": spec["graph"],
            "supervision_mode": "deep_progressive",
            "auxiliary_loss_weight": 0.5,
            "use_loop_gate": False,
        })
        gated = clone(base)
        gated.update({"use_loop_gate": True, "loop_gate_init": 0.5})
        config[f"RankMixer_{short}_DAG"] = base
        config[f"RankMixer_{short}_DAG_Gated"] = gated
        datasets[spec["dataset"]] = local_data[spec["dataset"]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(datasets, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
