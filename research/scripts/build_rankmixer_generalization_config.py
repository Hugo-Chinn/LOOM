#!/usr/bin/env python3
"""Build baseline/loop RankMixer configs for three non-Taobao datasets."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_generalization"
DATASETS = {
    "MerRec": ("MerRec_Action", "RankMixer_MerRec_Action", 5),
    "QKVideo": ("QK_Video_Action", "RankMixer_QK_Video_Action", 4),
    "KuaiRand": ("KuaiRand_Video_Action", "RankMixer_KuaiRand_Video_Action", 6),
}


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    official = yaml.safe_load((ROOT / "config/model_config.yaml").read_text())
    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    config = {"Base": clone(official["Base"])}
    config["Base"]["model_root"] = "./runs_local/checkpoints_rankmixer_generalization/"
    selected_data = {}

    for short, (dataset_id, official_id, num_tasks) in DATASETS.items():
        baseline = clone(official[official_id])
        looped = clone(baseline)
        looped.update({
            "model_type": "looped",
            "num_loops": num_tasks,
            "share_loop_parameters": True,
            "use_behavior_embedding": True,
            "task_token_mode": "none",
            "loop_scope": "refinement",
            "loop_topology": "linear",
            "use_loop_gate": False,
            "supervision_mode": "deep_progressive",
            "auxiliary_loss_weight": 0.5,
        })
        config[f"RankMixer_{short}_Baseline"] = baseline
        config[f"RankMixer_{short}_LoopDeep"] = looped
        if short == "MerRec":
            gated = clone(looped)
            gated.update({
                "use_loop_gate": True,
                "loop_gate_init": 0.5,
            })
            config["RankMixer_MerRec_GatedLoopDeep"] = gated
        selected_data[dataset_id] = local_data[dataset_id]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(selected_data, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
