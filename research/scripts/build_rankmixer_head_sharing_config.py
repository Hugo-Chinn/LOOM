#!/usr/bin/env python3
"""Build controlled RankMixer head-sharing ablations."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_head_sharing"


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    source_models = yaml.safe_load(
        (ROOT / "config_rankmixer_branch_gate_ablation/model_config.yaml").read_text()
    )
    source_data = yaml.safe_load(
        (ROOT / "config_rankmixer_branch_gate_ablation/dataset_config.yaml").read_text()
    )
    config = {"Base": clone(source_models["Base"])}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_rankmixer_head_sharing/",
        "hidden_analysis_dir": "./runs_local/hidden_analysis_head_sharing",
        "keep_checkpoint": False,
    })
    modes = {
        "Independent": {"head_sharing": "independent"},
        "SharedFirst": {"head_sharing": "shared_first"},
        "TopologyGrouped": {
            "head_sharing": "grouped",
            "head_groups": ["click", "middle", "middle", "buy"],
        },
        "TopologySharedFirst": {
            "head_sharing": "grouped_first",
            "head_groups": ["click", "middle", "middle", "buy"],
        },
        "TopologySharedTrunk": {
            "head_sharing": "grouped_trunk",
            "head_groups": ["click", "middle", "middle", "buy"],
        },
        "TopologySharedTrunkAdapter": {
            "head_sharing": "grouped_trunk_adapter",
            "head_groups": ["click", "middle", "middle", "buy"],
            "head_adapter_dim": 16,
        },
        "SharedTrunk": {"head_sharing": "shared_trunk"},
        "Shared": {"head_sharing": "shared"},
    }
    for seed in (20262027, 20262028, 20262029, 20262030):
        source_seed = seed if seed != 20262027 else 20262028
        template = clone(source_models[f"RankMixer_Branch_Seed{source_seed}"])
        template["seed"] = seed
        for suffix, updates in modes.items():
            item = clone(template)
            item.update(updates)
            config[f"RankMixer_Branch_{suffix}_Seed{seed}"] = item

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(
        yaml.safe_dump({"Taobao_Action": source_data["Taobao_Action"]}, sort_keys=False)
    )
    print(OUT)


if __name__ == "__main__":
    main()
