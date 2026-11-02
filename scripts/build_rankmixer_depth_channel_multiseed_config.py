#!/usr/bin/env python3
"""Build four-seed confirmation configs for RankMixer depth channel scaling."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_depth_channel_multiseed"
SEEDS = (20262027, 20262028, 20262029, 20262030)


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    head_sharing = yaml.safe_load(
        (ROOT / "config_rankmixer_head_sharing/model_config.yaml").read_text()
    )
    dag = yaml.safe_load(
        (ROOT / "config_rankmixer_hierarchical/model_config.yaml").read_text()
    )
    phase2 = yaml.safe_load(
        (ROOT / "config_rankmixer_hierarchical_phase2/model_config.yaml").read_text()
    )
    confirmed = yaml.safe_load(
        (ROOT / "config_rankmixer_multiseed_confirmation/model_config.yaml").read_text()
    )
    config = {"Base": clone(confirmed["Base"])}
    config["Base"]["model_root"] = (
        "./runs_local/checkpoints_rankmixer_depth_channel_multiseed/"
    )
    templates = {
        "Taobao": head_sharing["RankMixer_Branch_Independent_Seed20262027"],
        "MerRec": dag["RankMixer_MerRec_DAG_Gated"],
        "Kuai": phase2["RankMixer_KuaiRand_DAG_Gated_Mean_Pre2"],
    }
    for seed in SEEDS:
        for dataset, template in templates.items():
            item = clone(template)
            item.update(seed=seed, loop_depth_conditioning="channel_scale")
            config[f"RankMixer_{dataset}_DepthChannelScale_S{seed}"] = item

    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    dataset_ids = sorted({
        config[key]["dataset_id"] for key in config if key != "Base"
    })
    datasets = {dataset_id: local_data[dataset_id] for dataset_id in dataset_ids}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(datasets, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
