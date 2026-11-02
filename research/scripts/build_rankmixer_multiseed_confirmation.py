#!/usr/bin/env python3
"""Build paired multi-seed confirmation configs for the frozen Loop variants."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_multiseed_confirmation"
SEEDS = (20262028, 20262029, 20262030)


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    general = yaml.safe_load((ROOT / "config_rankmixer_generalization/model_config.yaml").read_text())
    dag = yaml.safe_load((ROOT / "config_rankmixer_hierarchical/model_config.yaml").read_text())
    phase2 = yaml.safe_load((ROOT / "config_rankmixer_hierarchical_phase2/model_config.yaml").read_text())
    taac = yaml.safe_load((ROOT / "config_rankmixer_taac_loop/model_config.yaml").read_text())

    templates = {
        # MerRec: matched Parallel, previous linear gated Loop, and frozen DAG winner.
        "MerRec_Parallel": general["RankMixer_MerRec_Baseline"],
        "MerRec_LinearGated": general["RankMixer_MerRec_GatedLoopDeep"],
        "MerRec_DAGGated": dag["RankMixer_MerRec_DAG_Gated"],
        # QK-Video: matched Parallel, linear Loop, and graph-aware gated Loop.
        "QK_Parallel": general["RankMixer_QKVideo_Baseline"],
        "QK_Linear": general["RankMixer_QKVideo_LoopDeep"],
        "QK_DAGGated": dag["RankMixer_QKVideo_DAG_Gated"],
        # KuaiRand: include the negative linear control and phase-2 PreLoop winner.
        "Kuai_Parallel": general["RankMixer_KuaiRand_Baseline"],
        "Kuai_Linear": general["RankMixer_KuaiRand_LoopDeep"],
        "Kuai_DAGMeanPre2": phase2["RankMixer_KuaiRand_DAG_Gated_Mean_Pre2"],
    }
    # TAAC is cheap to define but expensive to train. Confirm the complete frozen
    # six-cell ablation so selection on seed 20262027 cannot bias the conclusion.
    for name in (
        "Parallel", "Sequential", "Loop_Progressive", "Loop_Deep",
        "Loop_GatedDeep", "Loop_GatedDeep_Pre2",
    ):
        templates[f"TAAC_{name}"] = taac[f"RankMixer_TAAC_{name}"]

    config = {"Base": clone(general["Base"])}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_rankmixer_multiseed_confirmation/",
        "num_workers": 4,
    })
    for seed in SEEDS:
        for short, template in templates.items():
            item = clone(template)
            item["seed"] = seed
            config[f"RankMixer_{short}_S{seed}"] = item

    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    dataset_ids = sorted({item["dataset_id"] for item in templates.values()})
    datasets = {dataset_id: local_data[dataset_id] for dataset_id in dataset_ids}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(datasets, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
