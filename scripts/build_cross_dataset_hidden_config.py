#!/usr/bin/env python3
"""Build matched RankMixer configs for cross-dataset hidden-state analysis."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_cross_dataset_hidden"


def clone(x):
    return yaml.safe_load(yaml.safe_dump(x))


def main():
    general = yaml.safe_load((ROOT / "config_rankmixer_generalization/model_config.yaml").read_text())
    dag = yaml.safe_load((ROOT / "config_rankmixer_hierarchical/model_config.yaml").read_text())
    phase2 = yaml.safe_load((ROOT / "config_rankmixer_hierarchical_phase2/model_config.yaml").read_text())
    datasets = yaml.safe_load((ROOT / "config_rankmixer_multiseed_confirmation/dataset_config.yaml").read_text())

    selected = {
        "Hidden_MerRec_Raw": clone(general["RankMixer_MerRec_Baseline"]),
        "Hidden_MerRec_DAG": clone(dag["RankMixer_MerRec_DAG_Gated"]),
        "Hidden_QK_Raw": clone(general["RankMixer_QKVideo_Baseline"]),
        "Hidden_QK_DAG": clone(dag["RankMixer_QKVideo_DAG_Gated"]),
        "Hidden_Kuai_Raw": clone(general["RankMixer_KuaiRand_Baseline"]),
        "Hidden_Kuai_DAG": clone(phase2["RankMixer_KuaiRand_DAG_Gated_Mean_Pre2"]),
    }
    base = clone(general["Base"])
    base.update({
        "model_root": "./runs_local/checkpoints_cross_dataset_hidden/",
        "hidden_analysis_batches": 1,
        "hidden_analysis_samples": 8192,
        "hidden_analysis_projection_dim": 64,
        "hidden_analysis_dir": "./runs_local/cross_dataset_hidden",
        "keep_checkpoint": False,
    })
    config = {"Base": base, **selected}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    used = {v["dataset_id"] for v in selected.values()}
    (OUT / "dataset_config.yaml").write_text(
        yaml.safe_dump({name: datasets[name] for name in used}, sort_keys=False)
    )
    print(OUT)


if __name__ == "__main__":
    main()
