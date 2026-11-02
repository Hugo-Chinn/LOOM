#!/usr/bin/env python3
"""Build phase-2 DAG ablations for parent fusion, supervision, and pre-loops."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config_rankmixer_hierarchical"
OUT = ROOT / "config_rankmixer_hierarchical_phase2"


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def add(config, templates, source, expid, **updates):
    item = clone(templates[source])
    item.update(updates)
    config[expid] = item


def main():
    source_models = yaml.safe_load((SOURCE / "model_config.yaml").read_text())
    source_data = yaml.safe_load((SOURCE / "dataset_config.yaml").read_text())
    config = {"Base": clone(source_models["Base"])}
    config["Base"]["model_root"] = "./runs_local/checkpoints_rankmixer_hierarchical_phase2/"

    # MerRec needs residual gates for numerical stability. Its Purchase node has
    # three parents, so child-specific softmax edge weights are the key ablation.
    add(config, source_models, "RankMixer_MerRec_DAG_Gated", "RankMixer_MerRec_DAG_Gated_Softmax",
        dag_parent_fusion="softmax")
    add(config, source_models, "RankMixer_MerRec_DAG_Gated", "RankMixer_MerRec_DAG_Gated_Softmax_Pre2",
        dag_parent_fusion="softmax", num_preloops=2)

    # KuaiRand tests the three suspected failure modes independently:
    # equal parent fusion, insufficient root depth, and descendant deep loss.
    add(config, source_models, "RankMixer_KuaiRand_DAG_Gated", "RankMixer_KuaiRand_DAG_Gated_Softmax",
        dag_parent_fusion="softmax")
    add(config, source_models, "RankMixer_KuaiRand_DAG_Gated", "RankMixer_KuaiRand_DAG_Gated_Softmax_Pre2",
        dag_parent_fusion="softmax", num_preloops=2)
    add(config, source_models, "RankMixer_KuaiRand_DAG_Gated", "RankMixer_KuaiRand_DAG_Gated_Mean_Pre2",
        dag_parent_fusion="mean", num_preloops=2)
    add(config, source_models, "RankMixer_KuaiRand_DAG_Gated", "RankMixer_KuaiRand_DAG_Gated_Softmax_Progressive",
        dag_parent_fusion="softmax", supervision_mode="progressive")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(source_data, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
