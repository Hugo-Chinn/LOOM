#!/usr/bin/env python3
"""Build matched RankMixer Loop ablations for iteration-depth conditioning."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_iteration_depth"


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    confirmation = yaml.safe_load(
        (ROOT / "config_rankmixer_multiseed_confirmation/model_config.yaml").read_text()
    )
    templates = {
        # Best confirmed KuaiRand Loop variant: behavior DAG + gates + two pre-loops.
        "Kuai_DAGMeanPre2": confirmation["RankMixer_Kuai_DAGMeanPre2_S20262028"],
        # Best confirmed TAAC Loop variant: gated deep supervision + two pre-loops.
        "TAAC_GatedDeepPre2": confirmation[
            "RankMixer_TAAC_Loop_GatedDeep_Pre2_S20262028"
        ],
    }

    config = {"Base": clone(confirmation["Base"])}
    config["Base"]["model_root"] = "./runs_local/checkpoints_rankmixer_iteration_depth/"
    for short, template in templates.items():
        control = clone(template)
        control["use_iteration_depth_embedding"] = False
        depth = clone(template)
        depth["use_iteration_depth_embedding"] = True
        config[f"RankMixer_{short}_Control"] = control
        config[f"RankMixer_{short}_IterDepth"] = depth

    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    dataset_ids = sorted({item["dataset_id"] for item in templates.values()})
    datasets = {dataset_id: local_data[dataset_id] for dataset_id in dataset_ids}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(datasets, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
