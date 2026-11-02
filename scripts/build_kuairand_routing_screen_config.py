#!/usr/bin/env python3
"""Build the first KuaiRand hidden-state-guided Loop optimization screen."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_kuairand_routing_screen"


def clone(x):
    return yaml.safe_load(yaml.safe_dump(x))


def main():
    source = yaml.safe_load(
        (ROOT / "config_rankmixer_hierarchical_phase2/model_config.yaml").read_text()
    )
    datasets = yaml.safe_load(
        (ROOT / "config_rankmixer_multiseed_confirmation/dataset_config.yaml").read_text()
    )
    template = source["RankMixer_KuaiRand_DAG_Gated_Mean_Pre2"]
    variants = {
        "Kuai_Current": {},
        "Kuai_Bypass": {"use_base_state_bypass": True, "base_bypass_init": 0.5},
        "Kuai_Dynamic": {"dag_parent_fusion": "dynamic_sigmoid"},
        "Kuai_Dynamic_Bypass": {
            "dag_parent_fusion": "dynamic_sigmoid",
            "use_base_state_bypass": True,
            "base_bypass_init": 0.5,
        },
        "Kuai_Dynamic_Bypass_Adapter": {
            "dag_parent_fusion": "dynamic_sigmoid",
            "use_base_state_bypass": True,
            "base_bypass_init": 0.5,
            "child_adapter_dim": 16,
        },
    }
    base = clone(source["Base"])
    base.update({
        "model_root": "./runs_local/checkpoints_kuairand_routing_screen/",
        "hidden_analysis_batches": 1,
        "hidden_analysis_samples": 8192,
        "hidden_analysis_projection_dim": 64,
        "hidden_analysis_dir": "./runs_local/kuairand_routing_hidden",
        "keep_checkpoint": False,
    })
    configs = {"Base": base}
    for name, updates in variants.items():
        cfg = clone(template)
        cfg.update(updates)
        configs[name] = cfg
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(configs, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(
        yaml.safe_dump({"KuaiRand_Video_Action": datasets["KuaiRand_Video_Action"]}, sort_keys=False)
    )
    print(OUT)


if __name__ == "__main__":
    main()
