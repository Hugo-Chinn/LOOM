#!/usr/bin/env python3
"""Build paired-seed confirmation configs for the two KuaiRand candidates."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_kuairand_routing_multiseed"


def clone(x):
    return yaml.safe_load(yaml.safe_dump(x))


def main():
    screen = yaml.safe_load((ROOT / "config_kuairand_routing_screen/model_config.yaml").read_text())
    datasets = yaml.safe_load((ROOT / "config_kuairand_routing_screen/dataset_config.yaml").read_text())
    base = clone(screen["Base"])
    base.update({
        "model_root": "./runs_local/checkpoints_kuairand_routing_multiseed/",
        "hidden_analysis_batches": 0,
        "keep_checkpoint": False,
    })
    configs = {"Base": base}
    arms = {
        "Bypass": screen["Kuai_Bypass"],
        "Full": screen["Kuai_Dynamic_Bypass_Adapter"],
    }
    for seed in (20262028, 20262029, 20262030):
        for arm, template in arms.items():
            name = f"Kuai_{arm}_S{seed}"
            cfg = clone(template)
            cfg["seed"] = seed
            configs[name] = cfg
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(configs, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(datasets, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
