#!/usr/bin/env python3
"""Ensure every (dataset, strategy) arm in the OneTrans strategy study has all
4 seeds 20262027-20262030, copying parameters from an existing seed."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config_onetrans_loop_strategy_20261019" / "model_config.yaml"
SEEDS = [20262027, 20262028, 20262029, 20262030]
PATTERN = re.compile(r"^(?P<prefix>Strategy_onetrans_\w+?)_(?P<dataset>[A-Za-z0-9_]+)_S(?P<seed>\d{8})$")


def main():
    only_datasets = set(sys.argv[1:]) or None
    cfg = yaml.safe_load(CONFIG.read_text())
    by_key = {}
    for name, arm in cfg.items():
        if name == "Base":
            continue
        m = PATTERN.match(name)
        if not m:
            continue
        by_key.setdefault((m.group("prefix"), m.group("dataset")), {})[int(m.group("seed"))] = arm

    added = []
    for (prefix, dataset), seeds in sorted(by_key.items()):
        if only_datasets and dataset not in only_datasets:
            continue
        template_seed = sorted(seeds)[0]
        for seed in SEEDS:
            if seed in seeds:
                continue
            arm = dict(seeds[template_seed])
            arm["seed"] = seed
            cfg[f"{prefix}_{dataset}_S{seed}"] = arm
            added.append(f"{prefix}_{dataset}_S{seed}")
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False))
    print(f"added {len(added)} arms")
    for name in added[:20]:
        print("  ", name)


if __name__ == "__main__":
    main()
