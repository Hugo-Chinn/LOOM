#!/usr/bin/env python3
"""Build the rare-task loss-weighting study config.

Hypothesis: on KuaiRand the shared refinement is driven almost entirely by
is_click / long_view (gradient-norm share 1.00 / 0.95) while is_follow /
is_comment / is_forward sit at 0.04-0.08. Amplifying the rare tasks' primary
loss weight increases their share of the shared-block update without removing
their access to the refined state (which the bypass arm showed they need).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "config_rankmixer_multiseed_confirmation"
DST = ROOT / "config_rankmixer_task_weight_rare"
SEEDS = [20262028, 20262029, 20262030]
SPECS = [
    {
        "prefix": "RankMixer_Kuai_WeightRare4",
        "base": "RankMixer_Kuai_DAGMeanPre2_S20262028",
        # label order: is_click, is_follow, is_like, is_comment, is_forward, long_view
        "task_loss_weights": [1.0, 4.0, 1.0, 4.0, 4.0, 1.0],
    },
    {
        "prefix": "RankMixer_TAAC_WeightConv4",
        "base": "RankMixer_TAAC_Loop_GatedDeep_Pre2_S20262028",
        # label order: is_click, is_conversion
        "task_loss_weights": [1.0, 4.0],
    },
]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / "dataset_config.yaml", DST / "dataset_config.yaml")
    cfg = yaml.safe_load((SRC / "model_config.yaml").read_text())
    base = dict(cfg["Base"])
    base["model_root"] = "./runs_local/checkpoints_rankmixer_task_weight_rare/"
    out = {"Base": base}
    for spec in SPECS:
        template = dict(cfg[spec["base"]])
        for seed in SEEDS:
            arm = dict(template)
            arm["seed"] = seed
            arm["task_loss_weights"] = list(spec["task_loss_weights"])
            out["%s_S%d" % (spec["prefix"], seed)] = arm
    (DST / "model_config.yaml").write_text(
        yaml.safe_dump(out, sort_keys=True, default_flow_style=False, allow_unicode=True)
    )
    print("wrote", DST / "model_config.yaml", "arms:", len(out) - 1)


if __name__ == "__main__":
    main()
