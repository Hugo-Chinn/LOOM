#!/usr/bin/env python3
"""Build the rare-task bypass study config.

Arms: the current best looped template per boundary dataset, plus a
residual base-state bypass restricted to the tasks whose gradient barely
reaches the shared refinement block (KuaiRand: follow/comment/forward,
TAAC: conversion).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "config_rankmixer_multiseed_confirmation"
DST = ROOT / "config_rankmixer_bypass_rare"
SEEDS = [20262028, 20262029, 20262030]
SPECS = [
    {
        "prefix": "RankMixer_Kuai_BypassRare",
        "base": "RankMixer_Kuai_DAGMeanPre2_S20262028",
        "bypass_tasks": ["is_follow", "is_comment", "is_forward"],
    },
    {
        "prefix": "RankMixer_TAAC_BypassRare",
        "base": "RankMixer_TAAC_Loop_GatedDeep_Pre2_S20262028",
        "bypass_tasks": ["is_conversion"],
    },
]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / "dataset_config.yaml", DST / "dataset_config.yaml")
    cfg = yaml.safe_load((SRC / "model_config.yaml").read_text())
    base = dict(cfg["Base"])
    base["model_root"] = "./runs_local/checkpoints_rankmixer_bypass_rare/"
    base["keep_checkpoint"] = True
    out = {"Base": base}
    for spec in SPECS:
        template = dict(cfg[spec["base"]])
        for seed in SEEDS:
            arm = dict(template)
            arm["seed"] = seed
            arm["use_base_state_bypass"] = True
            arm["base_bypass_mode"] = "residual"
            arm["base_bypass_tasks"] = list(spec["bypass_tasks"])
            out["%s_S%d" % (spec["prefix"], seed)] = arm
    (DST / "model_config.yaml").write_text(
        yaml.safe_dump(out, sort_keys=True, default_flow_style=False, allow_unicode=True)
    )
    print("wrote", DST / "model_config.yaml", "arms:", len(out) - 1)


if __name__ == "__main__":
    main()
