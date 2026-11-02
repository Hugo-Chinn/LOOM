#!/usr/bin/env python3
"""Dump raw per-run results + the config knobs that vary, for a set of studies."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_all_local_experiments import parse_log, split_name  # noqa: E402
from scipy.stats import ttest_rel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTER = Path("/home/HugoChinn/LoopMT")
SEED_KEYS = ("seed",)

STUDIES = [
    # (key, config dir, runs dir, backbone label)
    ("structure_taobao_dag", "config_rankmixer_structure_ablation_dag", "checkpoints_rankmixer_structure_ablation_dag", "RankMixer"),
    ("structure_kuai", "config_rankmixer_structure_kuai", "checkpoints_rankmixer_structure_kuai", "RankMixer"),
    ("structure_merrec", "config_rankmixer_structure_merrec", "checkpoints_rankmixer_structure_merrec", "RankMixer"),
    ("structure_qk", "config_rankmixer_structure_qk", "checkpoints_rankmixer_structure_qk", "RankMixer"),
    ("capacity_kuai", "config_rankmixer_loop_capacity", "checkpoints_rankmixer_loop_capacity", "RankMixer"),
    ("capacity_general", "config_rankmixer_capacity_general", "checkpoints_rankmixer_capacity_general", "RankMixer"),
    ("capacity_taobao", "config_rankmixer_capacity_taobao", "checkpoints_rankmixer_capacity_taobao", "RankMixer"),
    ("bypass_rare", "config_rankmixer_bypass_rare", "checkpoints_rankmixer_bypass_rare", "RankMixer"),
    ("task_weight_rare", "config_rankmixer_task_weight_rare", "checkpoints_rankmixer_task_weight_rare", "RankMixer"),
    ("loop_depth", "config_rankmixer_loop_depth", "checkpoints_rankmixer_loop_depth", "RankMixer"),
    ("kuai_specialized", "config_rankmixer_kuai_specialized", "checkpoints_rankmixer_kuai_specialized", "RankMixer"),
    ("onetrans_strategy", "config_onetrans_loop_strategy_20261019", "checkpoints_onetrans_loop_strategy_20261019", "OneTrans"),
    ("onetrans_sweep1", "config_onetrans_taobao_sweep", "checkpoints_onetrans_taobao_sweep", "OneTrans"),
    ("onetrans_sweep2", "config_onetrans_taobao_sweep2", "checkpoints_onetrans_taobao_sweep2", "OneTrans"),
    ("onetrans_sweep3", "config_onetrans_taobao_sweep3", "checkpoints_onetrans_taobao_sweep3", "OneTrans"),
    ("onetrans_mtl_aligned", "config_onetrans_mtl_aligned", "checkpoints_onetrans_mtl_aligned", "OneTrans"),
    ("onetrans_ns_s_state", "config_onetrans_ns_s_state_multiseed", "checkpoints_onetrans_ns_s_state_multiseed", "OneTrans"),
    ("onetrans_targeted", "config_onetrans_targeted_followup_multiseed", "checkpoints_onetrans_targeted_followup_multiseed", "OneTrans"),
]


def load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text())


def load_results(runs_dir):
    d = defaultdict(lambda: defaultdict(dict))
    paths = {}
    parent = ROOT / "runs_local" / runs_dir
    for log in sorted(parent.rglob("*.log")):
        parsed = parse_log(log)
        if parsed is None:
            continue
        stem = log.name[:-4] if log.name.endswith(".log") else log.name
        if stem.endswith(".stdout"):
            stem = stem[:-7]
        var, seed = split_name(stem)
        ds = log.parent.name
        d[ds][var][seed] = {k: v["auc"] for k, v in parsed["metrics"].items()}
        paths[(ds, var, seed)] = str(log)
    return d, paths


def main():
    rows = []
    summary = {}
    for key, cfgdir, runsdir, backbone in STUDIES:
        cfg_path = ROOT / cfgdir / "model_config.yaml"
        if not cfg_path.exists():
            continue
        cfg = load_yaml(cfg_path)
        arms = {k: v for k, v in cfg.items() if k != "Base"}
        # knobs that vary across arms (ignore seed)
        common_keys = set.intersection(*[set(a) for a in arms.values()]) if arms else set()
        varying = {}
        for k in sorted(common_keys):
            vals = {name: json.dumps(a[k], sort_keys=True) for name, a in arms.items()}
            if len(set(vals.values())) > 1:
                varying[k] = vals
        res, paths = load_results(runsdir)
        for ds, armres in res.items():
            for arm, seeds in armres.items():
                for seed, tasks in seeds.items():
                    rows.append({
                        "study": key, "backbone": backbone, "dataset": ds, "arm": arm, "seed": seed,
                        "macro_auc": round(mean(tasks.values()), 6),
                        **{("auc_" + t): round(v, 6) for t, v in sorted(tasks.items())},
                        "log": paths.get((ds, arm, seed), ""),
                    })
        summary[key] = {
            "config": str(cfg_path), "runs_dir": f"runs_local/{runsdir}", "backbone": backbone,
            "base": cfg.get("Base", {}),
            "n_arms": len(arms),
            "varying_knobs": varying,
            "datasets": sorted(res),
            "arms": {ds: {a: sorted(s) for a, s in ar.items()} for ds, ar in res.items()},
        }

    # CSV
    cols = ["study", "backbone", "dataset", "arm", "seed", "macro_auc"]
    extra = sorted({c for r in rows for c in r if c.startswith("auc_")})
    cols += extra + ["log"]
    with open(OUTER / "reports/EXPERIMENT_RAW_RESULTS_20261025.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["study"], r["dataset"], r["arm"], r["seed"])):
            w.writerow(r)
    (OUTER / "reports/EXPERIMENT_CONFIG_INVENTORY_20261025.json").write_text(json.dumps(summary, indent=2))
    print("runs:", len(rows), "studies:", len(summary))
    for k, v in summary.items():
        print(" -", k, "| arms", v["n_arms"], "| datasets", v["datasets"], "| varying:", list(v["varying_knobs"])[:6])


if __name__ == "__main__":
    main()
