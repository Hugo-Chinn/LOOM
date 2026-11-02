#!/usr/bin/env python3
"""Paired comparison of the rare-task bypass arms against their control arms.

Control arms come from `checkpoints_rankmixer_multiseed_confirmation`
(Kuai_DAGMeanPre2, TAAC_Loop_GatedDeep_Pre2) on the same seeds.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy.stats import ttest_rel
except ImportError:
    ttest_rel = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_all_local_experiments import parse_log, split_name  # noqa: E402

OUTER = Path("/home/HugoChinn/LoopMT")
PAIRS = [
    {
        "dataset": "KuaiRand_Video_Action",
        "family": "rankmixer_bypass_rare",
        "arm": "Kuai_BypassRare",
        "control_family": "rankmixer_multiseed_confirmation",
        "control_arm": "Kuai_DAGMeanPre2",
        "control_parallel": "Kuai_Parallel",
    },
    {
        "dataset": "TencentGR_10M_Action",
        "family": "rankmixer_bypass_rare",
        "arm": "TAAC_BypassRare",
        "control_family": "rankmixer_multiseed_confirmation",
        "control_arm": "TAAC_Loop_GatedDeep_Pre2",
        "control_parallel": "TAAC_Parallel",
    },
]


def collect(family):
    table = defaultdict(lambda: defaultdict(dict))
    for log in sorted((ROOT / "runs_local" / ("checkpoints_%s" % family)).rglob("*.log")):
        parsed = parse_log(log)
        if parsed is None:
            continue
        dataset = log.parent.name
        variant, seed = split_name(log.name.split(".log")[0])
        table[dataset][variant][seed] = {
            "tasks": {t: v["auc"] for t, v in parsed["metrics"].items()},
            "macro": mean(v["auc"] for v in parsed["metrics"].values()),
        }
    return table


def compare(candidate, control):
    seeds = sorted(set(candidate) & set(control))
    if len(seeds) < 2:
        return None
    deltas = [candidate[s]["macro"] - control[s]["macro"] for s in seeds]
    result = {
        "seeds": seeds,
        "delta_macro_mean": mean(deltas),
        "delta_macro_std": stdev(deltas),
        "wins": sum(d > 0 for d in deltas),
        "n": len(seeds),
    }
    if ttest_rel is not None:
        result["paired_t_p_two_sided"] = float(ttest_rel(
            [candidate[s]["macro"] for s in seeds],
            [control[s]["macro"] for s in seeds],
        ).pvalue)
    tasks = sorted(set(candidate[seeds[0]]["tasks"]) & set(control[seeds[0]]["tasks"]))
    per_task = {}
    for task in tasks:
        deltas_t = [candidate[s]["tasks"][task] - control[s]["tasks"][task] for s in seeds]
        per_task[task] = {
            "delta_mean": mean(deltas_t),
            "delta_std": stdev(deltas_t),
            "wins": sum(d > 0 for d in deltas_t),
            "control_mean": mean(control[s]["tasks"][task] for s in seeds),
            "candidate_mean": mean(candidate[s]["tasks"][task] for s in seeds),
        }
    result["per_task"] = per_task
    return result


def summarize_arm(by_seed):
    seeds = sorted(by_seed)
    tasks = sorted({t for s in seeds for t in by_seed[s]["tasks"]})
    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "macro_mean": mean(by_seed[s]["macro"] for s in seeds),
        "macro_std": stdev([by_seed[s]["macro"] for s in seeds]) if len(seeds) > 1 else 0.0,
        "per_task": {
            t: {
                "mean": mean(by_seed[s]["tasks"][t] for s in seeds if t in by_seed[s]["tasks"]),
                "std": stdev([by_seed[s]["tasks"][t] for s in seeds if t in by_seed[s]["tasks"]])
                if len(seeds) > 1 else 0.0,
            }
            for t in tasks
        },
    }


def main():
    tables = {f: collect(f) for f in ("rankmixer_bypass_rare", "rankmixer_multiseed_confirmation")}
    report = {}
    for spec in PAIRS:
        dataset = spec["dataset"]
        cand = tables[spec["family"]][dataset].get(spec["arm"], {})
        if not cand:
            continue
        entry = {"candidate": summarize_arm(cand)}
        for key, family, arm in (
            ("vs_control", spec["control_family"], spec["control_arm"]),
            ("vs_parallel", spec["control_family"], spec["control_parallel"]),
        ):
            control = tables[family][dataset].get(arm, {})
            if control:
                entry[key] = compare(cand, control)
                entry[key + "_summary"] = summarize_arm(control)
        report[dataset] = entry
    out = OUTER / "reports/rankmixer_bypass_rare_results.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
