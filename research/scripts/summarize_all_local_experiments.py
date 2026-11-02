#!/usr/bin/env python3
"""Aggregate every local checkpoints_* experiment family into one table.

Parses the test-metric block of each run log, groups runs by
family / dataset / variant / seed, and reports macro AUC with paired
statistics against the family baseline (auto-detected when present).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy.stats import ttest_rel
except ImportError:  # keep the parser usable without scipy
    ttest_rel = None


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_local"
JSON_OUT = ROOT / "reports/local_experiment_inventory.json"
MD_OUT = ROOT / "reports/LOCAL_EXPERIMENT_INVENTORY.md"
KNOWN_DATASETS = (
    "Taobao_Action", "KuaiRand_Video_Action", "MerRec_Action",
    "QK_Video_Action", "TencentGR_10M_Action",
)
# Continuation studies keep their control seeds in the family they extend; merge
# those arms so the new family reports the full paired seed set.
EXTRA_LOG_DIRS = {
    "rankmixer_taac_six_seed": (
        "checkpoints_rankmixer_multiseed_confirmation/TencentGR_10M_Action",
        ("Sequential", "Loop_Deep", "Loop_GatedDeep_Pre2"),
        (20262028, 20262029, 20262030),
    ),
}
BASELINE_CANDIDATES = ("Baseline", "Branch", "Sequential", "Parallel", "Independent")
METRIC = re.compile(
    r"\[Task: ([^]]+)\]\[Metrics\].*?"
    r"(?:AUC: ([0-9.]+) - logloss: ([0-9.]+)|"
    r"logloss: ([0-9.]+) - AUC: ([0-9.]+))"
)
PARAMS = re.compile(r"Total number of parameters: ([0-9]+)\.")
TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+")
SEED = re.compile(r"(?:_Seed|_S|Seed|S)(\d{8})$")


def parse_log(path: Path):
    text = path.read_text(errors="ignore")
    if "******** Test evaluation ********" not in text:
        return None
    section = text.rsplit("******** Test evaluation ********", 1)[1]
    metrics = {}
    for match in METRIC.finditer(section):
        metrics[match.group(1)] = {
            "auc": float(match.group(5) or match.group(2)),
            "logloss": float(match.group(4) or match.group(3)),
        }
    if not metrics:
        return None
    param_match = PARAMS.search(text)
    times = [
        datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        for line in text.splitlines() if (m := TIMESTAMP.match(line))
    ]
    return {
        "metrics": metrics,
        "parameters": int(param_match.group(1)) if param_match else None,
        "wall_seconds": (times[-1] - times[0]).total_seconds() if len(times) >= 2 else None,
    }


def split_name(stem: str):
    """Return (variant, seed) from a run-log stem."""
    match = SEED.search(stem)
    seed = match.group(1) if match else "single"
    stem = stem[: match.start()] if match else stem
    for dataset in KNOWN_DATASETS:
        stem = stem.replace(f"_{dataset}", "_")
    stem = re.sub(r"^(RankMixer|OneTrans|OneTransAll|OneTransTarget)_", "", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return (stem or "run"), seed


def _shared_prefix_arm(variants, arm, suffix):
    """True when every other arm shares the arm's prefix, so a delta is meaningful."""
    prefix = arm[: -len(suffix)]
    if not prefix:
        return True
    return all(o.startswith(prefix) for o in variants if o != arm)


def find_baseline(variants):
    """Locate the control arm of a family, or None for cross-model leaderboards.

    A `<prefix>_<control>` arm only counts when all other arms share the same
    prefix; otherwise the family mixes unrelated models and no delta is
    meaningful (example: `hidden_analysis`).
    """
    for suffix in ("_Baseline", "_Branch", "_Sequential", "_Parallel", "_Independent"):
        for arm in variants:
            if arm.endswith(suffix) and _shared_prefix_arm(variants, arm, suffix):
                return arm
    for arm in variants:
        if arm in BASELINE_CANDIDATES:
            return arm
    return None


def paired(candidate: dict, control: dict):
    """Macro-AUC paired comparison over shared seeds."""
    common = sorted(set(candidate) & set(control))
    if len(common) < 2:
        return None
    deltas = [candidate[s] - control[s] for s in common]
    result = {
        "seeds": common,
        "delta_macro_mean": mean(deltas),
        "delta_macro_std": stdev(deltas) if len(deltas) > 1 else 0.0,
        "wins": sum(d > 0 for d in deltas),
    }
    if ttest_rel is not None:
        result["paired_t_p_two_sided"] = float(
            ttest_rel([candidate[s] for s in common], [control[s] for s in common]).pvalue
        )
    return result


def main():
    families = {}
    for ckpt_dir in sorted(RUNS.glob("checkpoints_*")):
        family = ckpt_dir.name[len("checkpoints_"):]
        # family -> dataset -> variant -> seed -> metrics
        table = defaultdict(lambda: defaultdict(dict))
        meta = defaultdict(lambda: defaultdict(dict))
        logs = sorted(ckpt_dir.rglob("*.log"))
        if family in EXTRA_LOG_DIRS:
            subdir, arms, seeds = EXTRA_LOG_DIRS[family]
            logs += sorted(
                RUNS / subdir / f"RankMixer_TAAC_{arm}_S{seed}.log"
                for arm in arms for seed in seeds
            )
        for log in logs:
            parsed = parse_log(log)
            if parsed is None:
                continue
            dataset = log.parent.name if log.parent != ckpt_dir else "unknown"
            variant, seed = split_name(log.name.split(".log")[0])
            macro = mean(v["auc"] for v in parsed["metrics"].values())
            table[dataset][variant][seed] = {
                "tasks": parsed["metrics"],
                "macro_auc": macro,
            }
            meta[dataset][variant][seed] = {
                "parameters": parsed["parameters"],
                "wall_seconds": parsed["wall_seconds"],
            }
        if not table:
            continue
        families[family] = {"table": table, "meta": meta}

    result = {}
    for family, payload in families.items():
        table = payload["table"]
        meta = payload["meta"]
        out = {}
        for dataset, variants in table.items():
            baseline = find_baseline(variants)
            control = {s: v["macro_auc"] for s, v in variants[baseline].items()} if baseline else None
            entry = {"baseline": baseline, "seeds": sorted(next(iter(variants.values()))), "variants": {}}
            for variant, by_seed in variants.items():
                macro = {s: v["macro_auc"] for s, v in by_seed.items()}
                tasks = sorted({t for v in by_seed.values() for t in v["tasks"]})
                params = [meta[dataset][variant][s]["parameters"] for s in by_seed]
                wall = [meta[dataset][variant][s]["wall_seconds"] for s in by_seed]
                entry["variants"][variant] = {
                    "n_seeds": len(by_seed),
                    "macro_auc": mean(macro.values()),
                    "macro_auc_std": stdev(macro.values()) if len(macro) > 1 else 0.0,
                    "per_task_auc": {
                        t: mean(v["tasks"][t]["auc"] for v in by_seed.values()) for t in tasks
                    },
                    "per_task_logloss": {
                        t: mean(v["tasks"][t]["logloss"] for v in by_seed.values()) for t in tasks
                    },
                    "parameters": next((p for p in params if p), None),
                    "minutes_per_run": (
                        mean(w for w in wall if w) / 60 if any(wall) else None
                    ),
                    "vs_baseline": paired(macro, control) if (control and variant != baseline) else None,
                }
            out[dataset] = entry
        result[family] = out

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# Local experiment inventory", "", f"Source: `runs_local/checkpoints_*` (auto-generated by `scripts/summarize_all_local_experiments.py`).", ""]
    for family in sorted(result, key=lambda f: (RUNS / f"checkpoints_{f}").stat().st_mtime, reverse=True):
        lines.append(f"## {family}")
        for dataset, entry in result[family].items():
            variants = entry["variants"]
            baseline = entry["baseline"]
            base_macro = variants[baseline]["macro_auc"] if baseline else None
            lines.append("")
            lines.append(f"**{dataset}** - baseline: `{baseline or 'n/a'}`, seeds: {len(entry['seeds'])}")
            lines.append("")
            lines.append("| Variant | Seeds | Macro AUC | Δ macro | p | Wins | Worst task Δ | Params | Min/run |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for variant in sorted(variants, key=lambda v: -variants[v]["macro_auc"]):
                v = variants[variant]
                cmp = v["vs_baseline"] or {}
                delta = v["macro_auc"] - base_macro if base_macro is not None else None
                worst = min(
                    (v["per_task_auc"][t] - b for t, b in
                     ((t, variants[baseline]["per_task_auc"][t]) for t in v["per_task_auc"])),
                    default=None,
                ) if baseline else None
                p = cmp.get("paired_t_p_two_sided")
                lines.append(
                    "| {v} | {n} | {m:.6f} | {d} | {p} | {w} | {worst} | {par} | {mins} |".format(
                        v=f"`{variant}`" + (" (base)" if variant == baseline else ""),
                        n=v["n_seeds"],
                        m=v["macro_auc"],
                        d=f"{delta:+.6f}" if delta is not None else "-",
                        p=f"{p:.3f}" if p is not None else "-",
                        w=f"{cmp.get('wins', '-')}/{len(cmp.get('seeds', []))}" if cmp else "-",
                        worst=f"{worst:+.6f}" if worst is not None else "-",
                        par=f"{v['parameters']:,}" if v["parameters"] else "-",
                        mins=f"{v['minutes_per_run']:.1f}" if v["minutes_per_run"] else "-",
                    )
                )
        lines.append("")
    MD_OUT.write_text("\n".join(lines) + "\n")
    print(JSON_OUT, MD_OUT, len(result), "families")


if __name__ == "__main__":
    main()
