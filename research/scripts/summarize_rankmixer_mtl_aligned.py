#!/usr/bin/env python3
"""Snapshot-scoped complete metrics, paired deltas and explicit reference gates."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_rankmixer_mtl_aligned")
    args = parser.parse_args()
    config = args.config.resolve()
    manifest_path = config / "manifest.json"
    snapshot = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    logdir = ROOT / "logs/rankmixer_mtl_aligned" / snapshot[:12]
    groups = defaultdict(dict)
    completed, pending = {}, []
    for expid, arm in manifest["arms"].items():
        if arm["status"] != "ready":
            continue
        stamp = logdir / f"{expid}.done.json"
        if not stamp.exists():
            pending.append(expid)
            continue
        result = json.loads(stamp.read_text())
        if result["snapshot"] != snapshot:
            raise RuntimeError(f"Invalid snapshot stamp {stamp}")
        values = result["metrics"]
        macro = mean(value["auc"] for value in values.values())
        completed[expid] = macro
        groups[(arm["study"], arm["dataset"], arm["variant"])][arm["seed"]] = result
    rows = []
    for (study, ds, variant), seeds in sorted(groups.items()):
        macros = {seed: mean(v["auc"] for v in result["metrics"].values())
                  for seed, result in seeds.items()}
        baseline = "Parallel" if study == "historical_replication" else "ParallelIndependentTower"
        prefix = "History" if study == "historical_replication" else "Head"
        paired_seeds = [s for s in sorted(seeds)
                        if f"{prefix}_{baseline}_{ds}_S{s}" in completed]
        candidate = [macros[s] for s in paired_seeds]
        control = [completed[f"{prefix}_{baseline}_{ds}_S{s}"] for s in paired_seeds]
        deltas = [a - b for a, b in zip(candidate, control)]
        row = {"study": study, "dataset": ds, "variant": variant,
               "seeds": sorted(seeds), "n": len(seeds), "macro_mean": mean(macros.values()),
               "macro_std": stdev(macros.values()) if len(seeds) > 1 else None,
               "per_seed": {s: r["metrics"] for s, r in seeds.items()},
               "paired_seeds": paired_seeds}
        if variant != baseline and deltas:
            row["delta"] = mean(deltas)
            row["wins"] = sum(d > 0 for d in deltas)
            if len(deltas) > 1:
                p = float(ttest_rel(candidate, control).pvalue)
                row["p_value"] = p if math.isfinite(p) else None
        rows.append(row)
    # The primary attribution test is tied vs untied, not the baseline leaderboard.
    primary = []
    for ds in sorted({a["dataset"] for a in manifest["arms"].values()}):
        pairs = [s for s in manifest["seeds"]
                 if f"Head_LoopMT_{ds}_S{s}" in completed
                 and f"Head_UntiedDeep_{ds}_S{s}" in completed]
        if not pairs:
            continue
        a = [completed[f"Head_LoopMT_{ds}_S{s}"] for s in pairs]
        b = [completed[f"Head_UntiedDeep_{ds}_S{s}"] for s in pairs]
        p = float(ttest_rel(a, b).pvalue) if len(pairs) > 1 else None
        primary.append({"dataset": ds, "seeds": pairs, "delta": mean(x-y for x, y in zip(a,b)),
                        "p_value": p if p is not None and math.isfinite(p) else None})
    out = ROOT / "reports"
    payload = {"snapshot": snapshot, "completed": len(completed), "pending": pending,
               "reference_blocked": [n for n, a in manifest["arms"].items()
                                     if a["status"] == "blocked_reference"],
               "rows": rows, "primary_tied_vs_untied": primary}
    name = "onetrans_mtl_aligned" if manifest.get("backbone") == "OneTrans" else "rankmixer_mtl_aligned"
    (out / f"{name}_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# RankMixer aligned MTL progress", "", f"Snapshot: `{snapshot[:12]}`.",
             f"Completed: {len(completed)}/260 ready runs; reference-blocked: 60.",
             "Screen seed + three confirmation seeds; no legacy results included.",
             "Partial-seed rows are provisional; p-values are unadjusted exploratory values.", "",
             "| Study | Dataset | Variant | Seeds | Macro AUC | Delta vs control |",
             "|---|---|---|---:|---:|---:|"]
    for r in rows:
        delta = f"{r['delta']:+.6f}" if "delta" in r else "-"
        lines.append(f"| {r['study']} | {r['dataset']} | {r['variant']} | "
                     f"{r['n']}/4 | {r['macro_mean']:.6f} | {delta} |")
    lines.extend(["", "## Primary: tied vs untied", ""])
    for r in primary:
        lines.append(f"- {r['dataset']}: {len(r['seeds'])}/4 paired seeds, "
                     f"delta {r['delta']:+.6f}; p={r['p_value']}.")
    (out / f"{name.upper()}_PROGRESS.md").write_text("\n".join(lines) + "\n")
    print(f"{len(completed)} complete, {len(pending)} pending, snapshot {snapshot[:12]}")


if __name__ == "__main__":
    main()
