#!/usr/bin/env python3
"""Best OneTrans arm per dataset, paired against that study's own control."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_all_local_experiments import parse_log, split_name  # noqa: E402
from scipy.stats import ttest_rel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "runs_local"
OUT = Path("/home/HugoChinn/LoopMT/reports/onetrans_best_per_dataset.json")
EXTERNAL = {
    "onetrans_targeted_followup_multiseed": ("onetrans_ns_s_state_multiseed", "OneTransState_Baseline"),
}


def load(fam):
    d = defaultdict(lambda: defaultdict(dict))
    for log in sorted((ROOT / fam).rglob("*.log")):
        parsed = parse_log(log)
        if parsed is None:
            continue
        stem = log.name[:-4] if log.name.endswith(".log") else log.name
        if stem.endswith(".stdout"):
            stem = stem[:-7]
        var, seed = split_name(stem)
        d[log.parent.name][var][seed] = {k: v["auc"] for k, v in parsed["metrics"].items()}
    return d


def main():
    fams = sorted(p.name for p in ROOT.glob("checkpoints_onetrans_*"))
    data = {f: load(f) for f in fams}
    rows = []
    for fam in fams:
        fam_short = fam[len("checkpoints_"):] if fam.startswith("checkpoints_") else fam
        for ds, arms in data[fam].items():
            control_name, C = None, {}
            cands = [a for a in arms if any(t in a for t in ("Parallel", "Baseline"))]
            if cands:
                control_name = max(cands, key=lambda a: len(arms[a]))
                C = arms[control_name]
            elif fam_short in EXTERNAL:
                f2, c2 = EXTERNAL[fam_short]
                key = f2 if f2.startswith("checkpoints_") else f"checkpoints_{f2}"
                C = data.get(key, {}).get(ds, {}).get(c2) or {}
                control_name = f"{c2} [{f2}]"
            if not C:
                continue
            for arm in arms:
                if arm == control_name:
                    continue
                common = sorted(set(arms[arm]) & set(C))
                if len(common) < 3:
                    continue
                av = [mean(arms[arm][s].values()) for s in common]
                cv = [mean(C[s].values()) for s in common]
                dv = [a - b for a, b in zip(av, cv)]
                rows.append({
                    "dataset": ds, "family": fam, "arm": arm, "control": control_name,
                    "n_seeds": len(common), "arm_macro": mean(av),
                    "arm_macro_std": stdev(av) if len(av) > 1 else 0.0,
                    "control_macro": mean(cv), "delta_macro": mean(dv),
                    "delta_macro_std": stdev(dv) if len(dv) > 1 else 0.0,
                    "wins": sum(d > 0 for d in dv),
                    "paired_t_p_two_sided": float(ttest_rel(av, cv).pvalue),
                })
    rows.sort(key=lambda r: (r["dataset"], -r["delta_macro"]))
    OUT.write_text(json.dumps(rows, indent=2))
    for ds in sorted({r["dataset"] for r in rows}):
        print(f"\n== {ds}")
        for r in [x for x in rows if x["dataset"] == ds][:4]:
            print(f"  {r['delta_macro']:+.6f}±{r['delta_macro_std']:.6f} wins={r['wins']}/{r['n_seeds']} "
                  f"p={r['paired_t_p_two_sided']:.4f} | {r['arm'][:36]:36s} macro={r['arm_macro']:.6f} | "
                  f"ctrl {r['control'][:22]:22s} {r['control_macro']:.6f} | {r['family']}")


if __name__ == "__main__":
    main()
