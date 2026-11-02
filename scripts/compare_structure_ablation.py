#!/usr/bin/env python3
"""Compare structural perturbations of the behaviour graph against the true design."""
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
OUTER = Path("/home/HugoChinn/LoopMT")
FOUR = {"20262027", "20262028", "20262029", "20262030"}
SEEDS = [20262027, 20262028, 20262029, 20262030]


def load(root):
    d = defaultdict(dict)
    for log in sorted(Path(root).rglob("*.log")):
        parsed = parse_log(log)
        if parsed is None:
            continue
        stem = log.name[:-4] if log.name.endswith(".log") else log.name
        if stem.endswith(".stdout"):
            stem = stem[:-7]
        var, seed = split_name(stem)
        d[var][seed] = {k: v["auc"] for k, v in parsed["metrics"].items()}
    return d


def macro(seeds):
    return mean(mean(seeds[s].values()) for s in sorted(seeds))


def main():
    abl = load(ROOT / "checkpoints_rankmixer_structure_ablation" / "Taobao_Action")
    branch = {}
    for fam in ("checkpoints_branch_gate_ablation", "checkpoints_rankmixer_capacity_taobao"):
        for arm, seeds in load(ROOT / fam / "Taobao_Action").items():
            if arm == "Branch" or arm == "Taobao_Branch":
                branch.update(seeds)
    fs = json.loads((OUTER / "reports/unirank_taobao_rankmixer_four_seed_full.json").read_text())
    raw = json.loads((OUTER / "reports/unirank_taobao_raw_rankmixer_four_seed.json").read_text())
    lin = {str(s): {t: v["AUC"] for t, v in fs["runs"][str(s)]["Linear"]["metrics"].items()} for s in SEEDS}
    par = {str(s): {t: v["AUC"] for t, v in raw["runs"][str(s)]["metrics"].items()} for s in SEEDS}
    bc = sorted(set(branch) & FOUR)
    print("controls: Branch %d seeds %.6f | Linear %.6f | Parallel %.6f"
          % (len(bc), macro({s: branch[s] for s in bc}), macro(lin), macro(par)))
    rows = []
    for name, seeds in abl.items():
        bshared = sorted(set(branch) & set(seeds) & FOUR)
        if len(bshared) < 3:
            continue
        common = bshared
        av = [mean(seeds[s].values()) for s in common]
        bv = [mean(branch[s].values()) for s in common]
        lv = [mean(lin[s].values()) for s in common]
        pv = [mean(par[s].values()) for s in common]
        rows.append({
            "arm": name, "n_seeds": len(common), "macro": mean(av),
            "macro_std": stdev(av) if len(av) > 1 else 0.0,
            "vs_branch": mean(av) - mean(bv),
            "vs_branch_wins": sum(a > b for a, b in zip(av, bv)),
            "vs_branch_p": float(ttest_rel(av, bv).pvalue),
            "vs_linear": mean(av) - mean(lv), "vs_parallel": mean(av) - mean(pv),
            "per_task": {t: mean(seeds[s][t] for s in common) for t in sorted(seeds[common[0]])},
        })
    rows.sort(key=lambda r: -r["macro"])
    for r in rows:
        print("%-30s n=%d macro=%.6f±%.6f | vs Branch %+.6f (%d/%d, p=%.4f) | vs Linear %+.6f | vs Parallel %+.6f"
              % (r["arm"], r["n_seeds"], r["macro"], r["macro_std"], r["vs_branch"],
                 r["vs_branch_wins"], r["n_seeds"], r["vs_branch_p"], r["vs_linear"], r["vs_parallel"]))
    (OUTER / "reports/rankmixer_structure_ablation_taobao.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
