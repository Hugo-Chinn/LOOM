#!/usr/bin/env python3
"""Build a human-readable experiment report: configs + raw per-seed results + paired deltas."""
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

ROOT = Path(__file__).resolve().parents[1]
OUTER = Path("/home/HugoChinn/LoopMT")
INV = json.loads((OUTER / "reports/EXPERIMENT_CONFIG_INVENTORY_20261025.json").read_text())

# study -> control spec: ("in", arm_name) or ("ref", (other_study, arm_name)) or ("perds", {dataset: arm})
CONTROL = {
    "structure_taobao_dag": ("in", "DAGTrue"),
    "structure_kuai": ("in", "Struct_Kuai_True"),
    "structure_merrec": ("in", "Struct_MerRec_True"),
    "structure_qk": ("in", "Struct_QK_True"),
    "capacity_taobao": ("in", "Branch"),
    "capacity_kuai": ("perds", {"KuaiRand_Video_Action": "Kuai_DAGMeanPre2",
                                "TencentGR_10M_Action": "TAAC_Loop_GatedDeep_Pre2"}),
    "capacity_general": ("perds", {"MerRec_Action": "MerRec_DAGGated",
                                   "QK_Video_Action": "QK_DAGGated"}),
    "onetrans_strategy": ("perds", {"Taobao_Action": "Strategy_onetrans_Parallel_Taobao_Action",
                                    "KuaiRand_Video_Action": "Strategy_onetrans_Parallel_KuaiRand_Video_Action",
                                    "MerRec_Action": "Strategy_onetrans_Parallel_MerRec_Action",
                                    "QK_Video_Action": "Strategy_onetrans_Parallel_QK_Video_Action",
                                    "TencentGR_10M_Action": "Strategy_onetrans_Parallel_TencentGR_10M_Action"}),
    "onetrans_sweep1": ("ref", ("onetrans_strategy", "Strategy_onetrans_LoopControl_Taobao_Action")),
    "onetrans_sweep2": ("ref", ("onetrans_strategy", "Strategy_onetrans_LoopControl_Taobao_Action")),
    "onetrans_sweep3": ("ref", ("onetrans_strategy", "Strategy_onetrans_LoopControl_Taobao_Action")),
    "onetrans_mtl_aligned": ("in", "Head_ParallelIndependent"),
    "onetrans_ns_s_state": ("in", "OneTransState_Baseline"),
    "onetrans_targeted": ("ref", ("onetrans_ns_s_state", "OneTransState_Baseline")),
}
NOTE = {
    "bypass_rare": "对照组在同 seed 的 `checkpoints_rankmixer_multiseed_confirmation` 里（Kuai_DAGMeanPre2 / TAAC_Loop_GatedDeep_Pre2），见 `reports/rankmixer_bypass_rare_results.json`。",
    "task_weight_rare": "对照组是 `checkpoints_rankmixer_multiseed_confirmation` 的 Kuai_DAGMeanPre2，见 `reports/rankmixer_weight_rare_kuai_results.json`。",
    "loop_depth": "对照组是 `checkpoints_rankmixer_loop_capacity` / `checkpoints_rankmixer_multiseed_confirmation` 的 Kuai_DAGMeanPre2，见 `reports/rankmixer_capacity_four_seed_results.json`。",
    "kuai_specialized": "对照组是 Kuai_DAGMeanPre2（跨族），见 `reports/rankmixer_capacity_four_seed_results.json` 与本族四 seed 统计。",
}


def load(study):
    runs_dir = INV[study]["runs_dir"]
    d = defaultdict(lambda: defaultdict(dict))
    for log in sorted((ROOT / runs_dir).rglob("*.log")):
        p = parse_log(log)
        if p is None:
            continue
        stem = log.name[:-4] if log.name.endswith(".log") else log.name
        if stem.endswith(".stdout"):
            stem = stem[:-7]
        var, seed = split_name(stem)
        d[log.parent.name][var][seed] = {k: v["auc"] for k, v in p["metrics"].items()}
    return d


CACHE = {}


def get(study):
    if study not in CACHE:
        CACHE[study] = load(study)
    return CACHE[study]


def arm_name_in(study, ds, short):
    """Find full arm key containing `short` for dataset ds."""
    arms = get(study).get(ds, {})
    for a in arms:
        if a == short or a.startswith(short):
            return a
    return None


def fmt_delta(cand, ctrl, four=True):
    keys = set(cand) & set(ctrl)
    if four:
        keys = {k for k in keys if k in {"20262027", "20262028", "20262029", "20262030"}}
    common = sorted(keys)
    if len(common) < 2:
        return None
    av = [mean(cand[s].values()) for s in common]
    cv = [mean(ctrl[s].values()) for s in common]
    dv = [a - b for a, b in zip(av, cv)]
    per = {t: (mean(cand[s][t] for s in common) - mean(ctrl[s][t] for s in common))
           for t in sorted(cand[common[0]])}
    return {"n": len(common), "delta": mean(dv), "std": stdev(dv) if len(dv) > 1 else 0.0,
            "wins": sum(x > 0 for x in dv), "p": float(ttest_rel(av, cv).pvalue), "per_task": per,
            "cand_macro": mean(av), "ctrl_macro": mean(cv)}


def main():
    out = ["# 实验配置与原始结果汇总（截至 2026-10-25）", "",
           "本文件由 `UniRank/scripts/dump_experiment_inventory.py` + `scripts/build_experiment_report.py` 自动生成。",
           "机器可读版本：",
           "",
           "- 逐 run 原始结果（717 条）：`reports/EXPERIMENT_RAW_RESULTS_20261025.csv`",
           "- 各研究配置清单：`reports/EXPERIMENT_CONFIG_INVENTORY_20261025.json`",
           "- 日志原始文件：`UniRank/runs_local/<runs_dir>/<dataset>/<expid>.log`（Test 指标在 `******** Test evaluation ********` 之后）",
           ""]
    for study, inv in INV.items():
        data = get(study)
        out.append(f"## {study}")
        out.append("")
        out.append(f"- backbone: `{inv['backbone']}` ｜ 配置：`{inv['config']}`")
        out.append(f"- 权重/日志目录：`{inv['runs_dir']}` ｜ 臂数：{inv['n_arms']}")
        base = inv.get("base", {})
        keep = {k: base[k] for k in ("dense_learning_rate", "dense_optimizer", "sparse_learning_rate",
                                     "batch_size", "epochs", "embedding_dim", "num_workers", "save_best_only") if k in base}
        if keep:
            out.append("- Base 公共参数：" + "，".join(f"`{k}={v}`" for k, v in keep.items()))
        varying = inv.get("varying_knobs", {})
        if varying:
            out.append("- 跨臂变化的配置：")
            for knob, vals in sorted(varying.items()):
                uniq = {}
                for arm, v in vals.items():
                    uniq.setdefault(v, []).append(arm)
                if len(uniq) > 8:
                    out.append(f"  - `{knob}`：{len(uniq)} 种取值（见 JSON 清单）")
                else:
                    parts = []
                    for v, arms in uniq.items():
                        label = v if len(v) < 60 else v[:57] + "..."
                        parts.append(f"{label} → {len(arms)} 臂")
                    out.append(f"  - `{knob}`：" + "；".join(parts))
        if study in NOTE:
            out.append(f"- 对照说明：{NOTE[study]}")
        out.append("")
        # raw results
        for ds in sorted(data):
            arms = data[ds]
            tasks = sorted(next(iter(next(iter(arms.values())).values())))
            out.append(f"### 原始结果 · {ds}")
            out.append("")
            header = "| 臂 | seeds | " + " | ".join(tasks) + " | Macro |"
            out.append(header)
            out.append("|---" * (len(tasks) + 3) + "|")
            ctrl_spec = CONTROL.get(study)
            ctrl_seeds = None
            if ctrl_spec:
                kind = ctrl_spec[0]
                if kind == "in":
                    ctrl_seeds = arms.get(ctrl_spec[1])
                elif kind == "perds":
                    ctrl_seeds = arms.get(ctrl_spec[1].get(ds, ""))
                elif kind == "ref":
                    other, short = ctrl_spec[1]
                    ref = get(other)
                    key = arm_name_in(other, ds, short)
                    ctrl_seeds = ref.get(ds, {}).get(key) if key else None
            rows = []
            for arm in sorted(arms):
                seeds = sorted(arms[arm])
                if not seeds:
                    continue
                per = [mean(arms[arm][s][t] for s in seeds) for t in tasks]
                std = stdev([mean(arms[arm][s].values()) for s in seeds]) if len(seeds) > 1 else 0.0
                cells = " | ".join(f"{v:.6f}" for v in per)
                macro = mean(mean(arms[arm][s].values()) for s in seeds)
                d = ""
                if ctrl_seeds is not None and arm != (ctrl_spec[1] if ctrl_spec and ctrl_spec[0] == "in" else None):
                    r = fmt_delta(arms[arm], ctrl_seeds)
                    if r:
                        d = f" (Δ {r['delta']:+.6f}±{r['std']:.6f}, {r['wins']}/{r['n']}, p={r['p']:.4f})"
                rows.append(f"| `{arm}` | {len(seeds)} | {cells} | {macro:.6f} ± {std:.6f}{d} |")
            out += rows
            out.append("")
    (OUTER / "reports/EXPERIMENT_CONFIG_AND_RESULTS_20261025.md").write_text("\n".join(out))
    print("wrote report, lines:", len(out))


if __name__ == "__main__":
    main()
