#!/usr/bin/env python3
"""Historical RankMixer ablations with explicit controls and seed provenance."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import ttest_rel
from summarize_all_local_experiments import parse_log, split_name

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
CANONICAL = {"20262027", "20262028", "20262029", "20262030"}
TASK_COUNTS = {
    "TencentGR_10M_Action": 2, "QK_Video_Action": 4,
    "KuaiRand_Video_Action": 6, "MerRec_Action": 5, "Taobao_Action": 4,
}


def summarize(records):
    values = [r["macro"] for r in records.values()]
    return {"seeds": sorted(records), "n": len(values), "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else None}


def compare(a, b):
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    deltas = [a[s]["macro"] - b[s]["macro"] for s in seeds]
    p = float(ttest_rel([a[s]["macro"] for s in seeds],
                        [b[s]["macro"] for s in seeds]).pvalue) if len(seeds) > 1 else None
    return {
        "seeds": seeds, "n": len(seeds), "delta": mean(deltas),
        "candidate_mean": mean(a[s]["macro"] for s in seeds),
        "control_mean": mean(b[s]["macro"] for s in seeds),
        "wins": sum(d > 0 for d in deltas),
        "p": p if p is not None and math.isfinite(p) else None,
        "per_task_delta": {
            t: mean(a[s]["metrics"][t]["auc"] - b[s]["metrics"][t]["auc"] for s in seeds)
            for t in a[seeds[0]]["metrics"]
        },
    }


def main():
    table = defaultdict(dict)
    skipped = []
    directories = sorted((ROOT / "runs_local").glob("checkpoints_rankmixer*"))
    directories.append(ROOT / "runs_local/checkpoints_branch_gate_ablation")
    for directory in directories:
        family = directory.name.removeprefix("checkpoints_")
        if family in {"rankmixer_mtl_aligned", "rankmixer_viz"}:
            continue
        for path in sorted(directory.rglob("*.log")):
            record = parse_log(path)
            if record is None:
                continue
            ds = path.parent.name
            if ds not in TASK_COUNTS or len(record["metrics"]) != TASK_COUNTS[ds]:
                skipped.append(str(path.relative_to(ROOT)))
                continue
            variant, seed = split_name(path.name.split(".log")[0])
            key = (family, ds, variant)
            if seed in table[key]:
                raise ValueError(f"Duplicate seed provenance: {key}/{seed}")
            table[key][seed] = {
                **record, "macro": mean(v["auc"] for v in record["metrics"].values()),
                "log": str(path.relative_to(ROOT)),
            }

    # Continuation controls are joined only by explicit arm identity and seed.
    def merged(ds, arm, families):
        result = {}
        for family in families:
            for seed, r in table.get((family, ds, arm), {}).items():
                if seed in result and result[seed]["metrics"] != r["metrics"]:
                    raise ValueError(f"Conflicting continuation control: {ds}/{arm}/{seed}")
                result[seed] = r
        return result

    def loop_control(ds):
        arm = {
            "KuaiRand_Video_Action": "Kuai_DAGMeanPre2",
            "TencentGR_10M_Action": "TAAC_Loop_GatedDeep_Pre2",
            "MerRec_Action": "MerRec_DAGGated",
            "QK_Video_Action": "QK_DAGGated",
            "Taobao_Action": "Taobao_Branch",
        }[ds]
        families = ["rankmixer_multiseed_confirmation", "rankmixer_loop_capacity",
                    "rankmixer_capacity_general", "rankmixer_capacity_taobao"]
        if ds == "Taobao_Action":
            # Original ungated Branch control is the independent-head study.
            result = merged(ds, "Branch_Independent", ["rankmixer_head_sharing"])
            result.update(merged(ds, arm, ["rankmixer_capacity_taobao"]))
        else:
            result = merged(ds, arm, families)
        return arm, result

    comparison_rows = []

    def add_comparison(family, ds, variant, candidate, control_name, control):
        for scope, seeds in [("all_available", set(candidate)), ("canonical_four", CANONICAL)]:
            a = {s: r for s, r in candidate.items() if s in seeds}
            b = {s: r for s, r in control.items() if s in seeds}
            if a:
                comparison_rows.append({
                    "family": family, "dataset": ds, "variant": variant, "scope": scope,
                    "summary": summarize(a), "control": control_name,
                    "comparison": compare(a, b),
                })

    for (family, ds, variant), candidate in sorted(table.items()):
        control_name, control = None, {}
        if family in {
            "rankmixer_loop_capacity", "rankmixer_capacity_general", "rankmixer_capacity_taobao",
            "rankmixer_task_weight_rare", "rankmixer_bypass_rare",
            "rankmixer_depth_channel_multiseed", "rankmixer_recurrence_conditioning",
            "rankmixer_safe_routing", "rankmixer_safe_routing_multiseed",
        }:
            control_name, control = loop_control(ds)
        elif family == "rankmixer_loop_depth":
            control_name = "Kuai_Wide4_WeightRare4" if "WeightRare4" in variant else "Kuai_Wide4"
            control = merged(ds, control_name, ["rankmixer_loop_capacity"])
        elif family == "rankmixer_head_sharing":
            control_name = "Branch_Independent"
            control = merged(ds, control_name, [family])
        elif family.startswith("rankmixer_preference_margin"):
            control_name = "Branch_Baseline"
            control = merged(ds, control_name, ["rankmixer_preference_margin"])
        elif family == "rankmixer_multiseed_confirmation":
            control_name = {
                "TencentGR_10M_Action": "TAAC_Parallel", "MerRec_Action": "MerRec_Parallel",
                "QK_Video_Action": "QK_Parallel", "KuaiRand_Video_Action": "Kuai_Parallel",
            }[ds]
            control = merged(ds, control_name, [family])
        if not control or variant == control_name:
            continue
        add_comparison(family, ds, variant, candidate, control_name, control)

    # Component comparisons use the same named arm rather than a generic baseline.
    for ds, family, arm, control_family, control_arm in [
        ("KuaiRand_Video_Action", "rankmixer_loop_capacity", "Kuai_Wide4_WeightRare4",
         "rankmixer_loop_capacity", "Kuai_Wide4"),
        ("KuaiRand_Video_Action", "rankmixer_capacity_general", "Kuai_Wide8_WeightRare4",
         "rankmixer_loop_capacity", "Kuai_Wide4_WeightRare4"),
        ("Taobao_Action", "rankmixer_capacity_taobao", "Taobao_Wide4_WeightDeep2",
         "rankmixer_capacity_taobao", "Taobao_Wide4"),
        ("MerRec_Action", "rankmixer_capacity_general", "MerRec_Wide4_WeightLow2",
         "rankmixer_capacity_general", "MerRec_Wide4"),
        ("TencentGR_10M_Action", "rankmixer_loop_capacity", "TAAC_Wide4_BypassConv",
         "rankmixer_bypass_rare", "TAAC_BypassRare"),
        ("MerRec_Action", "rankmixer_multiseed_confirmation", "MerRec_DAGGated",
         "rankmixer_multiseed_confirmation", "MerRec_LinearGated"),
        ("QK_Video_Action", "rankmixer_multiseed_confirmation", "QK_DAGGated",
         "rankmixer_multiseed_confirmation", "QK_Linear"),
    ]:
        add_comparison(family, ds, arm, table[(family, ds, arm)], control_arm,
                       table[(control_family, ds, control_arm)])
    ds = "TencentGR_10M_Action"
    for arm, control_arm in [
        ("TAAC_Loop_Deep", "TAAC_Loop_Progressive"),
        ("TAAC_Loop_GatedDeep", "TAAC_Loop_Deep"),
        ("TAAC_Loop_GatedDeep_Pre2", "TAAC_Loop_GatedDeep"),
    ]:
        family = "rankmixer_multiseed_confirmation"
        add_comparison(family, ds, arm, table[(family, ds, arm)], control_arm,
                       table[(family, ds, control_arm)])

    stem = "HISTORICAL_LOOP_ABLATIONS_20261019"
    lines = [
        "# Historical RankMixer Loop Ablations", "",
        "Source: complete test blocks in local logs; frozen MTL head runs are excluded.",
        "Macro AUC is an unweighted task mean. SD is sample SD across seeds.",
        "P-values are exploratory, two-sided paired t-tests, without multiplicity correction.",
        "Different candidate/control seed counts are allowed; deltas use ONLY shared seeds.",
        "Do not subtract full-sample means to recover a paired-subset delta.",
        "Canonical seeds: 20262027, 20262028, 20262029, 20262030.",
        "Single-seed arms have no estimable SD or significance test.",
        "Controls are named explicitly; several studies change more than one component.",
        "Seed labels 'single' are not used for cross-family pairing.",
        "", "## Seed Coverage Correction", "",
        "Complete 20262027 test logs exist for KuaiRand Wide4/WeightRare4, Taobao Wide4,",
        "and MerRec/QK capacity arms even where the original YAML does not list them.",
        "KuaiRand Wide4_WeightRare4 has six completed seeds: 20262027-20262032.",
        "TAAC Wide4_BypassConv still has only 20262028-20262030.",
        "Earlier claims based solely on YAML that all three improved arms lack 20262027",
        "were incorrect. Log presence establishes metrics, not frozen-config provenance.",
        "", "## Paired Comparisons", "",
        "| Family | Dataset | Variant | Scope | Candidate N | Macro mean +/- SD | Control | Paired N | Delta | Wins | p | Paired seeds |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in comparison_rows:
        s, c = row["summary"], row["comparison"]
        sd = f"{s['std']:.6f}" if s["std"] is not None else "NA"
        lines.append(
            f"| {row['family']} | {row['dataset']} | {row['variant']} | {row['scope']} | "
            f"{s['n']} | {s['mean']:.6f} +/- {sd} | {row['control']} | "
            + (f"{c['n']} | {c['delta']:+.6f} | {c['wins']}/{c['n']} | "
               f"{c['p'] if c['p'] is not None else 'NA'} | {', '.join(c['seeds'])} |"
               if c else "0 | NA | NA | NA | none |")
        )
    raw_rows, task_rows = [], []
    lines += ["", "## Full Arm and Per-Task Tables", ""]
    for (family, ds, variant), records in sorted(table.items()):
        s = summarize(records)
        sd = f"{s['std']:.6f}" if s["std"] is not None else "NA"
        lines += [
            f"### {family} / {ds} / {variant}", "",
            f"Seeds: {', '.join(s['seeds'])}. Macro: {s['mean']:.6f} +/- {sd}.", "",
            "| Task | N | AUC mean | AUC sample SD |", "|---|---:|---:|---:|",
        ]
        for task in next(iter(records.values()))["metrics"]:
            aucs = [r["metrics"][task]["auc"] for r in records.values()]
            task_sd = stdev(aucs) if len(aucs) > 1 else None
            row = {"family": family, "dataset": ds, "variant": variant, "task": task,
                   "n": len(aucs), "seeds": ",".join(s["seeds"]),
                   "auc_mean": mean(aucs), "auc_std": task_sd}
            task_rows.append(row)
            lines.append(f"| {task} | {len(aucs)} | {mean(aucs):.6f} | "
                         f"{f'{task_sd:.6f}' if task_sd is not None else 'NA'} |")
        lines.append("")
        for seed, r in records.items():
            for task, metrics in r["metrics"].items():
                raw_rows.append({"family": family, "dataset": ds, "variant": variant,
                                 "seed": seed, "task": task, **metrics, "log": r["log"]})
    (OUT / f"{stem}.md").write_text("\n".join(lines) + "\n")
    (OUT / f"{stem}.json").write_text(json.dumps({
        "canonical_seeds": sorted(CANONICAL), "comparisons": comparison_rows,
        "skipped_incomplete_task_logs": skipped,
    }, indent=2) + "\n")
    overview = [
        "# Historical Loop Ablation Overview", "",
        "Only original interaction-stage RankMixer Loops; no post-backbone head ablations.",
        "Canonical seeds: 20262027-20262030. Some studies have only a subset.",
        "Mean +/- sample SD across available canonical seeds; NA for one seed.",
        "Delta, wins and p use the explicitly listed control on shared seeds only.",
        "Paired t-tests are exploratory and unadjusted; selected best arms are not held-out tests.",
        "Some topology comparisons also change gating/conditioning; do not infer tying effects.",
        "",
    ]
    selections = [
        ("Capacity, Weighting and Bypass", {
            "Kuai_Wide4", "Kuai_Wide4_WeightRare4", "Kuai_WeightRare4",
            "Kuai_Wide8_WeightRare4", "Kuai_Depth2", "Kuai_Depth2_WeightRare4",
            "TAAC_BypassRare", "TAAC_Wide4_BypassConv", "Kuai_BypassRare",
            "Taobao_Wide4", "Taobao_Wide4_WeightDeep2", "MerRec_Wide4",
            "MerRec_Wide4_WeightLow2", "QK_Wide4",
        }),
        ("Supervision, Gate, Preloops and Topology", {
            "TAAC_Loop_Progressive", "TAAC_Loop_Deep", "TAAC_Loop_GatedDeep",
            "TAAC_Loop_GatedDeep_Pre2", "MerRec_DAGGated", "QK_DAGGated",
        }),
        ("Task Tower Sharing (Taobao)", {
            "Branch_Shared", "Branch_SharedFirst", "Branch_SharedTrunk",
            "Branch_TopologyGrouped", "Branch_TopologySharedTrunk",
        }),
        ("Recurrence Conditioning and Routing", {
            "Taobao_DepthChannelScale", "MerRec_DepthChannelScale", "Kuai_DepthChannelScale",
            "Taobao_BehaviorZero", "Taobao_BehaviorNormal002", "Taobao_BehaviorScaled01",
            "Kuai_AdapterOnly", "Kuai_AdapterResidualBypass",
        }),
        ("Preference Margin (Taobao)", {
            "Branch_FixedMargin", "Branch_ProgressiveMargin", "Branch_ReverseProgressiveMargin",
            "Branch_ProgressiveW0005", "Branch_ProgressiveW001", "Branch_ProgressiveW002",
            "Branch_ProgressiveW005", "Branch_ZeroMarginBPRW002", "Branch_W00075M020",
        }),
    ]
    for heading, variants in selections:
        overview += [
            f"## {heading}", "",
            "| Dataset | Variant | N | Mean +/- SD | Control | Paired N | Delta | Wins | p |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|",
        ]
        for r in comparison_rows:
            if r["scope"] != "canonical_four" or r["variant"] not in variants:
                continue
            s, c = r["summary"], r["comparison"]
            if c is None:
                continue
            sd = f"{s['std']:.6f}" if s["std"] is not None else "NA"
            p_text = f"{c['p']:.4g}" if c["p"] is not None else "NA"
            overview.append(
                f"| {r['dataset']} | {r['variant']} | {s['n']} | {s['mean']:.6f} +/- {sd} | "
                f"{r['control']} | {c['n']} | {c['delta']:+.6f} | {c['wins']}/{c['n']} | "
                f"{p_text} |"
            )
        overview.append("")
    overview += [
        "## Interpretation and Seed Corrections", "",
        "- KuaiRand Wide4+RareWeight4 is positive against its Loop control on canonical four seeds.",
        "- Width alone and weighting alone do not show the same gain; inspect their combination.",
        "- TAAC bypass+width remains three seeds, with no significant paired gain.",
        "- Taobao width-only gain from the earlier three seeds disappears after including 20262027.",
        "- MerRec/QK widening gives no demonstrated benefit; MerRec weighting is harmful.",
        "- Fully shared task towers strongly underperform independent towers on Taobao.",
        "- Strong preference-margin weights hurt; low weights give exploratory small gains.",
        "- These comparisons do not isolate recurrent parameter tying.",
        "",
        "Complete test logs show KuaiRand and Taobao improved arms already contain 20262027,",
        "contrary to the earlier YAML-only coverage check. TAAC wide+bypass still lacks it.",
        "Some controls at 20262027 are fresh legacy-family reruns, not frozen aligned runs;",
        "do not interchange control values across reports.",
        "",
        f"[Full seed scopes and per-task tables]({stem}.md)",
        f"[Paired statistics JSON]({stem}.json)",
        f"[Per-task means and SD CSV]({stem}_tasks.csv)",
        f"[Raw metrics and source logs CSV]({stem}_raw.csv)",
    ]
    (OUT / f"{stem}_OVERVIEW.md").write_text("\n".join(overview) + "\n")
    for suffix, rows in [("tasks", task_rows), ("raw", raw_rows)]:
        with (OUT / f"{stem}_{suffix}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Exported {len(table)} arms, {len(comparison_rows)} comparisons, "
          f"{len(task_rows)} task summaries; skipped {len(skipped)} incomplete-task logs")


if __name__ == "__main__":
    main()
