#!/usr/bin/env python3
"""Export the aligned four-seed results without mixing in legacy studies."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
DATASETS = [
    "TencentGR_10M_Action", "QK_Video_Action", "KuaiRand_Video_Action",
    "MerRec_Action", "Taobao_Action",
]
NAMES = ["TencentGR", "QK", "KuaiRand", "MerRec", "Taobao"]
VARIANTS = [
    "ParallelIndependentTower", "MMoE", "PLE", "STEM", "LoopMT",
    "SingleStep", "UntiedDeep", "LoopNoAux", "LoopFinal", "LoopNoStage",
]


def main():
    payloads = {
        name: json.loads((OUT / f"{name.lower()}_mtl_aligned_results.json").read_text())
        for name in ("RankMixer", "OneTrans")
    }
    stem = "MTL_ALL_RESULTS_20261019"
    lines = [
        "# MTL Results, 2026-10-19", "",
        "Scope: latest aligned RankMixer and OneTrans studies, five datasets.",
        "Seeds: 20262027, 20262028, 20262029, 20262030.",
        "The first seed also participated in screening; these are not four held-out seeds.",
        "Macro AUC is the unweighted mean of test task AUCs. SD is sample SD across seeds.",
        "Delta is an absolute AUC difference, not a relative percentage.",
        "All p-values are two-sided paired t-tests, unadjusted for multiple comparisons.",
        "",
        "## Completion", "",
        "| Backbone | Complete | Pending eligible | Reference-blocked | Snapshot |",
        "|---|---:|---:|---:|---|",
    ]
    for name, payload in payloads.items():
        lines.append(
            f"| {name} | {payload['completed']} | {len(payload['pending'])} | "
            f"{len(payload['reference_blocked'])} | {payload['snapshot'][:12]} |"
        )
    lines += [
        "", "520 eligible runs completed. 120 reference-blocked runs are not completed.",
        "SharedBottom with independent towers is the parallel alias: 40 entries are skipped.",
        "OneRank: exact generative objective/reference not verified.",
        "MoME: weight-mask formulation not verified. ResFlow: paper fusion not verified.",
        "", "## Architecture Boundaries", "",
        "Historical RankMixer Loop replaces the interaction stack with recurrent refinement.",
        "The new head study retains the full backbone and adds a post-backbone refinement head.",
        "These are different architectures; their results must not be pooled.",
        "STEM is a head-adapted implementation with private representation adapters,",
        "not a reproduction of original task-specific embedding tables.",
        "",
    ]
    summary_rows, seed_rows, task_rows, task_summary_rows = [], [], [], []
    task_lines = [
        "# Per-Task Four-Seed Results", "",
        "Seeds: 20262027, 20262028, 20262029, 20262030.",
        "Each cell is test AUC mean +/- sample SD across the four seeds (ddof=1).",
        "SD is not a standard error or confidence interval.",
        "Historical replication and fixed-backbone heads are separate studies.",
        "SharedBottom is the parallel alias; STEM is head-adapted.",
        "OneRank, MoME and ResFlow are reference-blocked and have no results here.",
        "The CSV also includes per-task logloss mean and sample SD.", "",
    ]
    for name, payload in payloads.items():
        for study, variants in [
            ("historical_replication", ["Parallel", "LinearDeep", "Finalist"]
             if name == "RankMixer" else ["Parallel", "FiLM", "QVLoRA"]),
            ("fixed_backbone_head", VARIANTS),
        ]:
            rows = {
                (r["dataset"], r["variant"]): r
                for r in payload["rows"] if r["study"] == study
            }
            lines += [
                f"## {name}: {study}", "",
                "Mean test Macro AUC (4 seeds); bold marks the highest mean within this table.",
                "", "| Variant | " + " | ".join(NAMES) + " |",
                "|---|" + "---:|" * len(DATASETS),
            ]
            for variant in variants:
                cells = []
                for ds in DATASETS:
                    row = rows[(ds, variant)]
                    cell = f"{row['macro_mean']:.6f}"
                    if row["macro_mean"] == max(rows[(ds, v)]["macro_mean"] for v in variants):
                        cell = f"**{cell}**"
                    cells.append(cell)
                lines.append(f"| {variant} | " + " | ".join(cells) + " |")
            lines += [
                "", "| Dataset | Variant | N | Mean | SD | Delta vs parallel | Wins | p |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
            for ds, short in zip(DATASETS, NAMES):
                for variant in variants:
                    r = rows[(ds, variant)]
                    delta = f"{r['delta']:+.6f}" if "delta" in r else "-"
                    p = f"{r['p_value']:.6g}" if r.get("p_value") is not None else "-"
                    lines.append(
                        f"| {short} | {variant} | {r['n']} | {r['macro_mean']:.6f} | "
                        f"{r['macro_std']:.6f} | {delta} | {r.get('wins', '-')} | {p} |"
                    )
                    summary_rows.append({
                        "backbone": name, "study": study, "dataset": ds, "variant": variant,
                        "n": r["n"], "macro_mean": r["macro_mean"], "macro_std": r["macro_std"],
                        "delta_vs_parallel": r.get("delta"), "wins": r.get("wins"),
                        "p_value_unadjusted": r.get("p_value"),
                    })
                    for seed, metrics in sorted(r["per_seed"].items()):
                        seed_rows.append({
                            "backbone": name, "study": study, "dataset": ds,
                            "variant": variant, "seed": seed,
                            "macro_auc": mean(m["auc"] for m in metrics.values()),
                            "macro_logloss": mean(m["logloss"] for m in metrics.values()),
                        })
                        for task, values in metrics.items():
                            task_rows.append({
                                "backbone": name, "study": study, "dataset": ds,
                                "variant": variant, "seed": seed, "task": task, **values,
                            })
            study_task_lines = [f"## {name}: {study} Per-Task AUC", ""]
            for ds in DATASETS:
                task_names = list(next(iter(rows[(ds, variants[0])]["per_seed"].values())))
                study_task_lines += [
                    f"### {ds}", "",
                    "| Variant | Seeds | " + " | ".join(task_names) + " |",
                    "|---|---:|" + "---:|" * len(task_names),
                ]
                for variant in variants:
                    r = rows[(ds, variant)]
                    seed_metrics = list(r["per_seed"].values())
                    if len(seed_metrics) != 4 or any(
                        set(metrics) != set(task_names) for metrics in seed_metrics
                    ):
                        raise ValueError(f"Incomplete task/seed coverage: {name}/{ds}/{variant}")
                    cells = []
                    for task in task_names:
                        aucs = [metrics[task]["auc"] for metrics in seed_metrics]
                        losses = [metrics[task]["logloss"] for metrics in seed_metrics]
                        auc_mean, auc_sd = mean(aucs), stdev(aucs)
                        task_summary_rows.append({
                            "backbone": name, "study": study, "dataset": ds,
                            "variant": variant, "task": task, "n": len(seed_metrics),
                            "auc_mean": auc_mean, "auc_std": auc_sd,
                            "logloss_mean": mean(losses), "logloss_std": stdev(losses),
                        })
                        cells.append(f"{auc_mean:.6f} +/- {auc_sd:.6f}")
                    study_task_lines.append(
                        f"| {variant} | {len(seed_metrics)}/4 | " + " | ".join(cells) + " |"
                    )
                study_task_lines.append("")
            lines.extend([""] + study_task_lines)
            task_lines.extend(study_task_lines)
        lines += [
            f"## {name}: Tied LoopMT vs UntiedDeep", "",
            "| Dataset | Delta | Unadjusted paired p |",
            "|---|---:|---:|",
        ]
        for r in payload["primary_tied_vs_untied"]:
            lines.append(f"| {r['dataset']} | {r['delta']:+.6f} | {r['p_value']:.6g} |")
        lines.append("")
    lines += [
        "## Interpretation", "",
        "Original RankMixer finalists improve the mean over parallel on four of five datasets.",
        "Taobao, QK and MerRec have unadjusted p<0.05; KuaiRand declines.",
        "This does not isolate parameter tying from topology, gates or preloop choices.",
        "New post-backbone LoopMT heads do not show overall superiority.",
        "No tied-vs-untied comparison reaches unadjusted p<0.05.",
        "", "## Legacy Results (Separate Protocols)", "",
        "Earlier runs are retained separately, not pooled with aligned results:",
        "- [Earlier experiment overview](EXPERIMENT_SUMMARY_20261012.md)",
        "- [Full earlier local inventory](LOCAL_EXPERIMENT_INVENTORY.md)",
        "- [Earlier inventory JSON](local_experiment_inventory.json)",
        "- [TencentGR six-seed study](RANKMIXER_TAAC_SIX_SEED_20261013.md)",
        "- [OneTrans all-dataset study](ONETRANS_ALL_DATASETS_MULTISEED_20261009.md)",
        "- [OneTrans NS/S-state study](ONETRANS_NS_S_STATE_MULTISEED_20261009.md)",
        "- [OneTrans targeted follow-up](ONETRANS_TARGETED_FOLLOWUP_MULTISEED_20261009.md)",
        "Earlier MTL_BASELINES_FOUR_SEED.md is confounded/obsolete and is not used here.",
        "", "## Machine-Readable Exports", "",
        f"- [{stem}_summary.csv]({stem}_summary.csv): all 130 variant/dataset groups.",
        f"- [{stem}_seeds.csv]({stem}_seeds.csv): all 520 seed-level records.",
        f"- [{stem}_tasks.csv]({stem}_tasks.csv): per-task test AUC and logloss for every run.",
        f"- [{stem}_task_summary.csv]({stem}_task_summary.csv): per-task four-seed means and sample SDs.",
        f"- [{stem}_per_task.md]({stem}_per_task.md): standalone per-task AUC tables.",
    ]
    (OUT / f"{stem}.md").write_text("\n".join(lines) + "\n")
    (OUT / f"{stem}_per_task.md").write_text("\n".join(task_lines) + "\n")
    for suffix, records in [
        ("summary", summary_rows), ("seeds", seed_rows), ("tasks", task_rows),
        ("task_summary", task_summary_rows),
    ]:
        with (OUT / f"{stem}_{suffix}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f"Exported {len(summary_rows)} groups, {len(seed_rows)} runs, "
          f"{len(task_rows)} task records to {OUT / (stem + '.md')}")


if __name__ == "__main__":
    main()
