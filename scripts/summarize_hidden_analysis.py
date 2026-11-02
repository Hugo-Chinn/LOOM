#!/usr/bin/env python3
"""Join hidden-state diagnostics with loop AUC changes."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ("RankMixer", "OneTrans", "EST", "HeMix", "UltraHSTU")


def load_hidden(model, variant):
    name = f"Hidden_{model}_{'Baseline' if variant == 'baseline' else 'Loop'}.json"
    return json.loads((ROOT / "runs_local/hidden_analysis" / name).read_text())


def main():
    loop_results = json.loads(
        (ROOT.parent / "reports/unirank_taobao_cross_backbone_loop_results.json").read_text()
    )
    deep = {row["backbone"]: row for row in loop_results if row["variant"] == "deep"}
    rows = []
    for model in BACKBONES:
        baseline = load_hidden(model, "baseline")
        loop = load_hidden(model, "loop")
        base_buy = baseline["task_stats"]["buy"]
        loop_buy = loop["task_stats"]["buy"]
        adjacent = loop["adjacent_progression"]
        rows.append({
            "backbone": model,
            "head_dimension": loop_buy["dimension"],
            "buy_auc_delta": deep[model]["buy_auc_delta"],
            "parameter_reduction": deep[model]["parameter_reduction"],
            "baseline_buy_norm": base_buy["norm_mean"],
            "loop_buy_norm": loop_buy["norm_mean"],
            "norm_ratio": loop_buy["norm_mean"] / base_buy["norm_mean"],
            "baseline_effective_rank": base_buy["effective_rank"],
            "loop_effective_rank": loop_buy["effective_rank"],
            "rank_ratio": loop_buy["effective_rank"] / base_buy["effective_rank"],
            "baseline_top1_fraction": base_buy["top1_variance_fraction"],
            "loop_top1_fraction": loop_buy["top1_variance_fraction"],
            "baseline_fisher": base_buy["fisher_ratio_projected"],
            "loop_fisher": loop_buy["fisher_ratio_projected"],
            "fisher_ratio": loop_buy["fisher_ratio_projected"] / base_buy["fisher_ratio_projected"],
            "click_cart_cosine": adjacent["is_click->cart"]["cosine_mean"],
            "cart_fav_cosine": adjacent["cart->fav"]["cosine_mean"],
            "fav_buy_cosine": adjacent["fav->buy"]["cosine_mean"],
            "click_cart_delta": adjacent["is_click->cart"]["normalized_delta_mean"],
            "cart_fav_delta": adjacent["cart->fav"]["normalized_delta_mean"],
            "fav_buy_delta": adjacent["fav->buy"]["normalized_delta_mean"],
        })
    output = ROOT.parent / "reports/unirank_taobao_hidden_analysis_summary.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(
            row["backbone"],
            f"auc={row['buy_auc_delta']:+.6f}",
            f"norm_x={row['norm_ratio']:.2f}",
            f"rank_x={row['rank_ratio']:.2f}",
            f"top1={row['loop_top1_fraction']:.3f}",
            f"fisher_x={row['fisher_ratio']:.2f}",
            f"last_delta={row['fav_buy_delta']:.3f}",
            f"last_cos={row['fav_buy_cosine']:.3f}",
        )


if __name__ == "__main__":
    main()
