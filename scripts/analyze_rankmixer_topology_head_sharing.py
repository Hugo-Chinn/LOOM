#!/usr/bin/env python3
"""Summarize fine-grained topology-aware Head sharing on Taobao."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from analyze_rankmixer_head_sharing import parse_log


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "UniRank/runs_local/checkpoints_rankmixer_head_sharing/Taobao_Action"
JSON_OUT = ROOT / "reports/rankmixer_topology_head_sharing_results.json"
REPORT_OUT = ROOT / "reports/RANKMIXER_TOPOLOGY_HEAD_SHARING_20261009.md"
SEEDS = (20262027, 20262028, 20262029, 20262030)
TASKS = ("is_click", "cart", "fav", "buy")
MODES = (
    "Independent",
    "TopologyGrouped",
    "TopologySharedTrunk",
)
SCREEN_MODES = (
    "TopologySharedFirst",
    "TopologySharedTrunkAdapter",
)
LABELS = {
    "Independent": "Independent heads",
    "TopologyGrouped": "Cart/Fav shared full head",
    "TopologySharedFirst": "Cart/Fav shared first layer",
    "TopologySharedTrunk": "Cart/Fav shared trunk",
    "TopologySharedTrunkAdapter": "Shared trunk + task adapter",
}


def mean_std(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def build_summary():
    runs = {}
    for mode in MODES:
        runs[mode] = {}
        for seed in SEEDS:
            path = LOG_DIR / f"RankMixer_Branch_{mode}_Seed{seed}.log"
            if not path.exists():
                raise FileNotFoundError(path)
            runs[mode][str(seed)] = parse_log(path)

    aggregate = {}
    deltas = {}
    for mode in MODES:
        aggregate[mode] = {
            "parameters": runs[mode][str(SEEDS[0])]["parameters"],
            "tasks": {},
        }
        for task in TASKS:
            aggregate[mode]["tasks"][task] = {
                metric: mean_std([
                    runs[mode][str(seed)]["metrics"][task][metric]
                    for seed in SEEDS
                ])
                for metric in ("AUC", "logloss")
            }
        aggregate[mode]["macro_auc"] = mean_std([
            statistics.mean([
                runs[mode][str(seed)]["metrics"][task]["AUC"] for task in TASKS
            ])
            for seed in SEEDS
        ])

        if mode == "Independent":
            continue
        deltas[mode] = {}
        for task in TASKS:
            by_seed = {
                str(seed): runs[mode][str(seed)]["metrics"][task]["AUC"]
                - runs["Independent"][str(seed)]["metrics"][task]["AUC"]
                for seed in SEEDS
            }
            deltas[mode][task] = {
                "by_seed": by_seed,
                **mean_std(list(by_seed.values())),
                "min": min(by_seed.values()),
                "max": max(by_seed.values()),
                "positive_seeds": sum(value > 0 for value in by_seed.values()),
            }
    screens = {}
    screen_seed = SEEDS[0]
    for mode in SCREEN_MODES:
        path = LOG_DIR / f"RankMixer_Branch_{mode}_Seed{screen_seed}.log"
        if not path.exists():
            raise FileNotFoundError(path)
        run = parse_log(path)
        baseline = runs["Independent"][str(screen_seed)]
        screens[mode] = {
            "seed": screen_seed,
            "run": run,
            "auc_deltas_vs_independent": {
                task: run["metrics"][task]["AUC"]
                - baseline["metrics"][task]["AUC"]
                for task in TASKS
            },
        }

    return {
        "date": "2026-10-09",
        "dataset": "Taobao_Action",
        "seeds": list(SEEDS),
        "runs": runs,
        "aggregate": aggregate,
        "paired_auc_deltas_vs_independent": deltas,
        "single_seed_screens": screens,
    }


def fmt(value):
    return f"{value:.6f}"


def render(summary):
    aggregate = summary["aggregate"]
    deltas = summary["paired_auc_deltas_vs_independent"]
    base_dense = aggregate["Independent"]["parameters"]["dense"]
    grouped = deltas["TopologyGrouped"]
    trunk = deltas["TopologySharedTrunk"]
    screens = summary["single_seed_screens"]
    lines = [
        "# RankMixer topology-aware Head sharing",
        "",
        "日期：2026-10-09",
        "",
        "Taobao Branch-only Loop，四个固定 seeds。Click 与 Buy 始终使用完整独立 Head，"
        "只改变同层级 Cart/Fav 的共享方式。",
        "",
        "## 结论",
        "",
        "- Cart/Fav **共享完整 Head** 时，Buy AUC 基本无损（四 seed 平均 "
        f"`{grouped['buy']['mean']:+.6f}`），但 Fav 平均下降 "
        f"`{grouped['fav']['mean']:+.6f}`。",
        "- Cart/Fav **共享 trunk、保留独立输出层** 后，Fav 损失缩小到 "
        f"`{trunk['fav']['mean']:+.6f}`，Cart 反而平均提升 "
        f"`{trunk['cart']['mean']:+.6f}`；但 Buy 平均下降 "
        f"`{trunk['buy']['mean']:+.6f}`，仅 1/4 seeds 正向。",
        "- 因此独立输出层确实有必要，但还不足以得到 Buy 泛化提升。当前最终任务优先时，"
        "完整 topology-grouped 的 Buy 更稳；多任务均衡时，独立 Heads 仍是最佳默认。",
        "- grouped-first 和 grouped-trunk+adapter 在 seed20262027 分别使 Buy 下降 "
        f"`{screens['TopologySharedFirst']['auc_deltas_vs_independent']['buy']:+.6f}` 和 "
        f"`{screens['TopologySharedTrunkAdapter']['auc_deltas_vs_independent']['buy']:+.6f}`，"
        "未进入多 seed 确认。",
        "",
        "## 四 seed 结果",
        "",
        "| Head | Cart AUC | Fav AUC | Buy AUC | Buy std | Dense reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = aggregate[mode]
        reduction = 1.0 - item["parameters"]["dense"] / base_dense
        lines.append(
            f"| {LABELS[mode]} | {fmt(item['tasks']['cart']['AUC']['mean'])} | "
            f"{fmt(item['tasks']['fav']['AUC']['mean'])} | "
            f"{fmt(item['tasks']['buy']['AUC']['mean'])} | "
            f"{fmt(item['tasks']['buy']['AUC']['sample_std'])} | {reduction:.2%} |"
        )

    lines.extend([
        "",
        "## 相对独立 Heads 的配对 AUC 差值",
        "",
        "| Head | Cart | Fav | Buy | Buy positive seeds | Buy worst seed |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES[1:]:
        d = deltas[mode]
        lines.append(
            f"| {LABELS[mode]} | {fmt(d['cart']['mean'])} | {fmt(d['fav']['mean'])} | "
            f"{fmt(d['buy']['mean'])} | {d['buy']['positive_seeds']}/4 | "
            f"{fmt(d['buy']['min'])} |"
        )

    lines.extend([
        "",
        "Buy 的逐 seed 差值：",
        "",
        "| Head | 20262027 | 20262028 | 20262029 | 20262030 | Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES[1:]:
        buy = deltas[mode]["buy"]
        values = [fmt(buy["by_seed"][str(seed)]) for seed in SEEDS]
        lines.append(
            f"| {LABELS[mode]} | " + " | ".join(values) + f" | {fmt(buy['mean'])} |"
        )

    lines.extend([
        "",
        "## 单 seed 筛选（未进入多 seed 确认）",
        "",
        "| Head | Seed | Cart delta | Fav delta | Buy delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for mode in SCREEN_MODES:
        item = screens[mode]
        d = item["auc_deltas_vs_independent"]
        lines.append(
            f"| {LABELS[mode]} | {item['seed']} | {fmt(d['cart'])} | "
            f"{fmt(d['fav'])} | {fmt(d['buy'])} |"
        )

    lines.extend([
        "",
        "## 研究判断",
        "",
        "细粒度共享揭示了一个稳定权衡：共享最终 readout 会直接损害 Fav；恢复独立 readout "
        "可以修复 Fav，却没有稳定改善 Buy。说明 Head 参数共享不是当前 Loop 泛化提升的主要瓶颈，"
        "继续增加 Head 结构复杂度的优先级较低。后续更值得回到样本级 Loop 更新强度、"
        "任务梯度冲突或跨数据集自适应上。",
    ])

    lines.extend([
        "",
        "机器可读结果：`reports/rankmixer_topology_head_sharing_results.json`。",
        "",
    ])
    return "\n".join(lines)


def main():
    summary = build_summary()
    JSON_OUT.write_text(json.dumps(summary, indent=2) + "\n")
    REPORT_OUT.write_text(render(summary))
    print(JSON_OUT)
    print(REPORT_OUT)


if __name__ == "__main__":
    main()
