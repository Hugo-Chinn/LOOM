#!/usr/bin/env python3
"""Summarize the controlled Taobao RankMixer head-sharing ablation."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = (
    ROOT / "UniRank/runs_local/checkpoints_rankmixer_head_sharing/Taobao_Action"
)
DEFAULT_JSON = ROOT / "reports/rankmixer_head_sharing_taobao_results.json"
DEFAULT_REPORT = ROOT / "reports/RANKMIXER_HEAD_SHARING_TAOBAO_20261009.md"
TASKS = ("is_click", "cart", "fav", "buy")
SEEDS = (20262027, 20262028, 20262029, 20262030)
MODES = (
    "Independent",
    "SharedFirst",
    "TopologyGrouped",
    "SharedTrunk",
    "Shared",
)
MODE_LABELS = {
    "Independent": "Independent heads",
    "SharedFirst": "Shared first layer",
    "TopologyGrouped": "Topology-grouped heads",
    "SharedTrunk": "Shared trunk + task outputs",
    "Shared": "Fully shared head",
}


def parse_log(path: Path):
    text = path.read_text()
    test_marker = "******** Test evaluation ********"
    if test_marker not in text:
        raise ValueError(f"Missing test evaluation in {path}")
    test_text = text.rsplit(test_marker, 1)[1]
    metrics = {}
    pattern = re.compile(r"\[Task: ([^]]+)\]\[Metrics\] (.+)")
    metric_pattern = re.compile(r"(AUC|logloss): ([0-9.]+)")
    for line in test_text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        task = match.group(1)
        metrics[task] = {
            key: float(value) for key, value in metric_pattern.findall(match.group(2))
        }
    missing = set(TASKS) - set(metrics)
    if missing:
        raise ValueError(f"Missing test tasks {sorted(missing)} in {path}")

    total_matches = re.findall(r"Total number of parameters: ([0-9]+)\.", text)
    dense_matches = re.findall(r"Number of dense parameters: ([0-9]+)\.", text)
    if not total_matches or not dense_matches:
        raise ValueError(f"Missing parameter counts in {path}")
    return {
        "metrics": metrics,
        "parameters": {
            "total": int(total_matches[-1]),
            "dense": int(dense_matches[-1]),
        },
    }


def mean_std(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def build_summary(log_dir: Path):
    runs = {}
    for mode in MODES:
        runs[mode] = {}
        for seed in SEEDS:
            path = log_dir / f"RankMixer_Branch_{mode}_Seed{seed}.log"
            if not path.exists():
                raise FileNotFoundError(path)
            runs[mode][str(seed)] = parse_log(path)

    aggregate = {}
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
                runs[mode][str(seed)]["metrics"][task]["AUC"]
                for task in TASKS
            ])
            for seed in SEEDS
        ])

    paired_deltas = {}
    for mode in MODES[1:]:
        paired_deltas[mode] = {}
        for task in TASKS:
            paired_deltas[mode][task] = {}
            for metric in ("AUC", "logloss"):
                deltas = {
                    str(seed): (
                        runs[mode][str(seed)]["metrics"][task][metric]
                        - runs["Independent"][str(seed)]["metrics"][task][metric]
                    )
                    for seed in SEEDS
                }
                paired_deltas[mode][task][metric] = {
                    "by_seed": deltas,
                    **mean_std(list(deltas.values())),
                    "min": min(deltas.values()),
                    "max": max(deltas.values()),
                    "positive_seeds": sum(value > 0 for value in deltas.values()),
                }
        macro_deltas = {
            str(seed): statistics.mean([
                runs[mode][str(seed)]["metrics"][task]["AUC"]
                - runs["Independent"][str(seed)]["metrics"][task]["AUC"]
                for task in TASKS
            ])
            for seed in SEEDS
        }
        paired_deltas[mode]["macro_auc"] = {
            "by_seed": macro_deltas,
            **mean_std(list(macro_deltas.values())),
            "min": min(macro_deltas.values()),
            "max": max(macro_deltas.values()),
            "positive_seeds": sum(value > 0 for value in macro_deltas.values()),
        }

    return {
        "date": "2026-10-09",
        "dataset": "Taobao_Action",
        "backbone": "RankMixer",
        "architecture": "Branch-only Loop + deep progressive supervision",
        "seeds": list(SEEDS),
        "runs": runs,
        "aggregate": aggregate,
        "paired_deltas_vs_independent": paired_deltas,
    }


def fmt(value):
    return f"{value:.6f}"


def render_report(summary):
    aggregate = summary["aggregate"]
    deltas = summary["paired_deltas_vs_independent"]
    independent_params = aggregate["Independent"]["parameters"]
    lines = [
        "# RankMixer task-head sharing ablation on Taobao",
        "",
        "日期：2026-10-09",
        "",
        "## 结论",
        "",
        "- **完全共享 head 明确失败**：Buy AUC 四 seed 平均下降 "
        f"`{abs(deltas['Shared']['buy']['AUC']['mean']):.6f}`，4/4 seeds 下降。",
        "- **全任务按层共享也没有稳定收益**：`shared_first` 和 `shared_trunk` 的 Buy AUC "
        f"分别平均下降 `{abs(deltas['SharedFirst']['buy']['AUC']['mean']):.6f}` 和 "
        f"`{abs(deltas['SharedTrunk']['buy']['AUC']['mean']):.6f}`；其中 `shared_trunk` 4/4 seeds 下降。",
        "- **按反馈拓扑选择性共享最有希望**：仅让同层级 Cart/Fav 共用 head，Buy AUC "
        f"平均只变化 `{deltas['TopologyGrouped']['buy']['AUC']['mean']:+.6f}`，"
        "2/4 seeds 正向，并将 Buy seed 标准差降低约一半。",
        "- 但 topology-grouped 会使 Fav AUC 平均下降 "
        f"`{abs(deltas['TopologyGrouped']['fav']['AUC']['mean']):.6f}`，因此它是最终 Buy 优先的"
        "参数共享方案，不是所有任务无损方案。",
        "",
        "## 实验控制",
        "",
        "所有实验使用相同的 Taobao Branch-only RankMixer Loop、behavior embedding、"
        "deep progressive supervision、训练数据、1 epoch 预算和 seeds。唯一变化是 head：",
        "",
        "1. `independent`：每任务独立 `256→128→64→1`；",
        "2. `shared_first`：共享 `256→128`，每任务独立 `128→64→1`；",
        "3. `topology_grouped`：Click、Buy 独立，Cart/Fav 共用完整 head；",
        "4. `shared_trunk`：共享 `256→128→64`，每任务独立 `64→1`；",
        "5. `shared`：所有任务共享完整 `256→128→64→1`。",
        "",
        "## 四 seed 测试集结果",
        "",
        "| Head | Click AUC | Cart AUC | Fav AUC | Buy AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        cells = []
        for task in TASKS:
            stats = aggregate[mode]["tasks"][task]["AUC"]
            cells.append(f"{fmt(stats['mean'])} ± {fmt(stats['sample_std'])}")
        lines.append(f"| {MODE_LABELS[mode]} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "| Head | Macro AUC | Macro delta | Buy std |",
        "|---|---:|---:|---:|",
    ])
    for mode in MODES:
        macro = aggregate[mode]["macro_auc"]
        macro_delta = 0.0 if mode == "Independent" else deltas[mode]["macro_auc"]["mean"]
        buy_std = aggregate[mode]["tasks"]["buy"]["AUC"]["sample_std"]
        lines.append(
            f"| {MODE_LABELS[mode]} | {fmt(macro['mean'])} ± {fmt(macro['sample_std'])} | "
            f"{fmt(macro_delta)} | {fmt(buy_std)} |"
        )

    lines.extend([
        "",
        "### 相对独立 head 的配对 AUC 差值",
        "",
        "| Variant | Click | Cart | Fav | Buy |",
        "|---|---:|---:|---:|---:|",
    ])
    for mode in MODES[1:]:
        cells = [fmt(deltas[mode][task]["AUC"]["mean"]) for task in TASKS]
        lines.append(f"| {MODE_LABELS[mode]} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "Buy AUC 的逐 seed 配对差值：",
        "",
        "| Variant | 20262027 | 20262028 | 20262029 | 20262030 | Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES[1:]:
        buy = deltas[mode]["buy"]["AUC"]
        values = [fmt(buy["by_seed"][str(seed)]) for seed in summary["seeds"]]
        lines.append(
            f"| {MODE_LABELS[mode]} | " + " | ".join(values) + f" | {fmt(buy['mean'])} |"
        )

    lines.extend([
        "",
        "## 参数量",
        "",
        "| Head | Total params | Dense params | Dense reduction |",
        "|---|---:|---:|---:|",
    ])
    for mode in MODES:
        params = aggregate[mode]["parameters"]
        dense_reduction = 1.0 - params["dense"] / independent_params["dense"]
        lines.append(
            f"| {MODE_LABELS[mode]} | {params['total']:,} | {params['dense']:,} | "
            f"{dense_reduction:.2%} |"
        )

    lines.extend([
        "",
        "## 解释",
        "",
        "`shared_first` 在两个 seed 上下降、两个 seed 上提升，说明全局共享第一层对初始化较敏感；"
        "`shared_trunk` 虽然保留了独立输出层，但 4/4 seeds 的 Buy 都下降，说明独立 readout 不是"
        "唯一需求，中间特征变换也需要一定任务专属容量。",
        "",
        "`topology_grouped` 的 Buy 均值与独立 heads 几乎相同，同时方差从 "
        f"`{aggregate['Independent']['tasks']['buy']['AUC']['sample_std']:.6f}` 降到 "
        f"`{aggregate['TopologyGrouped']['tasks']['buy']['AUC']['sample_std']:.6f}`。但 Cart/Fav 共用同一排序函数"
        "直接损害 Fav，因此更合理的后续方向不是继续共享完整 head，而是让同层级任务共享低层 trunk、"
        "同时保留各自的输出层或小型 task adapter。",
        "",
        "四个 seeds 对小效应仍不足以提供强显著性证据；这里应把 topology-grouped 表述为"
        "“Buy 基本无损且方差更低”，不能声称它显著优于独立 heads。",
        "",
        "机器可读结果：`reports/rankmixer_head_sharing_taobao_results.json`。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    summary = build_summary(args.log_dir)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n")
    args.report.write_text(render_report(summary))
    print(args.json)
    print(args.report)


if __name__ == "__main__":
    main()
