#!/usr/bin/env python3
"""Summarize local single-GPU Taobao results for all UniRank baselines."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("is_click", "cart", "fav", "buy")
MODELS = (
    "RankMixer",
    "OneTrans",
    "UniMixer",
    "HyFormer",
    "MixFormer",
    "TokenMixer",
    "HiFormer",
    "INFNet",
    "EST",
    "LONGER",
    "Zenith",
    "HeMix",
    "TokenFormer",
    "UltraHSTU",
    "SSR",
)


def select_log(model: str) -> tuple[Path, str]:
    if model == "RankMixer":
        return ROOT / "runs_local/taobao_phase1/RankMixer_Taobao_P1_Parallel.stdout.log", "compile"
    if model == "OneTrans":
        return ROOT / "runs_local/taobao_onetrans_multigpu/OneTrans_Taobao_P1_Parallel.stdout.log", "compile"
    if model == "TokenFormer":
        return ROOT / "runs_local/taobao_all_baselines/TokenFormer_Taobao_Action.retry.stdout.log", "eager"
    return ROOT / f"runs_local/taobao_all_baselines/{model}_Taobao_Action.stdout.log", "compile"


def parse(path: Path) -> dict:
    text = path.read_bytes().replace(b"\r", b"\n").replace(b"\0", b"").decode("utf-8", "ignore")
    parameter_match = re.search(r"Total number of parameters: (\d+)", text)
    metrics: dict[str, list[dict[str, float]]] = {}
    pattern = re.compile(
        r"\[Task: ([^]]+)\]\[Metrics\] "
        r"(?:AUC: ([0-9.]+) - logloss: ([0-9.]+)|logloss: ([0-9.]+) - AUC: ([0-9.]+))"
    )
    for match in pattern.finditer(text):
        if match.group(2):
            value = {"AUC": float(match.group(2)), "LogLoss": float(match.group(3))}
        else:
            value = {"AUC": float(match.group(5)), "LogLoss": float(match.group(4))}
        metrics.setdefault(match.group(1), []).append(value)
    return {
        "complete": "Training finished." in text and all(len(metrics.get(t, [])) >= 2 for t in TASKS),
        "parameters": int(parameter_match.group(1)) if parameter_match else None,
        "test": {task: metrics[task][-1] for task in TASKS if metrics.get(task)},
    }


def main():
    rows = []
    for model in MODELS:
        log, execution = select_log(model)
        result = parse(log)
        result.update(model=model, log=str(log), execution=execution)
        rows.append(result)

    rows.sort(key=lambda row: row.get("test", {}).get("buy", {}).get("AUC", -1), reverse=True)
    output_json = ROOT.parent / "reports/unirank_taobao_all_baselines_results.json"
    output_json.write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# Taobao all UniRank baselines — local protocol",
        "",
        "Date: 2026-09-23",
        "",
        "Protocol: one GPU per model, batch size 8192, one epoch, seed 20262027, same public Taobao blocked release and evaluation code.",
        "",
        "| Rank | Model | Params | Click AUC | Cart AUC | Fav AUC | Buy AUC | Buy LogLoss | Mode |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows, 1):
        test = row["test"]
        lines.append(
            f"| {rank} | {row['model']} | {row['parameters']:,} | "
            f"{test['is_click']['AUC']:.6f} | {test['cart']['AUC']:.6f} | "
            f"{test['fav']['AUC']:.6f} | {test['buy']['AUC']:.6f} | "
            f"{test['buy']['LogLoss']:.6f} | {row['execution']} |"
        )
    lines += [
        "",
        "TokenFormer uses eager mode because PyTorch 2.6 Inductor failed to lower `flex_attention_backward` on this runtime. Model, data, optimizer, batch, seed, and epoch settings are unchanged.",
    ]
    output_md = ROOT.parent / "reports/UNIRANK_TAOBAO_ALL_BASELINES_20260923.md"
    output_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not all(row["complete"] for row in rows):
        raise SystemExit("At least one baseline is incomplete")


if __name__ == "__main__":
    main()
