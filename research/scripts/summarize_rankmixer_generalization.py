#!/usr/bin/env python3
"""Summarize RankMixer loop generalization across non-Taobao datasets."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_local/rankmixer_generalization"
PAIRS = {
    "MerRec": ("RankMixer_MerRec_Baseline", "RankMixer_MerRec_GatedLoopDeep"),
    "QK-Video": ("RankMixer_QKVideo_Baseline", "RankMixer_QKVideo_LoopDeep"),
    "KuaiRand": ("RankMixer_KuaiRand_Baseline", "RankMixer_KuaiRand_LoopDeep"),
}


def parse(expid):
    path = RUNS / f"{expid}.stdout.log"
    text = path.read_bytes().replace(b"\r", b"\n").decode("utf-8", "ignore")
    params = re.search(r"Total number of parameters: (\d+)", text)
    metrics = {}
    pattern = re.compile(
        r"\[Task: ([^]]+)\]\[Metrics\] "
        r"(?:AUC: ([0-9.]+) - logloss: ([0-9.]+)|logloss: ([0-9.]+) - AUC: ([0-9.]+))"
    )
    for match in pattern.finditer(text):
        metrics[match.group(1)] = {
            "AUC": float(match.group(5) or match.group(2)),
            "LogLoss": float(match.group(4) or match.group(3)),
        }
    return {
        "expid": expid,
        "parameters": None if params is None else int(params.group(1)),
        "metrics": metrics,
        "training_finished": "Training finished." in text,
        "test_started": "******** Test evaluation ********" in text,
    }


def main():
    result = {}
    for dataset, (baseline_id, loop_id) in PAIRS.items():
        baseline = parse(baseline_id)
        looped = parse(loop_id)
        deltas = {
            task: {
                "AUC": looped["metrics"][task]["AUC"] - baseline["metrics"][task]["AUC"],
                "LogLoss": looped["metrics"][task]["LogLoss"] - baseline["metrics"][task]["LogLoss"],
            }
            for task in baseline["metrics"]
        }
        result[dataset] = {"baseline": baseline, "loop": looped, "delta": deltas}
    failure = RUNS / "RankMixer_MerRec_LoopDeep.stdout.log"
    text = failure.read_bytes().replace(b"\r", b"\n").decode("utf-8", "ignore")
    averages = [float(x) for x in re.findall(r"avg_loss=([0-9.eE+-]+)", text)]
    result["MerRec_ungated_failure"] = {
        "stopped_at_progress_percent": 13,
        "last_average_loss": averages[-1],
        "reason": "ungated five-stage loop remained near loss 112 and did not learn",
    }
    output = ROOT.parent / "reports/unirank_rankmixer_generalization_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
