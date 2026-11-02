"""Bounded, validation-only ablation selection; preserve per-task test outputs."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import yaml
from .config import LOOMConfig
from .core import LOOM
from .report import summarize


VARIANTS = {
    "full": {},
    "no_bcu": {"behavior_conditioning": False},
    "no_mas": {"supervision": "progressive", "auxiliary_loss_weight": 0.0},
    "no_cbm": {"parent_fusion": "first"},
    "untied": {"shared_update": False},
    "depth_embedding": {"depth_conditioning": True},
    "wide4": {"expansion_factor": 4},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS),
                        default=["full", "no_bcu", "no_mas", "no_cbm", "untied"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20262027, 20262028, 20262029, 20262030])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-runs", type=int, default=20)
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants) or len(set(args.seeds)) != len(args.seeds):
        parser.error("variants and seeds must be unique")
    count = len(args.variants) * len(args.seeds)
    if count > args.max_runs:
        parser.error(f"{count} runs exceeds budget {args.max_runs}")
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    base = yaml.safe_load(args.config.read_text())
    core = LOOM(LOOMConfig(**base["model"]))
    if "no_cbm" in args.variants and not any(len(p) > 1 for p in core.parents):
        parser.error("no_cbm is N/A: this graph has no multi-parent merge")
    args.output.mkdir(parents=True)
    scores = {}
    for variant in args.variants:
        document = copy.deepcopy(base)
        document["model"].update(VARIANTS[variant])
        config_path = args.output / f"{variant}.yaml"
        config_path.write_text(yaml.safe_dump(document, sort_keys=False))
        scores[variant] = []
        for seed in args.seeds:
            out = args.output / variant / f"S{seed}"
            subprocess.run([sys.executable, "-m", "loom.cli", "train", "--config",
                            str(config_path), "--data", str(args.data), "--output", str(out),
                            "--seed", str(seed), "--device", args.device], check=True)
            result = json.loads((out / "metrics.json").read_text())
            epoch = result["selected_epoch"] - 1
            aucs = [m["auc"] for m in result["history"][epoch]["validation"].values()
                    if m["auc"] is not None]
            scores[variant].append(sum(aucs) / len(aucs))
    mean_scores = {v: sum(s) / len(s) for v, s in scores.items()}
    (args.output / "validation_selection.json").write_text(json.dumps({
        "validation_mean_auc": mean_scores,
        "selected_variant": max(mean_scores, key=mean_scores.get),
        "seeds": args.seeds,
        "warning": "Selection uses validation only. Test metrics remain per task in each run.",
    }, indent=2))
    summarize(args.output)


if __name__ == "__main__":
    main()
