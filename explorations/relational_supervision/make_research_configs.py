"""Derive RelSup arm configs from a prepared historical Full config directory.

    python -m explorations.relational_supervision.make_research_configs \
        --source configs/local/taobao-full --output configs/local/taobao-relsup \
        --model-root /absolute/outputs/taobao-relsup

`--source` is the output of `research/prepare_native_config.py`. Every arm,
including `full`, copies the Full hyperparameters of the same seed and only
switches `model` to RankMixerRelSup and adds `relsup_*` keys.
"""
import argparse
import copy
import json
from pathlib import Path
import re

import yaml

from .arms import PHASE1, expand

SEED = re.compile(r"^(?P<prefix>.+)_S(?P<seed>\d+)$")


def build(source, output, model_root, arms, log_every=200):
    models = yaml.safe_load((source / "model_config.yaml").read_text())
    by_seed, prefixes = {}, set()
    for expid, params in models.items():
        if expid == "Base":
            continue
        match = SEED.match(expid)
        if not match:
            raise ValueError(f"experiment id without _S<seed> suffix: {expid}")
        if params.get("model") != "RankMixer" or params.get("model_type") != "looped":
            raise ValueError(f"{expid}: source must be a looped RankMixer config")
        if any(key.startswith("relsup_") for key in params):
            raise ValueError(f"{expid}: source already contains relsup keys")
        prefixes.add(match["prefix"])
        by_seed[int(match["seed"])] = params
    if len(prefixes) != 1 or not by_seed:
        raise ValueError("source must hold exactly one Full family across seeds")
    if output.exists():
        raise ValueError("config output already exists; refusing overwrite")
    resolved = {"Base": copy.deepcopy(models["Base"])}
    if model_root is not None:
        resolved["Base"]["model_root"] = str(model_root.resolve())
    expids = []
    for arm, overrides in arms.items():
        for seed in sorted(by_seed):
            params = copy.deepcopy(by_seed[seed])
            params["model"] = "RankMixerRelSup"
            params["relsup_log_every"] = log_every
            params.update({f"relsup_{key}": value for key, value in overrides.items()})
            expids.append(f"{arm}_S{seed}")
            resolved[expids[-1]] = params
    output.mkdir(parents=True)
    (output / "model_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    (output / "dataset_config.yaml").write_text(
        (source / "dataset_config.yaml").read_text())
    (output / "expids.txt").write_text("\n".join(expids) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "source": str(source), "source_family": prefixes.pop(),
        "seeds": sorted(by_seed), "arms": arms,
        "changes": ["model -> RankMixerRelSup", "relsup_* keys", "checkpoint root"],
    }, indent=2))
    return expids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--arms", nargs="+", default=list(PHASE1))
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--max-runs", type=int, default=20)
    args = parser.parse_args()
    selected = expand(args.arms, args.scales, args.margin)
    created = build(args.source, args.output, args.model_root, selected)
    if len(created) > args.max_runs:
        raise SystemExit(f"{len(created)} runs exceeds budget {args.max_runs}; "
                         f"delete {args.output} and raise --max-runs deliberately")
    print("\n".join(created))
