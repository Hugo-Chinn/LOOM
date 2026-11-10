"""Derive protocol-conforming RelSup configs from a prepared Full config.

    # phase 1: control + one arm per term, development seeds (20 runs)
    python -m explorations.relational_supervision.make_research_configs --phase 1 \
        --source configs/local/taobao-full --output configs/local/taobao-relsup-p1 \
        --model-root /absolute/outputs/taobao-relsup
    # phase 2: remaining grid points of the phase-1 survivors
    ... --phase 2 --selection runs_local/relsup-taobao/phase1/selection.json \
        --output configs/local/taobao-relsup-p2 --max-runs 80
    # phase 3: combination of the frozen settings; confirm: fresh seeds
    ... --phase 3 | confirm --selection runs_local/relsup-taobao/phase2/selection.json

`--source` is the output of `research/prepare_native_config.py`. Every arm
copies the Full hyperparameters of its seed; only `model`, `relsup_*`, the
checkpoint root and (for confirmation) `seed` change. Arms, grids and seeds
come from `arms.py`; anything else is refused. `--model-root` must be the same
absolute directory in every phase so later phases reuse the `full` logs.
"""
import argparse
import copy
import json
from pathlib import Path
import re

import yaml

from . import arms as A
from .environment import snapshot
from .structure import relations, topology_parents

SEED = re.compile(r"^(?P<prefix>.+)_S(?P<seed>\d+)$")


def load_source(source):
    models = yaml.safe_load((source / "model_config.yaml").read_text())
    datasets = yaml.safe_load((source / "dataset_config.yaml").read_text())
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
        if params.get("seed") != int(match["seed"]):
            raise ValueError(f"{expid}: seed field differs from experiment id")
        monitor = params.get("monitor")
        if not isinstance(monitor, dict) or not {"AUC", "logloss"} <= set(monitor) \
                or monitor["logloss"] != 0:
            raise ValueError(f"{expid}: monitor must log AUC and logloss (weight 0) so "
                             "validation logloss is available without changing selection")
        prefixes.add(match["prefix"])
        by_seed[int(match["seed"])] = params
    if len(prefixes) != 1:
        raise ValueError("source must hold exactly one Full family")
    if set(by_seed) != set(A.DEV_SEEDS):
        raise ValueError(f"source seeds {sorted(by_seed)} are not the development "
                         f"seed set {list(A.DEV_SEEDS)}")
    reference = {k: v for k, v in by_seed[A.DEV_SEEDS[0]].items() if k != "seed"}
    for seed, params in by_seed.items():
        if {k: v for k, v in params.items() if k != "seed"} != reference:
            raise ValueError(f"seed {seed} differs from other seeds beyond `seed`")
    dataset_id = reference["dataset_id"]
    tasks = [column["name"] for column in datasets[dataset_id]["label_col"]]
    topology = reference.get("loop_topology", "linear")
    graph = reference.get("behavior_graph")
    parents = topology_parents(
        topology, len(tasks),
        [[tasks.index(p) for p in graph[t]] for t in tasks] if topology == "dag" else None)
    return models, by_seed, prefixes.pop(), tasks, relations(parents)


def select_arms(phase, selection):
    if phase == "1":
        return A.phase1(), A.DEV_SEEDS
    if selection is None:
        raise ValueError(f"phase {phase} requires --selection from collect_research")
    if selection.get("protocol") != A.protocol_constants():
        raise ValueError("selection was made under different protocol constants")
    if phase == "2":
        if selection.get("stage") != "phase1":
            raise ValueError("phase 2 needs a phase1 selection.json")
        return A.phase2(selection["survivors"]), A.DEV_SEEDS
    if selection.get("stage") != "phase2":
        raise ValueError(f"phase {phase} needs a phase2 selection.json (frozen settings)")
    frozen = selection["frozen"]
    if phase == "3":
        return A.frozen_arms(frozen, combine=True), A.DEV_SEEDS
    chosen = {A.BASELINE: A.phase1()[A.BASELINE]}
    chosen.update(A.frozen_arms(frozen))
    if len(frozen) > 1:
        chosen.update(A.frozen_arms(frozen, combine=True))
    return chosen, A.CONFIRM_SEEDS


def build(source, output, model_root, phase, selection=None, max_runs=20, log_every=200):
    if not model_root.is_absolute():
        raise ValueError("--model-root must be absolute")
    models, by_seed, family, tasks, rel = load_source(source)
    chosen, seeds = select_arms(phase, selection)
    if not chosen:
        raise ValueError("nothing to run: no surviving term")
    if len(chosen) * len(seeds) > max_runs:
        raise ValueError(f"{len(chosen) * len(seeds)} runs exceeds budget {max_runs}; "
                         "raise --max-runs deliberately")
    if output.exists():
        raise ValueError("config output already exists; refusing overwrite")
    resolved = {"Base": copy.deepcopy(models["Base"])}
    resolved["Base"]["model_root"] = str(model_root)
    template = by_seed[A.DEV_SEEDS[0]]
    expids, described = [], {}
    for arm, spec in chosen.items():
        described[arm] = {**spec, "targets": A.target_tasks(spec["terms"], rel, tasks)}
        for seed in seeds:
            params = copy.deepcopy(by_seed.get(seed, template))
            params["seed"] = seed
            params["model"] = "RankMixerRelSup"
            params["relsup_log_every"] = log_every
            params.update({f"relsup_{k}": v for k, v in spec["overrides"].items()})
            expids.append(f"{arm}_S{seed}")
            resolved[expids[-1]] = params
    output.mkdir(parents=True)
    (output / "model_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    (output / "dataset_config.yaml").write_text(
        (source / "dataset_config.yaml").read_text())
    (output / "expids.txt").write_text("\n".join(expids) + "\n")
    (output / "protocol.json").write_text(json.dumps({
        "phase": phase, "protocol": A.protocol_constants(), "seeds": list(seeds),
        "seed_role": "confirmation" if phase == "confirm" else "development",
        "arms": described, "tasks": tasks, "edges": rel.edges, "siblings": rel.siblings,
        "source": str(source), "source_family": family,
        "dataset_id": template["dataset_id"], "model_root": str(model_root),
        "changes": ["model -> RankMixerRelSup", "relsup_* keys", "checkpoint root"]
                   + (["seed"] if phase == "confirm" else []),
        "created_with": {k: v for k, v in snapshot().items()
                         if k in ("commit", "dirty_files", "python")},
    }, indent=2))
    return expids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["1", "2", "3", "confirm"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--max-runs", type=int, default=20)
    args = parser.parse_args()
    chosen_selection = json.loads(args.selection.read_text()) if args.selection else None
    try:
        print("\n".join(build(args.source, args.output, args.model_root, args.phase,
                              chosen_selection, args.max_runs)))
    except ValueError as error:
        raise SystemExit(f"refused: {error}")
