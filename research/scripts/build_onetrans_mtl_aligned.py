#!/usr/bin/env python3
"""OneTrans-native backbone, same MTL task matrix and four seeds as RankMixer."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from build_rankmixer_mtl_aligned import HEADS, BLOCKED, LOOP, SEEDS, fingerprint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_onetrans_mtl_aligned"


def build(out):
    family = "config_onetrans_all_datasets_multiseed"
    source = yaml.safe_load((ROOT / family / "model_config.yaml").read_text())
    data = yaml.safe_load((ROOT / family / "dataset_config.yaml").read_text())
    base = deepcopy(source["Base"])
    base["model_root"] = "./runs_local/checkpoints_onetrans_mtl_aligned/"
    config, arms = {"Base": base}, {}
    for ds in data:
        baseline_id = f"OneTransAll_{ds}_Baseline_S20262027"
        baseline = source[baseline_id]
        for seed in SEEDS:
            for variant, source_variant in (
                    ("Parallel", "Baseline"), ("FiLM", "FiLM"), ("QVLoRA", "QVLoRA")):
                expid = f"History_{variant}_{ds}_S{seed}"
                source_id = f"OneTransAll_{ds}_{source_variant}_S20262027"
                config[expid] = {**deepcopy(source[source_id]), "seed": seed}
                arms[expid] = {
                    "study": "historical_replication", "variant": variant,
                    "dataset": ds, "seed": seed, "status": "ready",
                    "source": f"{family}/{source_id}",
                    "control": f"History_Parallel_{ds}_S{seed}",
                }
            for variant, overrides in HEADS.items():
                expid = f"Head_{variant}_{ds}_S{seed}"
                model = variant if variant in (
                    "ParallelIndependentTower", "SharedBottom", "MMoE", "PLE",
                    "STEM", "MoME", "ResFlow", "OneRank", "LoopMT") else "LoopMT"
                item = deepcopy(baseline)
                for key in ("model_type", "enable_loop_diagnostics"):
                    item.pop(key, None)
                # OneTrans defaults to expansion 4 in the historical model.
                item.setdefault("expansion_factor", 4)
                item.update(model=model, seed=seed, bottom_type="onetrans",
                            task_token_mode="none")
                if model == "LoopMT":
                    item.update(LOOP, num_loops=baseline["num_tasks"])
                item.update(overrides)
                if variant == "PLE":
                    item["expert_hidden_units"] = [
                        int(baseline["token_dim"]) * int(baseline["num_ns_token"])]
                if variant == "OneRank":
                    item["head_num_heads"] = overrides["num_heads"]
                    item["num_heads"] = baseline["num_heads"]
                config[expid] = item
                arms[expid] = {
                    "study": "fixed_backbone_head", "variant": variant,
                    "dataset": ds, "seed": seed,
                    "status": "blocked_reference" if variant in BLOCKED else
                    "alias" if variant == "SharedBottom" else "ready",
                    "source": f"{family}/{baseline_id}",
                    "control": f"Head_ParallelIndependentTower_{ds}_S{seed}",
                    "note": BLOCKED.get(variant, "STEM pooled-representation adaptation."
                                        if variant == "STEM" else ""),
                }
    hashes = fingerprint()
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "protocol_version": 1, "backbone": "OneTrans", "seeds": list(SEEDS),
        "seed_policy": "Historical seeds 20262027-29 plus matched fourth seed 20262030.",
        "source_hashes": hashes, "arms": arms,
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (("model_config.yaml", config), ("dataset_config.yaml", data)):
        (out / name).write_text(yaml.safe_dump(value, sort_keys=False))
    manifest["config_sha256"] = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ("model_config.yaml", "dataset_config.yaml")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for study in ("historical_replication", "fixed_backbone_head"):
        ready = [n for n, a in arms.items() if a["study"] == study and a["status"] == "ready"]
        (out / f"{study}_expids.txt").write_text("\n".join(ready) + "\n")
    print(f"{len(arms)} configured; {sum(a['status']=='ready' for a in arms.values())} ready -> {out}")
    return config, data, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    build(parser.parse_args().out)
