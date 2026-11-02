#!/usr/bin/env python3
"""Report RankMixer parameter breakdown: embeddings, non-loop model, loop.

Groups every trainable parameter into:

  * embedding  : sparse feature embedding tables
  * tokenizer  : unified tokenizer (feature -> tokens)
  * attention  : multi-head target attention
  * encoder    : unified interaction block (parallel backbone only)
  * heads      : per-task readout towers and output layers
  * loop_core  : the shared refinement block applied at every loop step
  * loop_aux   : loop-side parameters (behavior embedding, gates, fusion,
                 adapters, depth conditioning, trust routers)
  * other      : anything not matched above
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unirank.features import FeatureMap  # noqa: E402
from unirank.utils import load_config  # noqa: E402
import model_zoo  # noqa: E402


LOOP_AUX_NAMES = {
    "behavior_embedding",
    "iteration_depth_embedding",
    "loop_gate_logits",
    "preloop_gate_logits",
    "depth_gate_base_logit",
    "depth_gate_slope_raw",
    "branch_fusion_logits",
    "branch_fusion_router",
    "dag_parent_fusion_logits",
    "dag_parent_routers",
    "child_adapters",
    "base_bypass_logits",
    "depth_channel_scale",
    "depth_modulation",
    "sample_trust_routers",
    "shared_sample_trust_router",
    "preference_score_head",
}


def classify(name: str) -> str:
    top = name.split(".")[0]
    if top == "embedding_layer":
        return "embedding"
    if top in {"loop_block", "loop_blocks"}:
        return "loop_core"
    if top in LOOP_AUX_NAMES:
        return "loop_aux"
    if top in {"unified_tokenizer_layer", "task_tokens"}:
        return "tokenizer"
    if top == "attention_layers":
        return "attention"
    if top == "unified_interaction_layers":
        return "encoder"
    if top in {
        "tower", "task_output_layers", "shared_first_layer", "grouped_towers",
        "grouped_first_layers", "task_head_tails", "grouped_task_tails",
        "grouped_trunks", "grouped_task_outputs", "grouped_task_adapters",
    }:
        return "heads"
    return "other"


def breakdown(model):
    counts = {}
    unmatched = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = classify(name)
        counts[group] = counts.get(group, 0) + int(param.numel())
        if group == "other":
            unmatched.append(name)
    return counts, unmatched


def run_one(config_dir: str, expid: str):
    params = load_config(config_dir, expid)
    params.update(
        gpu=-1, distributed=False, rank=0, local_rank=0, world_size=1,
        num_workers=0, enable_bf16=False,
    )
    data_dir = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(data_dir))
    fmap.load(str(data_dir / "feature_map.json"), params)
    model = model_zoo.RankMixer(fmap, **params)
    counts, unmatched = breakdown(model)
    total = sum(counts.values())
    dense = total - counts.get("embedding", 0)
    loop = counts.get("loop_core", 0) + counts.get("loop_aux", 0)
    return {
        "expid": expid,
        "dataset_id": params["dataset_id"],
        "model_type": params.get("model_type"),
        "loop_topology": params.get("loop_topology"),
        "num_loops": params.get("num_loops"),
        "num_preloops": params.get("num_preloops"),
        "token_dim": params.get("token_dim"),
        "num_layers": params.get("num_layers"),
        "counts": counts,
        "total": total,
        "dense_total": dense,
        "loop_total": loop,
        "loop_share_of_dense": loop / dense if dense else None,
        "embedding_share_of_total": counts.get("embedding", 0) / total if total else None,
        "unmatched": unmatched,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=str, required=True,
                        help="JSON list of {config, expid}")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    jobs = json.loads(Path(args.jobs).read_text())
    results = []
    for job in jobs:
        results.append(run_one(job["config"], job["expid"]))
        print("done:", job["expid"], flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
