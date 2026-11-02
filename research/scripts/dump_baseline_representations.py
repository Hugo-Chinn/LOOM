#!/usr/bin/env python3
"""Dump representations for any model_zoo class (loop stages and/or head inputs).

For loop models we keep the per-stage pooled token representation
(`__loop_representations`). For every model we additionally capture the input
of each task tower, which is the representation the head actually reads. This
gives a comparable "head-input" space across loop and non-loop baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unirank.features import FeatureMap  # noqa: E402
from unirank.preprocess import FeatureProcessor, build_dataset  # noqa: E402
from unirank.pytorch.dataloaders import RankDataLoader  # noqa: E402
from unirank.utils import load_config  # noqa: E402
import model_zoo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--expid", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=32768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = load_config(args.config, args.expid)
    params.update(gpu=int(args.gpu), distributed=False, rank=0, local_rank=0,
                  world_size=1, num_workers=0, enable_bf16=False)
    processor = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = build_dataset(processor, **params)
    data_dir = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(data_dir))
    fmap.load(str(data_dir / "feature_map.json"), params)
    model_cls = getattr(model_zoo, params.get("model", "RankMixer"))
    model = model_cls(fmap, **params).cuda()
    ckpt = Path(args.checkpoint_dir) / params["dataset_id"] / f"{args.expid}.model"
    model.load_weights(str(ckpt))
    model.eval()
    labels = list(fmap.labels)
    num_tasks = len(labels)

    # Different model families expose the per-task head differently; use the
    # first candidate that yields one module per task.
    candidates = []
    taps = getattr(model, "task_head_input_taps", None)
    if taps is not None:
        candidates.append(list(taps))
    tower = getattr(model, "tower", None)
    if tower is not None and len(tower) > 0:
        candidates.append(list(tower))
    tails = getattr(model, "task_head_tails", None)
    if tails is not None and len(tails) > 0:
        candidates.append(list(tails))
    layers = getattr(model, "tower_layers", None)
    if layers is not None and len(layers) > 0:
        candidates.append([layer[0] for layer in layers])
    heads = getattr(model, "head_layers", None)
    if heads is not None and len(heads) > 0:
        candidates.append([head[0] if hasattr(head, "__getitem__") else head for head in heads])
    modules = next((c for c in candidates if len(c) == num_tasks), candidates[0] if candidates else [])
    captured = {}
    handles = []

    def make_hook(index):
        def hook(_module, inputs):
            if index not in captured:
                captured[index] = inputs[0].detach()
        return hook

    for index, module in enumerate(modules):
        handles.append(module.register_forward_pre_hook(make_hook(index)))

    _, valid_gen = RankDataLoader(fmap, stage="train", **params).make_iterator()
    head_inputs = []
    stage_reps = []
    ys = []
    total = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(valid_gen):
            if batch_index >= args.batches or total >= args.max_samples:
                break
            captured.clear()
            output = model(batch)
            y = model.get_labels(batch)
            take = min(output[f"{labels[0]}_pred"].shape[0], args.max_samples - total)
            if "__loop_representations" in output:
                stage_reps.append(output["__loop_representations"][:take].float().cpu().numpy())
            if len(captured) == num_tasks:
                stacked = torch.stack([captured[i][:take] for i in range(num_tasks)], dim=1)
                head_inputs.append(stacked.float().cpu().numpy())
            ys.append(torch.stack([t[:take].float().cpu() for t in y], dim=1).numpy())
            total += take
    for h in handles:
        h.remove()

    out = {"labels": np.array(labels), "y": np.concatenate(ys, axis=0)}
    if head_inputs:
        out["head_inputs"] = np.concatenate(head_inputs, axis=0)
    if stage_reps:
        out["stage_reps"] = np.concatenate(stage_reps, axis=0)
        out["stage_names"] = np.array(["base"] + labels if out["stage_reps"].shape[1] == num_tasks + 1
                                      else [f"stage_{i}" for i in range(out["stage_reps"].shape[1])])
    out["meta"] = np.array(json.dumps({
        "expid": args.expid, "dataset_id": params["dataset_id"],
        "model": params.get("model"), "backbone": getattr(model, "model_id", params.get("model")),
        "loop_topology": params.get("loop_topology"), "num_loops": params.get("num_loops"),
        "num_preloops": params.get("num_preloops"), "head_sharing": params.get("head_sharing"),
        "samples": int(total), "has_stages": "stage_reps" in out, "has_head_inputs": "head_inputs" in out,
    }))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(args.out), **out)
    print(json.dumps({"out": args.out, "samples": int(total),
                      "has_stages": "stage_reps" in out, "has_head_inputs": "head_inputs" in out}))


if __name__ == "__main__":
    main()
