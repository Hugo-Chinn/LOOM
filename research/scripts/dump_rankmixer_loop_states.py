#!/usr/bin/env python3
"""Dump pooled hidden states at every loop stage for tSNE visualisation.

For looped models `__loop_representations` already gives the pooled token
representation of each behaviour node. The pre-loop base state is captured
with a hook on the tokenizer. Parallel models expose a single state, captured
with a hook on the interaction block. Compilation is disabled so hooks fire.
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


def capture_base_state(model):
    """Register hooks returning (tokenizer_output, interaction_output) pooled."""
    captured = {}

    def unwrap(module):
        return getattr(module, "_orig_mod", module)

    # Checkpoints are saved from the compiled model, so keep the compiled
    # wrapper and hook the module object itself (it is still an nn.Module).
    tokenizer = model.unified_tokenizer_layer

    def tok_hook(module, inputs, output):
        captured["tokenizer"] = output

    handle = tokenizer.register_forward_hook(tok_hook)

    interaction = None
    if getattr(model, "unified_interaction_layers", None) is not None:
        interaction = model.unified_interaction_layers

        def inter_hook(module, inputs, output):
            captured["interaction"] = output

        handle2 = interaction.register_forward_hook(inter_hook)
    else:
        handle2 = None
    return captured, handle, handle2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expid", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=8192)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    params = load_config(args.config, args.expid)
    params.update(
        gpu=int(args.gpu), distributed=False, rank=0, local_rank=0, world_size=1,
        num_workers=0, enable_bf16=False,
    )
    processor = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = build_dataset(
        processor, **params
    )
    data_dir = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(data_dir))
    fmap.load(str(data_dir / "feature_map.json"), params)
    model = model_zoo.RankMixer(fmap, **params).cuda()
    checkpoint = Path(args.checkpoint_dir) / params["dataset_id"] / f"{args.expid}.model"
    model.load_weights(str(checkpoint))
    model.eval()

    _, valid_gen = RankDataLoader(fmap, stage="train", **params).make_iterator()
    captured, handle, handle2 = capture_base_state(model)

    reps = []
    labels = []
    base_reps = []
    total = 0
    with torch.no_grad():
        for index, batch in enumerate(valid_gen):
            if index >= args.batches or total >= args.max_samples:
                break
            captured.clear()
            output = model(batch)
            y_true = model.get_labels(batch)
            if "__loop_representations" in output:
                stage_reps = output["__loop_representations"].float().cpu().numpy()
            else:
                pooled = captured["interaction"].mean(dim=1)
                stage_reps = pooled.float().cpu().numpy()[:, None, :]
            if "tokenizer" in captured:
                base = captured["tokenizer"].mean(dim=1).float().cpu().numpy()
            else:
                base = np.zeros_like(stage_reps[:, 0, :])
            take = min(stage_reps.shape[0], args.max_samples - total)
            reps.append(stage_reps[:take])
            base_reps.append(base[:take])
            labels.append(
                torch.stack([y[:take].float().cpu() for y in y_true], dim=1).numpy()
            )
            total += take
    handle.remove()
    if handle2 is not None:
        handle2.remove()

    reps = np.concatenate(reps, axis=0)
    base_reps = np.concatenate(base_reps, axis=0)
    labels = np.concatenate(labels, axis=0)
    all_reps = np.concatenate([base_reps[:, None, :], reps], axis=1)
    if all_reps.shape[1] == len(fmap.labels) + 1:
        stage_names = ["base"] + list(fmap.labels)
    elif reps.shape[1] == 1:
        stage_names = ["base", "parallel_out"]
    else:
        stage_names = ["base"] + [f"stage_{i}" for i in range(reps.shape[1])]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        representations=all_reps,
        labels=labels,
        stage_names=np.array(stage_names),
        task_names=np.array(list(fmap.labels)),
        meta=json.dumps({
            "expid": args.expid,
            "dataset_id": params["dataset_id"],
            "model_type": params.get("model_type"),
            "loop_topology": params.get("loop_topology"),
            "num_loops": params.get("num_loops"),
            "num_preloops": params.get("num_preloops"),
            "samples": int(all_reps.shape[0]),
            "stages": int(all_reps.shape[1]),
            "dim": int(all_reps.shape[2]),
        }),
    )
    print(json.dumps({
        "out": str(out_path),
        "samples": int(all_reps.shape[0]),
        "stages": int(all_reps.shape[1]),
        "dim": int(all_reps.shape[2]),
        "stage_names": stage_names,
    }))


if __name__ == "__main__":
    main()
