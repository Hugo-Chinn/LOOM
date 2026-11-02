#!/usr/bin/env python3
"""Real-batch gradient and initialization audits for recurrence variants."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unirank.features import FeatureMap
from unirank.preprocess import FeatureProcessor, build_dataset
from unirank.pytorch.dataloaders import RankDataLoader
from unirank.utils import load_config
import model_zoo


CONFIG = ROOT / "config_rankmixer_recurrence_conditioning"
DATASETS = ("Taobao",)
VARIANTS = ("DepthSinusoidal", "BehaviorDepthMod", "BehaviorAdapter16")
SEED = 20262028


def expid(dataset, variant):
    return f"RankMixer_{dataset}_{variant}_S{SEED}"


def build_model(name):
    params = load_config(str(CONFIG), name)
    params.update(
        gpu=0, distributed=False, rank=0, local_rank=0, world_size=1,
        batch_size=8, max_len=24, num_workers=0,
        enable_torch_compile=False, enable_bf16=True,
    )
    processor = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = (
        build_dataset(processor, **params)
    )
    data_dir = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(data_dir))
    fmap.load(str(data_dir / "feature_map.json"), params)
    torch.manual_seed(SEED)
    return model_zoo.RankMixer(fmap, **params).cuda(), fmap, params


def common_parameter_delta(control, variant):
    excluded = (
        "depth_channel_scale", "depth_modulation", "behavior_adapters",
    )
    left = control.state_dict()
    right = variant.state_dict()
    shared = sorted(
        key for key in left.keys() & right.keys()
        if not key.startswith(excluded)
    )
    return max(
        float((left[key].detach().float().cpu()
               - right[key].detach().float().cpu()).abs().max())
        for key in shared
    )


def main():
    rows = []
    for dataset in DATASETS:
        control, _, _ = build_model(expid(dataset, "DepthChannelScale"))
        # Disable the identity-initialized variant-only parameter to obtain the
        # exact current model while retaining the same construction path.
        control.depth_channel_scale = None
        for variant_name in VARIANTS:
            model, fmap, params = build_model(expid(dataset, variant_name))
            delta = None
            if variant_name not in {
                    "BehaviorZero", "BehaviorNormal002", "BehaviorScaled01"}:
                delta = common_parameter_delta(control, model)
            loader, _ = RankDataLoader(
                fmap, stage="train", **params
            ).make_iterator()
            batch = next(iter(loader))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
            labels = model.get_labels(batch)
            float_output = {
                key: value.float() if torch.is_tensor(value) else value
                for key, value in output.items()
            }
            loss = model.compute_loss(float_output, labels)
            loss.backward()
            variant_parameters = {
                name: parameter for name, parameter in model.named_parameters()
                if name.startswith((
                    "depth_channel_scale", "depth_modulation",
                    "behavior_adapters", "behavior_embedding",
                ))
            }
            rows.append({
                "dataset": dataset,
                "expid": expid(dataset, variant_name),
                "variant": variant_name,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "loss": float(loss.detach()),
                "common_parameter_max_delta": delta,
                "finite_gradients": all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ),
                "variant_gradient_norms": {
                    name: None if parameter.grad is None
                    else float(parameter.grad.detach().float().norm())
                    for name, parameter in variant_parameters.items()
                },
                "behavior_embedding_norms": (
                    None if model.behavior_embedding is None else
                    model.behavior_embedding.weight.detach().float().norm(dim=1).cpu().tolist()
                ),
                "loop_step_means": output["__loop_steps"].detach().float().mean(dim=0).cpu().tolist(),
            })
            del model, batch, output, loss
            torch.cuda.empty_cache()
        del control
        torch.cuda.empty_cache()
    out = ROOT.parent / "reports/rankmixer_recurrence_conditioning_sanity.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
