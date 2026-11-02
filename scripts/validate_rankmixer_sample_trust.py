#!/usr/bin/env python3
"""Audit exact initialization parity and gradients on a real Taobao batch."""
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


CONFIG = ROOT / "config_rankmixer_sample_trust"
SEED = 20262028
VARIANTS = ("Stage", "Shared", "StageKeep")


def load_params(expid):
    params = load_config(str(CONFIG), expid)
    params.update(
        gpu=0, distributed=False, rank=0, local_rank=0, world_size=1,
        batch_size=8, max_len=24, num_workers=0,
        enable_torch_compile=False, enable_bf16=True,
    )
    return params


def main():
    first = f"RankMixer_Taobao_SampleTrustStage_S{SEED}"
    params = load_params(first)
    processor = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = (
        build_dataset(processor, **params)
    )
    data_dir = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(data_dir))
    fmap.load(str(data_dir / "feature_map.json"), params)
    loader, _ = RankDataLoader(fmap, stage="train", **params).make_iterator()
    batch = next(iter(loader))

    control_params = dict(params)
    control_params.update(
        model_id="SampleTrustControlAudit", sample_trust_gate="none",
        sample_trust_reg_weight=0.0,
    )
    torch.manual_seed(SEED)
    control = model_zoo.RankMixer(fmap, **control_params).cuda()
    control.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        control_output = control(batch)

    rows = []
    for variant in VARIANTS:
        expid = f"RankMixer_Taobao_SampleTrust{variant}_S{SEED}"
        item = load_params(expid)
        item.update(
            train_data=params["train_data"], valid_data=params["valid_data"],
            test_data=params["test_data"],
        )
        torch.manual_seed(SEED)
        model = model_zoo.RankMixer(fmap, **item).cuda()
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
        prediction_delta = max(
            float((output[f"{label}_pred"] - control_output[f"{label}_pred"])
                  .abs().max().float())
            for label in fmap.labels
        )
        trust = output["__sample_trust"].detach().float()
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            train_output = model(batch)
        labels = model.get_labels(batch)
        loss = model.compute_loss({
            key: value.float() if torch.is_tensor(value) else value
            for key, value in train_output.items()
        }, labels)
        loss.backward()
        trust_parameters = {
            name: parameter for name, parameter in model.named_parameters()
            if "sample_trust" in name
        }
        rows.append({
            "variant": variant,
            "prediction_max_delta": prediction_delta,
            "trust_min": float(trust.min()),
            "trust_max": float(trust.max()),
            "loss": float(loss.detach()),
            "finite_gradients": all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ),
            "trust_gradient_norms": {
                name: None if parameter.grad is None else
                float(parameter.grad.detach().float().norm())
                for name, parameter in trust_parameters.items()
            },
        })
        del model, output, train_output, loss
        torch.cuda.empty_cache()
    path = ROOT.parent / "reports/rankmixer_sample_trust_sanity.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

