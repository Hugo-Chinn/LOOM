#!/usr/bin/env python3
"""CPU micro-batch sanity check for OneTrans loop variants."""
from __future__ import annotations
import argparse
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


DEFAULT = [
    "OneTrans_Taobao_P2_WholeEncoderLoop",
    "OneTrans_Taobao_P2_BaseRefiner_L1",
    "OneTrans_Taobao_P2_BaseRefiner_L3",
    "OneTrans_Taobao_P3_BaseRefiner_Branch",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expids", nargs="*", default=DEFAULT)
    parser.add_argument("--config", type=Path, default=ROOT / "config_local")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "reports/unirank_taobao_onetrans_loop_cpu_sanity.json",
    )
    args = parser.parse_args()
    results = []

    for expid in args.expids:
        params = load_config(str(args.config), expid)
        params.update(
            gpu=-1,
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            enable_bf16=False,
            enable_torch_compile=False,
            num_workers=0,
            batch_size=2,
            max_len=8,
        )
        processor = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = build_dataset(
            processor, **params
        )
        data_dir = Path(params["data_root"]) / params["dataset_id"]
        feature_map = FeatureMap(params["dataset_id"], str(data_dir))
        feature_map.load(str(data_dir / "feature_map.json"), params)
        model = model_zoo.OneTrans(feature_map, **params)
        loader, _ = RankDataLoader(feature_map, stage="train", **params).make_iterator()
        batch = next(iter(loader))
        output = model(batch)
        labels = model.get_labels(batch)
        loss = model.compute_loss(output, labels)
        loss.backward()
        loop_module = model.loop_block if model.loop_block is not None else model.ns_refiner
        loop_gradients = []
        if loop_module is not None:
            loop_gradients = [
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in loop_module.parameters()
            ]
        results.append(
            {
                "expid": expid,
                "model_type": model.model_type,
                "supervision_mode": model.supervision_mode,
                "parameters": sum(p.numel() for p in model.parameters()),
                "batch_size": int(labels[0].shape[0]),
                "output_shapes": {k: list(v.shape) for k, v in output.items() if not k.startswith("__")},
                "loss": float(loss.detach()),
                "finite_gradients": all(
                    bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                    if p.grad is not None
                ),
                "loop_gradients": all(loop_gradients) if loop_gradients else None,
                "fusion_gradients": (
                    None if model.branch_fusion_logits is None else
                    bool(model.branch_fusion_logits.grad is not None
                         and torch.isfinite(model.branch_fusion_logits.grad).all())
                ),
                "peft_type": model.peft_type,
                "peft_gradient_summary": {
                    "present": sum(
                        p.grad is not None for name, p in model.named_parameters()
                        if any(tag in name for tag in ("lora", "film", "task_adapters"))
                    ),
                    "finite": all(
                        p.grad is None or bool(torch.isfinite(p.grad).all())
                        for name, p in model.named_parameters()
                        if any(tag in name for tag in ("lora", "film", "task_adapters"))
                    ),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
