#!/usr/bin/env python3
"""Real blocked-batch sanity checks for OneTrans NS/S state mechanisms."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_zoo
from unirank.features import FeatureMap
from unirank.preprocess import FeatureProcessor, build_dataset
from unirank.pytorch.dataloaders import RankDataLoader
from unirank.utils import load_config


DATASETS = ("Taobao_Action", "KuaiRand_Video_Action")
VARIANTS = (
    "Baseline", "RecurrentNS", "IndependentNS", "AnchoredScratchNS",
    "DecoupledScratchNS", "SOnlyMemory",
)
SEED = 20262027


def main():
    config_dir = ROOT / "config_onetrans_ns_s_state_multiseed"
    output_path = ROOT / "reports/onetrans_ns_s_state_sanity.json"
    results = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            expid = f"OneTransState_{dataset}_{variant}_S{SEED}"
            print(f"[sanity] start {expid}", flush=True)
            params = load_config(str(config_dir), expid)
            params.update(
                gpu=-1, distributed=False, rank=0, local_rank=0, world_size=1,
                enable_bf16=False, enable_torch_compile=False, num_workers=0,
                batch_size=2, max_len=8,
            )
            processor = FeatureProcessor(**params)
            params["train_data"], params["valid_data"], params["test_data"] = \
                build_dataset(processor, **params)
            data_dir = Path(params["data_root"]) / dataset
            feature_map = FeatureMap(dataset, str(data_dir))
            feature_map.load(str(data_dir / "feature_map.json"), params)
            model = model_zoo.OneTrans(feature_map, **params)
            loader, _ = RankDataLoader(
                feature_map, stage="train", **params).make_iterator()
            batch = next(iter(loader))
            output = model(batch)
            labels = model.get_labels(batch)
            loss = model.compute_loss(output, labels)
            loss.backward()
            named_grads = {
                name: float(parameter.grad.float().norm())
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            required_prefixes = []
            if variant != "Baseline":
                required_prefixes += ["ns_refiner", "refiner_gate_logits"]
            if variant == "SOnlyMemory":
                required_prefixes += ["s_memory_writer", "memory_gate_logits"]
            missing_gradients = [
                prefix for prefix in required_prefixes
                if not any(name.startswith(prefix) for name in named_grads)
            ]
            result = {
                "expid": expid,
                "dataset": dataset,
                "variant": variant,
                "parameters": sum(p.numel() for p in model.parameters()),
                "loss": float(loss.detach()),
                "finite_gradients": all(
                    bool(torch.isfinite(p.grad).all())
                    for p in model.parameters() if p.grad is not None),
                "missing_required_gradients": missing_gradients,
                "output_shapes": {
                    key: list(value.shape) for key, value in output.items()
                    if not key.startswith("__")
                },
            }
            if not result["finite_gradients"] or missing_gradients:
                raise RuntimeError(f"sanity failed: {result}")
            results.append(result)
            print(
                f"[sanity] passed {expid}: loss={result['loss']:.6f}, "
                f"params={result['parameters']:,}", flush=True)
            del model, loader, batch, output, labels, loss, processor, feature_map
            gc.collect()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
