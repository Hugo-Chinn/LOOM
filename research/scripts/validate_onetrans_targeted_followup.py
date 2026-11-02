#!/usr/bin/env python3
"""Real blocked-batch sanity checks for targeted OneTrans follow-ups."""
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


EXPERIMENTS = (
    ("Taobao_Action", "ContextDAGMean"),
    ("Taobao_Action", "ContextDAGLearned"),
    ("KuaiRand_Video_Action", "DecProtect05"),
    ("KuaiRand_Video_Action", "DecProtect10"),
    ("KuaiRand_Video_Action", "CrossOnly"),
    ("KuaiRand_Video_Action", "CrossOnlyProtect05"),
)
SEED = 20262027


def main():
    config_dir = ROOT / "config_onetrans_targeted_followup_multiseed"
    output_path = ROOT / "reports/onetrans_targeted_followup_sanity.json"
    results = []
    for dataset, variant in EXPERIMENTS:
        expid = f"OneTransTarget_{dataset}_{variant}_S{SEED}"
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
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        required = ["ns_refiner", "refiner_gate_logits"]
        if variant == "ContextDAGLearned":
            required.append("branch_fusion_logits")
        missing = [
            prefix for prefix in required
            if not any(name.startswith(prefix) for name in named_grads)
        ]
        result = {
            "expid": expid,
            "parameters": sum(p.numel() for p in model.parameters()),
            "loss": float(loss.detach()),
            "finite_gradients": all(
                bool(torch.isfinite(grad).all()) for grad in named_grads.values()),
            "missing_required_gradients": missing,
            "protected_prediction_count": len(
                output.get("__protected_state_predictions", [])),
        }
        if variant in {"DecProtect05", "DecProtect10", "CrossOnlyProtect05"}:
            expected = 6  # follow retained at 4 later stages, comment at 2.
            if result["protected_prediction_count"] != expected:
                raise RuntimeError(f"unexpected protected count: {result}")
        if not result["finite_gradients"] or missing:
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
