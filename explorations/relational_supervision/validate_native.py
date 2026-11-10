"""Bounded real-training-batch correctness and memory check; no evaluation."""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch

from .arms import phase1
from .losses import RelSupConfig
from .runtime import isolate_metadata
from .research_model import RankMixerRelSup
from model_zoo.RankMixer import RankMixer
from unirank.features import FeatureMap
from unirank.preprocess import FeatureProcessor, build_dataset
from unirank.pytorch.dataloaders import RankDataLoader
from unirank.pytorch.torch_utils import seed_everything
from unirank.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_float32_matmul_precision("high")
    params = load_config(str(args.config.resolve()), "full_S20262027")
    params = isolate_metadata(params, args.output)
    params.update(gpu=0, distributed=False, rank=0, local_rank=0, world_size=1,
                  enable_bf16=True, model_root=str(args.output / "checkpoints"))
    seed_everything(params["seed"])
    processor = FeatureProcessor(**params)
    build_dataset(processor, **params)
    data_dir = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(data_dir))
    fmap.load(str(data_dir / "feature_map.json"), params)
    model = RankMixerRelSup(fmap, **params)
    loader = RankDataLoader(fmap, stage="train", **{**params, "valid_data": None}).make_iterator()[0]
    batch = next(iter(loader))
    labels = model.get_labels(batch)
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch)
    out = {k: v.float() if v.is_floating_point() else v for k, v in out.items()}
    torch.testing.assert_close(model.compute_loss(out, labels), RankMixer.add_loss(model, out, labels))
    del out
    model._max_gradient_norm = 10.0
    model._optimizer_zero_grad()
    torch.cuda.reset_peak_memory_stats()
    checks = []
    for index, (arm, spec) in enumerate(phase1().items()):
        model.relsup = RelSupConfig(**spec["overrides"])
        model._batch_index = index
        started = time.monotonic()
        loss = model.train_step(batch)
        torch.cuda.synchronize()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss in {arm}")
        checks.append({"arm": arm, "loss_finite": True,
                       "seconds": time.monotonic() - started})
        logging.info("Validation step complete: %s", checks[-1])
        del loss
    # Fresh backward verifies finite dense and sparse gradients after optimizer steps.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch)
    loss = model.compute_loss({k: v.float() for k, v in out.items()}, labels)
    loss.backward()
    gradients = [p.grad.coalesce().values() if p.grad.is_sparse else p.grad
                 for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    result = {"dataset": params["dataset_id"], "tasks": fmap.labels,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "batch_size": len(labels[0]), "parameters": sum(p.numel() for p in model.parameters()),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "zero_weight_loss_parity": True, "finite_gradients": True,
              "bf16": True, "compile": params["enable_torch_compile"], "checks": checks,
              "note": "Training-batch execution check only; no validation/test metrics or reusable weights."}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    del model, loader, batch, out, loss
    gc.collect()


if __name__ == "__main__":
    main()
