#!/usr/bin/env python3
"""Reproducible eager BF16 compute microbenchmarks; no accuracy claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    job = json.loads(args.job)
    runtime = Path(job["runtime"])
    sys.path.insert(0, str(runtime))
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from unirank.features import FeatureMap
    from unirank.utils import load_config
    import model_zoo

    torch.set_num_threads(2)
    torch.manual_seed(20262027)
    torch.cuda.set_device(0)
    params = load_config(str(ROOT / job["config"]), job["expid"])
    params.update(gpu=0, distributed=False, rank=0, local_rank=0, world_size=1,
                  enable_torch_compile=False, gradient_checkpointing=False, verbose=0,
                  model_root=str(ROOT / "runs_local/efficiency_microbench"))
    ds = params["dataset_id"]
    data_dir = Path(params["data_root"]) / ds
    fmap = FeatureMap(ds, str(data_dir))
    fmap.load(str(data_dir / "feature_map.json"), params)
    model = getattr(model_zoo, params["model"])(fmap, **params)
    model.model_to_device()
    counts = {"total": 0, "embedding": 0, "dense": 0, "trainable": 0}
    groups = {}
    for name, p in model.named_parameters():
        counts["total"] += p.numel()
        counts["trainable"] += p.numel() if p.requires_grad else 0
        kind = "embedding" if name.startswith("embedding_layer.") else "dense"
        counts[kind] += p.numel()
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()

    def inputs(batch):
        generator = torch.Generator().manual_seed(20262027)
        context, items = {}, {}
        length = int(params.get("max_len", 100))
        for name, spec in fmap.features.items():
            if spec["type"] == "meta":
                continue
            target = items if spec.get("source") == "item" or name == "action" else context
            shape = (batch, length + 1) if target is items else (batch,)
            if spec["type"] == "categorical":
                target[name] = torch.randint(1, int(spec["vocab_size"]), shape,
                                             generator=generator, device="cpu").to("cuda")
            elif spec["type"] == "numeric":
                target[name] = torch.rand(shape, generator=generator).to("cuda")
            elif spec["type"] == "sequence":
                shape = (*shape, int(spec["max_len"]))
                target[name] = torch.randint(1, int(spec["vocab_size"]), shape,
                                             generator=generator).to("cuda")
            else:
                raise ValueError(f"Unsupported feature: {name}/{spec['type']}")
        if "action" in items:
            items["action"][:, -1] = 0
        return context, items, torch.ones(batch, length, device="cuda", dtype=torch.bool)

    def timed(fn, warmup, repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        return {
            "median_ms": statistics.median(times),
            "mean_ms": statistics.mean(times),
            "p95_ms": sorted(times)[max(0, int(0.95 * len(times)) - 1)],
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "repeats": repeats,
        }

    result = {
        **job, "dataset": ds, "counts": counts, "parameter_groups": groups,
        "precision": "BF16 autocast, FP32 parameter storage",
        "execution": "eager, compile disabled", "input": "synthetic full-length history, real feature vocabularies",
        "history_length": params.get("max_len", 100), "trained_weights": False,
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "source_sha256": {
            str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (runtime / "model_zoo").glob("*.py")
        },
    }
    model.eval()
    result["inference"] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in (1, 256):
            x = inputs(batch)
            stat = timed(lambda: model(x), 5, args.repeats)
            stat["samples_per_second"] = batch * 1000 / stat["median_ms"]
            result["inference"][str(batch)] = stat
        x = inputs(1)
        with FlopCounterMode(display=False) as counter:
            model(x)
        result["forward_gflops_per_sample"] = counter.get_total_flops() / 1e9
        result["flop_operators"] = {str(k): v for k, v in counter.flop_counts["Global"].items()}
    result["flop_scope"] = (
        "PyTorch FlopCounterMode registered operators; multiply-add counts as 2 FLOPs. "
        "Embedding lookup, normalization, activation, data movement and unregistered "
        "operators may be omitted. This is counted operator FLOPs, not complete hardware FLOPs."
    )
    model.train()
    x = inputs(256)
    targets = [torch.randint(0, 2, (256, 1), device="cuda").float() for _ in fmap.labels]

    def train_step():
        model._optimizer_zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(x)
        predictions = {k: v.float() if torch.is_tensor(v) else v
                       for k, v in predictions.items()}
        loss = model.add_loss(predictions, targets)
        loss.backward()
        model._optimizer_step()

    result["training_256"] = timed(train_step, 2, 10)
    result["training_256"]["samples_per_second"] = 256000 / result["training_256"]["median_ms"]
    # Count forward+backward separately from optimizer math.
    model._optimizer_zero_grad()
    with FlopCounterMode(display=False) as counter:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(x)
        predictions = {k: v.float() if torch.is_tensor(v) else v
                       for k, v in predictions.items()}
        loss = model.add_loss(predictions, targets)
        loss.backward()
    result["forward_backward_gflops_per_sample"] = counter.get_total_flops() / (256 * 1e9)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE {job['name']} {ds}", flush=True)


if __name__ == "__main__":
    main()
