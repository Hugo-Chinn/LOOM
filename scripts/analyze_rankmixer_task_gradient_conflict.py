#!/usr/bin/env python3
"""Measure per-task gradient conflicts on shared RankMixer Loop parameters.

For a trained checkpoint this reports, on fixed batches:

  * pairwise cosine between per-task gradients on shared refinement parameters
  * per-task gradient norms and their ratio against the summed gradient
  * how much a PCGrad-style projection would remove from each task gradient
  * per-task / per-stage gradient norms on the behavior embedding

Scope is limited to shared refinement parameters (loop_block, behavior_embedding)
so historical controls stay comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unirank.features import FeatureMap  # noqa: E402
from unirank.preprocess import FeatureProcessor, build_dataset  # noqa: E402
from unirank.pytorch.dataloaders import RankDataLoader  # noqa: E402
from unirank.utils import load_config  # noqa: E402
import model_zoo  # noqa: E402


def flatten_grads(params):
    chunks = []
    for param in params:
        if param.grad is None:
            chunks.append(torch.zeros(param.numel(), device=param.device))
        else:
            chunks.append(param.grad.detach().reshape(-1))
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def zero_grads(params):
    for param in params:
        param.grad = None


def cosine(a, b):
    denom = float(a.norm() * b.norm())
    if denom == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def collect_scopes(model):
    scopes = {}
    if getattr(model, "loop_block", None) is not None:
        scopes["loop_block"] = [p for p in model.loop_block.parameters()]
    elif getattr(model, "loop_blocks", None) is not None:
        for index, block in enumerate(model.loop_blocks):
            scopes["loop_blocks[%d]" % index] = [p for p in block.parameters()]
    if getattr(model, "behavior_embedding", None) is not None:
        scopes["behavior_embedding"] = [model.behavior_embedding.weight]
    return scopes


def summarize(values):
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "frac_negative": float((tensor < 0).to(torch.float64).mean()),
    }


def measure_conflicts(model, loader, labels, scopes, max_batches):
    num_tasks = len(labels)
    stats = {
        scope: {
            "pair_cosine": {(i, j): [] for i in range(num_tasks) for j in range(i + 1, num_tasks)},
            "norm": [[] for _ in range(num_tasks)],
            "cos_to_total": [[] for _ in range(num_tasks)],
            "norm_ratio_to_total": [[] for _ in range(num_tasks)],
            "pcgrad_removed_fraction": [[] for _ in range(num_tasks)],
        }
        for scope in scopes
    }
    batch_count = 0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch_count += 1
        output = model(batch)
        y_true = model.get_labels(batch)
        task_losses = [
            model.loss_fn[i](output["%s_pred" % labels[i]], y_true[i], reduction="mean")
            for i in range(num_tasks)
        ]
        total_loss = torch.stack(task_losses).sum()
        for scope, params_list in scopes.items():
            zero_grads(params_list)
            total_loss.backward(retain_graph=True)
            total_grad = flatten_grads(params_list)
            total_norm = float(total_grad.norm())
            task_grads = []
            for task_idx in range(num_tasks):
                zero_grads(params_list)
                task_losses[task_idx].backward(retain_graph=True)
                grad = flatten_grads(params_list).clone()
                task_grads.append(grad)
                entry = stats[scope]
                entry["norm"][task_idx].append(float(grad.norm()))
                entry["cos_to_total"][task_idx].append(cosine(grad, total_grad))
                entry["norm_ratio_to_total"][task_idx].append(
                    float(grad.norm()) / total_norm if total_norm > 0 else 0.0
                )
            for task_idx in range(num_tasks):
                grad = task_grads[task_idx].clone()
                original_norm = float(grad.norm())
                for other_idx in range(num_tasks):
                    if other_idx == task_idx:
                        continue
                    other = task_grads[other_idx]
                    dot = float(torch.dot(grad, other))
                    if dot < 0:
                        grad = grad - (dot / float(other.dot(other))) * other
                new_norm = float(grad.norm())
                removed = 0.0 if original_norm == 0 else 1.0 - new_norm / original_norm
                stats[scope]["pcgrad_removed_fraction"][task_idx].append(removed)
            for i in range(num_tasks):
                for j in range(i + 1, num_tasks):
                    stats[scope]["pair_cosine"][(i, j)].append(cosine(task_grads[i], task_grads[j]))
            zero_grads(params_list)
    return stats, batch_count


def measure_stage_sensitivity(model, loader_factory, labels, max_batches):
    weight = model.behavior_embedding.weight
    num_stages = weight.shape[0]
    norms = [[[] for _ in range(num_stages)] for _ in labels]
    loader = loader_factory()
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        output = model(batch)
        y_true = model.get_labels(batch)
        for task_idx, label in enumerate(labels):
            zero_grads([weight])
            loss = model.loss_fn[task_idx](output["%s_pred" % label], y_true[task_idx], reduction="mean")
            loss.backward(retain_graph=True)
            if weight.grad is None:
                continue
            for stage in range(num_stages):
                norms[task_idx][stage].append(float(weight.grad[stage].norm()))
        zero_grads([weight])
    return norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--expid", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--stage", type=str, default="valid")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    params = load_config(args.config, args.expid)
    params.update(
        gpu=int(args.gpu), distributed=False, rank=0, local_rank=0, world_size=1,
        num_workers=0, enable_bf16=False,
    )
    processor = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = build_dataset(processor, **params)
    data_dir = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(data_dir))
    fmap.load(str(data_dir / "feature_map.json"), params)
    model = model_zoo.RankMixer(fmap, **params).cuda()
    checkpoint = Path(args.checkpoint_dir) / params["dataset_id"] / ("%s.model" % args.expid)
    model.load_weights(str(checkpoint))
    model.eval()

    def loader_factory():
        # stage="train" builds (train_gen, valid_gen); we always measure on the
        # fixed validation stream so batches are deterministic.
        _, valid_gen = RankDataLoader(fmap, stage="train", **params).make_iterator()
        return valid_gen

    loader = loader_factory()
    labels = list(fmap.labels)
    scopes = collect_scopes(model)
    stats, batch_count = measure_conflicts(model, loader, labels, scopes, args.batches)

    report = {
        "expid": args.expid,
        "dataset_id": params["dataset_id"],
        "checkpoint": str(checkpoint),
        "stage": args.stage,
        "batches": batch_count,
        "labels": labels,
        "loop_topology": getattr(model, "loop_topology", None),
        "supervision_mode": getattr(model, "supervision_mode", None),
        "scopes": {},
    }
    for scope in scopes:
        entry = stats[scope]
        report["scopes"][scope] = {
            "task_grad_norm": [summarize(v) for v in entry["norm"]],
            "cos_to_total": [summarize(v) for v in entry["cos_to_total"]],
            "norm_ratio_to_total": [summarize(v) for v in entry["norm_ratio_to_total"]],
            "pcgrad_removed_fraction": [summarize(v) for v in entry["pcgrad_removed_fraction"]],
            "pair_cosine": {
                "%s|%s" % (labels[i], labels[j]): summarize(entry["pair_cosine"][(i, j)])
                for i in range(len(labels))
                for j in range(i + 1, len(labels))
            },
        }
    if getattr(model, "behavior_embedding", None) is not None:
        norms = measure_stage_sensitivity(model, loader_factory, labels, args.batches)
        report["behavior_embedding_stage_norm"] = {
            labels[task_idx]: [summarize(v) for v in norms[task_idx]]
            for task_idx in range(len(labels))
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
