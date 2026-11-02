#!/usr/bin/env python3
"""Per-task gradient participation on OneTrans shared refinement parameters.

Same diagnostic as the RankMixer one: backprop each task's primary loss
separately, then report gradient norms, cosine to the summed gradient, and
pairwise cosine on the parameters that every task shares (the recurrent
refinement block, plus the behaviour/state embeddings when present).
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
    chunks = [p.grad.detach().reshape(-1) if p.grad is not None
              else torch.zeros(p.numel(), device=p.device) for p in params]
    return torch.cat(chunks, dim=0) if chunks else None


def cosine(a, b):
    denom = float(a.norm() * b.norm())
    return 0.0 if denom == 0 else float(torch.dot(a, b) / denom)


def collect_scopes(model):
    scopes = {}
    for name in ("ns_refiner", "loop_block", "s_refiner", "behavior_embedding"):
        module = getattr(model, name, None)
        if module is None:
            continue
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            scopes[name] = params
    return scopes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expid", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    params = load_config(args.config, args.expid)
    params.update(gpu=int(args.gpu), distributed=False, rank=0, local_rank=0,
                  world_size=1, num_workers=0, enable_bf16=False)
    processor = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = build_dataset(processor, **params)
    data_dir = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(data_dir))
    fmap.load(str(data_dir / "feature_map.json"), params)
    model_cls = getattr(model_zoo, params.get("model", "OneTrans"))
    model = model_cls(fmap, **params).cuda()
    checkpoint = Path(args.checkpoint_dir) / params["dataset_id"] / f"{args.expid}.model"
    model.load_weights(str(checkpoint))
    model.eval()

    _, valid_gen = RankDataLoader(fmap, stage="train", **params).make_iterator()
    labels = list(fmap.labels)
    num_tasks = len(labels)
    scopes = collect_scopes(model)
    stats = {s: {"norm": [[] for _ in range(num_tasks)],
                 "cos_to_total": [[] for _ in range(num_tasks)]} for s in scopes}

    index = -1
    for index, batch in enumerate(valid_gen):
        if index >= args.batches:
            break
        output = model(batch)
        y_true = model.get_labels(batch)
        losses = [model.loss_fn[i](output["%s_pred" % labels[i]], y_true[i], reduction="mean")
                  for i in range(num_tasks)]
        total = torch.stack(losses).sum()
        for scope, plist in scopes.items():
            for p in plist:
                p.grad = None
            total.backward(retain_graph=True)
            total_grad = flatten_grads(plist)
            tnorm = float(total_grad.norm())
            grads = []
            for i in range(num_tasks):
                for p in plist:
                    p.grad = None
                losses[i].backward(retain_graph=True)
                g = flatten_grads(plist).clone()
                grads.append(g)
                stats[scope]["norm"][i].append(float(g.norm()))
                stats[scope]["cos_to_total"][i].append(cosine(g, total_grad))
            if "pair" not in stats[scope]:
                stats[scope]["pair"] = {(i, j): [] for i in range(num_tasks) for j in range(i + 1, num_tasks)}
            for i in range(num_tasks):
                for j in range(i + 1, num_tasks):
                    stats[scope]["pair"][(i, j)].append(cosine(grads[i], grads[j]))
            for p in plist:
                p.grad = None
    def sm(v):
        t = torch.tensor(v, dtype=torch.float64)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)),
                "min": float(t.min()), "max": float(t.max())}

    report = {"expid": args.expid, "dataset_id": params["dataset_id"],
              "model": params.get("model"), "model_type": params.get("model_type"),
              "refiner_mode": params.get("refiner_mode"),
              "batches": max(0, index + 1), "labels": labels, "scopes": {}}
    for scope in scopes:
        norms = [sm(stats[scope]["norm"][i])["mean"] for i in range(num_tasks)]
        report["scopes"][scope] = {
            "task_grad_norm": [sm(stats[scope]["norm"][i]) for i in range(num_tasks)],
            "cos_to_total": [sm(stats[scope]["cos_to_total"][i]) for i in range(num_tasks)],
            "imbalance_max_min": (max(norms) / min(norms)) if min(norms) > 0 else None,
            "pair_cosine": {"%s|%s" % (labels[i], labels[j]): sm(stats[scope]["pair"][(i, j)])
                            for i in range(num_tasks) for j in range(i + 1, num_tasks)},
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
