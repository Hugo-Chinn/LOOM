#!/usr/bin/env python3
"""Post-hoc behaviour-embedding permutation probe.

Train once, then at inference permute which behaviour code each node injects.
If the learned codes are position-specific, permutations must hurt even though
the graph, weights and parameter count are untouched.
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
from sklearn.metrics import roc_auc_score  # noqa: E402
import model_zoo  # noqa: E402


def evaluate(model, test_gen, labels, device):
    model.eval()
    preds = {l: [] for l in labels}
    trues = {l: [] for l in labels}
    with torch.no_grad():
        for batch in test_gen:
            output = model(batch)
            y = model.get_labels(batch)
            for i, l in enumerate(labels):
                preds[l].append(output[f"{l}_pred"].float().cpu().numpy().reshape(-1))
                trues[l].append(y[i].float().cpu().numpy().reshape(-1))
    return {l: float(roc_auc_score(np.concatenate(trues[l]), np.concatenate(preds[l]))) for l in labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--expid", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--permutations", default="identity,swap_click_buy,rotate,reverse,random42")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = load_config(args.config, args.expid)
    params.update(gpu=int(args.gpu), distributed=False, rank=0, local_rank=0,
                  world_size=1, num_workers=0, enable_bf16=False)
    processor = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = build_dataset(processor, **params)
    data_dir = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(data_dir))
    fmap.load(str(data_dir / "feature_map.json"), params)
    model = model_zoo.RankMixer(fmap, **params).cuda()
    ckpt = Path(args.checkpoint_dir) / params["dataset_id"] / f"{args.expid}.model"
    model.load_weights(str(ckpt))
    labels = list(fmap.labels)
    rows = model.behavior_embedding.weight.shape[0]

    _, test_gen = RankDataLoader(fmap, stage="train", **params).make_iterator()  # valid stream

    perms = {"identity": list(range(rows))}
    if rows >= 4:
        p = list(range(rows)); p[0], p[-1] = p[-1], p[0]
        perms["swap_root_sink"] = p
        perms["rotate"] = [ (i + 1) % rows for i in range(rows) ]
        perms["reverse"] = list(reversed(range(rows)))
        rng = np.random.default_rng(42); r = list(range(rows)); rng.shuffle(r)
        perms["random42"] = r

    original = model.behavior_embedding.weight.data.clone()
    results = {}
    for name in args.permutations.split(","):
        name = name.strip()
        if name not in perms:
            continue
        perm = torch.tensor(perms[name], device=original.device)
        with torch.no_grad():
            model.behavior_embedding.weight.copy_(original[perm])
        aucs = evaluate(model, test_gen, labels, None)
        results[name] = {"permutation": perms[name], "per_task_auc": aucs,
                         "macro_auc": sum(aucs.values()) / len(aucs)}
        print(name, perms[name], "macro=%.6f" % results[name]["macro_auc"], flush=True)
    with torch.no_grad():
        model.behavior_embedding.weight.copy_(original)

    Path(args.out).write_text(json.dumps({"expid": args.expid, "dataset_id": params["dataset_id"],
                                          "labels": labels, "results": results}, indent=2))


if __name__ == "__main__":
    main()
