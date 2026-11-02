#!/usr/bin/env python3
"""One real micro-batch forward/backward check per generalization config."""
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

EXPIDS = [
    "RankMixer_MerRec_Baseline", "RankMixer_MerRec_LoopDeep",
    "RankMixer_QKVideo_Baseline", "RankMixer_QKVideo_LoopDeep",
    "RankMixer_KuaiRand_Baseline", "RankMixer_KuaiRand_LoopDeep",
]


def main():
    rows = []
    for expid in EXPIDS:
        params = load_config(str(ROOT / "config_rankmixer_generalization"), expid)
        params.update(gpu=0, distributed=False, rank=0, local_rank=0, world_size=1,
                      batch_size=32, num_workers=0, enable_torch_compile=False,
                      enable_bf16=True, max_len=32)
        processor = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = build_dataset(processor, **params)
        data_dir = Path(params["data_root"]) / params["dataset_id"]
        fmap = FeatureMap(params["dataset_id"], str(data_dir))
        fmap.load(str(data_dir / "feature_map.json"), params)
        model = model_zoo.RankMixer(fmap, **params).cuda()
        loader, _ = RankDataLoader(fmap, stage="train", **params).make_iterator()
        batch = next(iter(loader))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
        labels = model.get_labels(batch)
        loss = model.compute_loss({key: value.float() for key, value in output.items()}, labels)
        loss.backward()
        rows.append({
            "expid": expid,
            "dataset": params["dataset_id"],
            "tasks": list(fmap.labels),
            "parameters": sum(p.numel() for p in model.parameters()),
            "loss": float(loss.detach()),
            "finite_gradients": all(
                bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                if p.grad is not None
            ),
            "shared_loop_gradients": (
                None if model.loop_block is None else all(
                    p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.loop_block.parameters()
                )
            ),
        })
        del model, batch, output, loss
        torch.cuda.empty_cache()
    out = ROOT.parent / "reports/unirank_rankmixer_generalization_sanity.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
