#!/usr/bin/env python3
"""Real micro-batch checks for hierarchical RankMixer DAGs."""
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
    "RankMixer_MerRec_DAG", "RankMixer_MerRec_DAG_Gated",
    "RankMixer_QKVideo_DAG", "RankMixer_QKVideo_DAG_Gated",
    "RankMixer_KuaiRand_DAG", "RankMixer_KuaiRand_DAG_Gated",
]


def main():
    rows = []
    for expid in EXPIDS:
        params = load_config(str(ROOT / "config_rankmixer_hierarchical"), expid)
        params.update(gpu=0, distributed=False, rank=0, local_rank=0, world_size=1,
                      batch_size=16, max_len=24, num_workers=0,
                      enable_torch_compile=False, enable_bf16=True)
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
            "labels": list(fmap.labels),
            "topological_order": [fmap.labels[i] for i in model.graph_topological_order],
            "parents": {
                fmap.labels[i]: [fmap.labels[p] for p in model.graph_parents[i]]
                for i in range(model.num_tasks)
            },
            "descendants": {
                fmap.labels[i]: [fmap.labels[d] for d in model.graph_descendants[i]]
                for i in range(model.num_tasks)
            },
            "parameters": sum(p.numel() for p in model.parameters()),
            "loss": float(loss.detach()),
            "finite_gradients": all(
                bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                if p.grad is not None
            ),
            "shared_loop_gradients": all(
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in model.loop_block.parameters()
            ),
            "gate_gradients": (
                None if model.loop_gate_logits is None else
                bool(model.loop_gate_logits.grad is not None and torch.isfinite(model.loop_gate_logits.grad).all())
            ),
            "step_means_in_label_order": output["__loop_steps"].detach().float().mean(dim=0).cpu().tolist(),
        })
        del model, batch, output, loss
        torch.cuda.empty_cache()
    out = ROOT.parent / "reports/unirank_rankmixer_hierarchical_sanity.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
