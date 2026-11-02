#!/usr/bin/env python3
"""Real-batch sanity checks for gated and branched RankMixer loops."""
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
    "RankMixer_Gated_Linear",
    "RankMixer_Gated_Linear_Deep",
    "RankMixer_Branch",
    "RankMixer_Branch_Gated",
    "RankMixer_Branch_Gated_StepReg",
    "RankMixer_Branch_Gated_StepVarReg",
]


def main():
    rows = []
    for expid in EXPIDS:
        params = load_config(str(ROOT / "config_rankmixer_optimized"), expid)
        params.update(gpu=0, distributed=False, rank=0, local_rank=0, world_size=1,
                      batch_size=128, num_workers=0, enable_torch_compile=False,
                      enable_bf16=True, enable_loop_diagnostics=True)
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
        loss = model.compute_loss({k: v.float() for k, v in output.items()}, labels)
        loss.backward()
        rows.append({
            "expid": expid,
            "topology": model.loop_topology,
            "gated": model.use_loop_gate,
            "parameters": sum(p.numel() for p in model.parameters()),
            "loss": float(loss.detach()),
            "finite_gradients": all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
            "loop_gradients": all(p.grad is not None for p in model.loop_block.parameters()),
            "gate_gradients": None if model.loop_gate_logits is None else bool(model.loop_gate_logits.grad is not None and torch.isfinite(model.loop_gate_logits.grad).all()),
            "fusion_gradients": None if model.branch_fusion_logits is None else bool(model.branch_fusion_logits.grad is not None and torch.isfinite(model.branch_fusion_logits.grad).all()),
            "gates": None if model.loop_gate_logits is None else torch.sigmoid(model.loop_gate_logits.detach()).cpu().tolist(),
            "fusion_gates": None if model.branch_fusion_logits is None else torch.sigmoid(model.branch_fusion_logits.detach()).cpu().tolist(),
            "step_means": output["__loop_steps"].detach().float().mean(dim=0).cpu().tolist(),
            "diagnostics": getattr(model, "last_loop_diagnostics", None),
        })
        del model, batch, output, loss
        torch.cuda.empty_cache()
    out = ROOT.parent / "reports/unirank_taobao_rankmixer_optimized_sanity.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
