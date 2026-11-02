#!/usr/bin/env python3
"""Measure learned sample-trust distributions on fixed validation batches."""
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


CONFIG = ROOT / "config_rankmixer_sample_trust"
SEED = 20262028
VARIANTS = ("Stage", "Shared", "StageKeep")


def main():
    results = {}
    for variant in VARIANTS:
        expid = f"RankMixer_Taobao_SampleTrust{variant}_S{SEED}"
        params = load_config(str(CONFIG), expid)
        params.update(
            gpu=0, distributed=False, rank=0, local_rank=0, world_size=1,
            num_workers=0, enable_bf16=True,
        )
        processor = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = (
            build_dataset(processor, **params)
        )
        data_dir = Path(params["data_root"]) / params["dataset_id"]
        fmap = FeatureMap(params["dataset_id"], str(data_dir))
        fmap.load(str(data_dir / "feature_map.json"), params)
        model = model_zoo.RankMixer(fmap, **params).cuda()
        checkpoint = (
            ROOT / "runs_local/checkpoints_rankmixer_sample_trust"
            / params["dataset_id"] / f"{expid}.model"
        )
        model.load_weights(str(checkpoint))
        model.eval()
        _, loader = RankDataLoader(
            fmap, stage="train", **params
        ).make_iterator()
        trust_batches = []
        step_batches = []
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= 8:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(batch)
                trust_batches.append(output["__sample_trust"].detach().float().cpu())
                step_batches.append(output["__loop_steps"].detach().float().cpu())
        trust = torch.cat(trust_batches, dim=0)
        steps = torch.cat(step_batches, dim=0)
        stages = []
        for stage in range(trust.size(1)):
            values = trust[:, stage]
            quantiles = torch.quantile(
                values, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99])
            )
            stages.append({
                "stage": stage,
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
                "min": float(values.min()),
                "max": float(values.max()),
                "q01": float(quantiles[0]),
                "q05": float(quantiles[1]),
                "q50": float(quantiles[2]),
                "q95": float(quantiles[3]),
                "q99": float(quantiles[4]),
                "fraction_below_099": float((values < 0.99).float().mean()),
                "fraction_below_095": float((values < 0.95).float().mean()),
                "relative_step_mean": float(steps[:, stage].mean()),
            })
        results[variant] = {
            "num_samples": trust.size(0),
            "stages": stages,
        }
        del model, trust, steps
        torch.cuda.empty_cache()
    out = ROOT.parent / "reports/rankmixer_sample_trust_gate_analysis.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
