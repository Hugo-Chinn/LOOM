#!/usr/bin/env python3
"""Real-batch CPU sanity check for cross-dataset OneTrans variants."""
from __future__ import annotations

import argparse
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


DATASETS = (
    "TencentGR_10M_Action",
    "Taobao_Action",
    "QK_Video_Action",
    "MerRec_Action",
    "KuaiRand_Video_Action",
)
VARIANTS = ("Baseline", "FiLM", "QVLoRA")
SEED = 20262027


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="*", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument(
        "--output", type=Path,
        default=ROOT.parent / "reports/onetrans_all_datasets_sanity.json")
    args = parser.parse_args()
    config_dir = ROOT / "config_onetrans_all_datasets_multiseed"
    results = []
    for dataset_id in args.datasets:
        for variant in args.variants:
            expid = f"OneTransAll_{dataset_id}_{variant}_S{SEED}"
            print(f"[sanity] start {expid}", flush=True)
            params = load_config(str(config_dir), expid)
            params.update(
                gpu=-1,
                distributed=False,
                rank=0,
                local_rank=0,
                world_size=1,
                enable_bf16=False,
                enable_torch_compile=False,
                num_workers=0,
                batch_size=2,
                max_len=8,
            )
            processor = FeatureProcessor(**params)
            params["train_data"], params["valid_data"], params["test_data"] = \
                build_dataset(processor, **params)
            data_dir = Path(params["data_root"]) / params["dataset_id"]
            feature_map = FeatureMap(params["dataset_id"], str(data_dir))
            feature_map.load(str(data_dir / "feature_map.json"), params)
            model = model_zoo.OneTrans(feature_map, **params)
            loader, _ = RankDataLoader(
                feature_map, stage="train", **params).make_iterator()
            batch = next(iter(loader))
            output = model(batch)
            labels = model.get_labels(batch)
            loss = model.compute_loss(output, labels)
            loss.backward()
            results.append({
                "expid": expid,
                "dataset": dataset_id,
                "variant": variant,
                "num_tasks": model.num_tasks,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad),
                "output_shapes": {
                    key: list(value.shape)
                    for key, value in output.items() if not key.startswith("__")
                },
                "loss": float(loss.detach()),
                "finite_gradients": all(
                    bool(torch.isfinite(p.grad).all())
                    for p in model.parameters() if p.grad is not None
                ),
            })
            print(
                f"[sanity] passed {expid}: loss={results[-1]['loss']:.6f}, "
                f"params={results[-1]['parameters']:,}",
                flush=True,
            )
            del output, loss, labels, batch, loader, model, feature_map, processor
            gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
