#!/usr/bin/env python3
"""Instantiate official RankMixer and run one real TAAC blocked batch."""
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


params = load_config(str(ROOT / "config_local"),
                     "RankMixer_TencentGR_10M_Action_Local")
params.update(gpu=0, distributed=False, rank=0, local_rank=0, world_size=1,
              enable_bf16=True, num_workers=0, enable_torch_compile=False)

processor = FeatureProcessor(**params)
params["train_data"], params["valid_data"], params["test_data"] = \
    build_dataset(processor, **params)
feature_map = FeatureMap(params["dataset_id"],
                         str(Path(params["data_root"]) / params["dataset_id"]))
feature_map.load(str(Path(params["data_root"]) / params["dataset_id"] /
                     "feature_map.json"), params)
model = model_zoo.RankMixer(feature_map, **params)

loader, _ = RankDataLoader(feature_map, stage="train", **params).make_iterator()
batch = next(iter(loader))
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(batch)
# The official training loop exits autocast before probability-space BCE.
labels = model.get_labels(batch)
loss = model.compute_loss(
    {key: value.float() for key, value in output.items()}, labels
)
loss.backward()

total = sum(p.numel() for p in model.parameters())
embedding = sum(p.numel() for p in model.embedding_layer.parameters())
result = {
    "total_parameters": total,
    "embedding_parameters": embedding,
    "dense_parameters": total - embedding,
    "official_total_parameters": 490447298,
    "parameter_delta": total - 490447298,
    "batch_size": int(labels[0].shape[0]),
    "tasks": list(feature_map.labels),
    "output_shapes": {key: list(value.shape) for key, value in output.items()},
    "loss": float(loss.detach()),
    "finite_gradients": all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        if p.grad is not None
    ),
    "mask_shape": list(batch[2].shape),
    "history_nonzero": int(batch[2].sum()),
}
out = ROOT.parent / "reports/unirank_taac_real_batch_validation.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
