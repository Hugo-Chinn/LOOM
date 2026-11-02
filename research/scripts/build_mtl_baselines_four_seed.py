"""Build the four-seed multi-task baseline sweep.

Hyper-parameters per dataset are copied from the ``OneTransAll_*_Baseline``
arms so the routing baselines stay comparable with the runs already on record.
Seeds are the canonical four: 20262027-20262030.

Usage:
    python scripts/build_mtl_baselines_four_seed.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = REPO / "config_onetrans_all_datasets_multiseed"
OUT_DIR = REPO / "config_mtl_baselines_four_seed"

DATASETS = ["TencentGR_10M_Action", "QK_Video_Action", "KuaiRand_Video_Action",
            "MerRec_Action", "Taobao_Action"]
SEEDS = [20262027, 20262028, 20262029, 20262030]

# Per-model overrides. Everything else comes from the dataset template.
MODEL_OVERRIDES = {
    # Both shared-bottom arms keep the RankMixer backbone: they differ only in
    # how much the towers share.
    "SharedBottom": {"bottom_type": "mixer", "head_sharing": "shared_first"},
    "ParallelIndependentTower": {"bottom_type": "mixer"},
    "MMoE": {"num_experts": 4},
    "PLE": {"num_levels": 2, "num_shared_experts": 2, "num_task_experts": 2},
    "STEM": {"num_shared_experts": 2, "num_task_experts": 2,
             "task_embedding": "residual", "private_rank": 16},
    "MoME": {"num_experts": 4, "base_hidden_units": [256],
             "mask_sparsity_weight": 1.0e-4},
    "ResFlow": {"fusion_mode": "additive", "use_residual_flow": True,
                "stop_gradient": False},
    "OneRank": {"bottom_type": "mixer", "head_num_layers": 2, "num_heads": 4,
                "candidate_pool_mode": "batch"},
    "LoopMT": {"readout": "progressive", "deep_supervision": True,
               "auxiliary_loss_weight": 0.5, "refinement_layers": 1},
}
MODELS = list(MODEL_OVERRIDES)

TEMPLATE_KEYS = ("embedding_dim", "token_dim", "num_ns_token", "num_tasks",
                 "num_layers", "batch_size", "max_len", "tokenizer_type",
                 "tower_hidden_units", "tower_activations", "epochs",
                 "accumulation_steps", "shuffle", "net_dropout", "monitor",
                 "monitor_mode", "metrics", "task", "loss")

BASE = {
    "model_root": "./runs_local/checkpoints_mtl_baselines_four_seed/",
    "num_workers": 4,
    "verbose": 1,
    "early_stop_patience": 2,
    "pickle_feature_encoder": True,
    "save_best_only": True,
    "eval_steps": None,
    "debug_mode": False,
    "group_id": "user_id",
    "use_features": None,
    "feature_specs": None,
    "feature_config": None,
    "dense_optimizer": "AdamW",
    "sparse_optimizer": "Adagrad",
    "dense_learning_rate": 1.0e-4,
    "sparse_learning_rate": 0.05,
    "dense_weight_decay": 0,
    "enable_torch_compile": True,
    "gradient_checkpointing": False,
}


def load_dataset_templates() -> dict:
    config = yaml.safe_load((SOURCE_CONFIG / "model_config.yaml").read_text())
    templates = {}
    for name, arm in config.items():
        if name == "Base" or "Baseline" not in name:
            continue
        templates.setdefault(arm["dataset_id"], arm)
    missing = [ds for ds in DATASETS if ds not in templates]
    if missing:
        raise SystemExit(f"no baseline template for {missing}")
    return templates


def build_expids(templates: dict) -> list[str]:
    """Dataset-major ordering so each (dataset, model) completes all seeds first."""
    expids = []
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                expids.append(f"{model}_{dataset}_S{seed}")
    return expids


def arm_for(expid: str, templates: dict) -> dict:
    model, dataset, seed_tag = expid.rsplit("_", 2)[0], None, None
    # expid format: <Model>_<Dataset>_S<seed>; dataset may contain underscores.
    seed_tag = expid.rsplit("_", 1)[1]
    remainder = expid[: -(len(seed_tag) + 1)]
    model = remainder
    for candidate in DATASETS:
        if remainder.endswith("_" + candidate):
            model = remainder[: -(len(candidate) + 1)]
            dataset = candidate
            break
    if dataset is None or model not in MODEL_OVERRIDES:
        raise SystemExit(f"cannot parse expid {expid}")
    template = templates[dataset]
    arm = {key: template[key] for key in TEMPLATE_KEYS if key in template}
    arm["model"] = model
    arm["dataset_id"] = dataset
    arm["seed"] = int(seed_tag.replace("S", ""))
    arm["num_heads"] = template.get("num_heads", 2)
    arm["expansion_factor"] = 2
    arm.update(MODEL_OVERRIDES[model])
    return arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_config.yaml").write_text(
        (SOURCE_CONFIG / "dataset_config.yaml").read_text())

    templates = load_dataset_templates()
    expids = build_expids(templates)
    blocks = [yaml.safe_dump({"Base": BASE}, sort_keys=True, default_flow_style=False)]
    for expid in expids:
        arm = arm_for(expid, templates)
        body = yaml.safe_dump({expid: arm}, sort_keys=True, default_flow_style=False)
        blocks.append(body)
    (out_dir / "model_config.yaml").write_text("\n".join(blocks))
    (out_dir / "expids.txt").write_text("\n".join(expids) + "\n")
    print(f"{len(expids)} arms -> {out_dir}")


if __name__ == "__main__":
    main()
