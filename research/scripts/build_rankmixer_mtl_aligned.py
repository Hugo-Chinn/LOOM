#!/usr/bin/env python3
"""Freeze historical RankMixer settings and separate head/replication studies."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_mtl_aligned"
SEEDS = (20262027, 20262028, 20262029, 20262030)
REFERENCES = {
    "TencentGR_10M_Action": ("config_rankmixer_multiseed_confirmation",
                           "RankMixer_TAAC_Parallel_S20262028",
                           "RankMixer_TAAC_Loop_Deep_S20262028",
                           "RankMixer_TAAC_Loop_GatedDeep_Pre2_S20262028"),
    "QK_Video_Action": ("config_rankmixer_multiseed_confirmation",
                       "RankMixer_QK_Parallel_S20262028",
                       "RankMixer_QK_Linear_S20262028",
                       "RankMixer_QK_DAGGated_S20262028"),
    "KuaiRand_Video_Action": ("config_rankmixer_multiseed_confirmation",
                            "RankMixer_Kuai_Parallel_S20262028",
                            "RankMixer_Kuai_Linear_S20262028",
                            "RankMixer_Kuai_DAGMeanPre2_S20262028"),
    "MerRec_Action": ("config_rankmixer_multiseed_confirmation",
                     "RankMixer_MerRec_Parallel_S20262028",
                     "RankMixer_MerRec_LinearGated_S20262028",
                     "RankMixer_MerRec_DAGGated_S20262028"),
    "Taobao_Action": ("config_rankmixer_taobao_raw_multiseed",
                     "RankMixer_Taobao_Raw_Seed20262028", None, None),
}
HEADS = {
    "ParallelIndependentTower": {},
    "SharedBottom": {"head_sharing": "independent"},
    "MMoE": {"num_experts": 4, "expert_hidden_units": [256]},
    "PLE": {"num_levels": 2, "num_shared_experts": 2,
            "num_task_experts": 2, "expert_hidden_units": [256]},
    "STEM": {"num_shared_experts": 2, "num_task_experts": 2,
             "expert_hidden_units": [256], "stem_fusion": "aftb",
             "task_embedding": "residual", "private_rank": 16},
    "MoME": {"num_experts": 4, "base_hidden_units": [256],
             "mask_sparsity_weight": 1e-4},
    "ResFlow": {"fusion_mode": "additive", "stop_gradient": False},
    "OneRank": {"head_num_layers": 2, "num_heads": 4,
                "candidate_pool_mode": "none"},
    "LoopMT": {},
    "SingleStep": {"num_loops": 1, "use_stage_embedding": False,
                   "deep_supervision": False, "readout": "final"},
    "UntiedDeep": {"share_loop_parameters": False},
    "LoopNoAux": {"deep_supervision": False},
    "LoopFinal": {"readout": "final", "deep_supervision": False},
    "LoopNoStage": {"use_stage_embedding": False},
}
BLOCKED = {
    "OneRank": "Exact generative objective and official reference not verified.",
    "MoME": "Current activation-mask head is not a verified weight-mask reproduction.",
    "ResFlow": "Paper residual/score fusion formulation not verified; current variant multiplies odds.",
}
LOOP = {
    "share_loop_parameters": True, "refinement_layers": 1,
    "refinement_expansion": 2, "use_stage_embedding": True,
    "stage_embedding_init": "default", "readout": "progressive",
    "deep_supervision": True, "auxiliary_loss_weight": 0.5,
    "num_preloops": 0, "use_loop_gate": False,
}
BACKBONE_KEYS = (
    "embedding_dim", "token_dim", "num_ns_token", "num_layers",
    "expansion_factor", "tokenizer_type", "attention_dim", "attention_dropout",
    "attention_activation_type", "net_dropout", "tower_hidden_units",
    "tower_activations", "batch_size", "max_len", "epochs", "accumulation_steps",
    "shuffle", "task", "loss", "monitor", "monitor_mode", "metrics",
)


def load(family):
    return yaml.safe_load((ROOT / family / "model_config.yaml").read_text())


def fingerprint():
    paths = sorted(set(ROOT.glob("model_zoo/*.py")) |
                   set(ROOT.glob("unirank/**/*.py")) |
                   {ROOT / "run_expid.py", Path(__file__).resolve(),
                    ROOT / "scripts/run_rankmixer_mtl_aligned.py",
                    ROOT / "scripts/queue_rankmixer_mtl_aligned.py",
                    ROOT / "scripts/summarize_all_local_experiments.py"})
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.exists()}


def build(out):
    historical = load("config_rankmixer_multiseed_confirmation")
    taobao = load("config_rankmixer_multiseed")
    base = deepcopy(historical["Base"])
    base["model_root"] = "./runs_local/checkpoints_rankmixer_mtl_aligned/"
    config = {"Base": base}
    datasets, arms = {}, {}
    for ds, (family, parallel_id, linear_id, finalist_id) in REFERENCES.items():
        source = load(family)
        parallel = deepcopy(source[parallel_id])
        source_data = yaml.safe_load((ROOT / family / "dataset_config.yaml").read_text())
        datasets[ds] = source_data[ds]
        if ds == "Taobao_Action":
            linear = taobao["RankMixer_LinearDeep_Seed20262028"]
            finalist = taobao["RankMixer_BranchGated_Seed20262028"]
        else:
            linear, finalist = source[linear_id], source[finalist_id]
        for seed in SEEDS:
            for variant, template in (
                    ("Parallel", parallel), ("LinearDeep", linear), ("Finalist", finalist)):
                expid = f"History_{variant}_{ds}_S{seed}"
                config[expid] = {**deepcopy(template), "seed": seed}
                arms[expid] = {
                    "study": "historical_replication", "variant": variant,
                    "dataset": ds, "seed": seed, "status": "ready",
                    "source": family if ds != "Taobao_Action" or variant == "Parallel"
                    else "config_rankmixer_multiseed",
                    "control": f"History_Parallel_{ds}_S{seed}",
                }
            for variant, overrides in HEADS.items():
                expid = f"Head_{variant}_{ds}_S{seed}"
                model = variant if variant in (
                    "ParallelIndependentTower", "SharedBottom", "MMoE", "PLE",
                    "STEM", "MoME", "ResFlow", "OneRank", "LoopMT") else "LoopMT"
                item = {k: deepcopy(parallel[k]) for k in BACKBONE_KEYS if k in parallel}
                item.update(model=model, dataset_id=ds, seed=seed,
                            bottom_type="mixer", task_token_mode="none")
                if model == "LoopMT":
                    item.update(LOOP, num_loops=parallel["num_tasks"])
                item["num_tasks"] = parallel["num_tasks"]
                item.update(overrides)
                config[expid] = item
                status = "blocked_reference" if variant in BLOCKED else "ready"
                if variant == "SharedBottom":
                    status = "alias"
                arms[expid] = {
                    "study": "fixed_backbone_head", "variant": variant,
                    "dataset": ds, "seed": seed, "status": status,
                    "source": f"{family}/{parallel_id}",
                    "control": f"Head_ParallelIndependentTower_{ds}_S{seed}",
                    "note": BLOCKED.get(variant, (
                        "AFTB routing with a pooled-representation adapter, not private embeddings."
                        if variant == "STEM" else
                        "Identical to ParallelIndependentTower; no duplicate training."
                        if variant == "SharedBottom" else "")),
                }
    manifest = {
        "protocol_version": 1, "seeds": list(SEEDS),
        "seed_policy": "Historical screen seed 20262027 + confirmation seeds 20262028-30.",
        "source_hashes": fingerprint(), "arms": arms,
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (("model_config.yaml", config), ("dataset_config.yaml", datasets)):
        (out / name).write_text(yaml.safe_dump(value, sort_keys=False))
    manifest["config_sha256"] = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ("model_config.yaml", "dataset_config.yaml")
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for study in ("historical_replication", "fixed_backbone_head"):
        ready = [n for n, a in arms.items() if a["study"] == study and a["status"] == "ready"]
        (out / f"{study}_expids.txt").write_text("\n".join(ready) + "\n")
    print(f"{len(arms)} configured; {sum(a['status'] == 'ready' for a in arms.values())} ready -> {out}")
    return config, datasets, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    build(parser.parse_args().out)
