#!/usr/bin/env python3
"""Validation-selected prediction ensemble of retained native Loop checkpoints."""
import hashlib
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_buy_one_step_distillation import RankMixer
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.flop_counter import FlopCounterMode
from unirank.features import FeatureMap
from unirank.utils import load_config
from unirank.pytorch.dataloaders import RankDataLoader

OUT = ROOT / "reports/topology_ensemble_taobao_20261026"
SPECS = [
    ("linear", "config_viz_taobao_rankmixer_heads",
     "History_LinearDeep_Taobao_Action_S20262027", None),
    ("branch", "config_viz_taobao_rankmixer_heads",
     "History_Finalist_Taobao_Action_S20262027", None),
    ("dag", "config_rankmixer_structure_ablation_dag", "DAGTrue_S20262027",
     "runs_local/checkpoints_rankmixer_structure_probe/Taobao_Action/DAGTrue_S20262027.model"),
]


def metrics(y, p):
    return {"auc": [roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1])],
            "logloss": [log_loss(y[:, i], p[:, i]) for i in range(y.shape[1])]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["taobao", "kuai", "merrec", "qk"],
                        default="taobao")
    args = parser.parse_args()
    out = OUT if args.dataset == "taobao" else (
        ROOT / "reports/ensemble_extension_20261026" / args.dataset)
    specs = SPECS
    seed = 20262027
    if args.dataset != "taobao":
        seed = 20262028
        short, loop = {"kuai": ("Kuai", "DAGMeanPre2"),
                       "merrec": ("MerRec", "DAGGated"),
                       "qk": ("QK", "DAGGated")}[args.dataset]
        specs = [
            ("parallel", "config_rankmixer_viz",
             f"RankMixer_{short}_Parallel_S{seed}", None),
            ("dag", "config_rankmixer_viz",
             f"RankMixer_{short}_{loop}_S{seed}", None),
        ]
        if args.dataset == "kuai":
            specs.append(("bypass", "config_rankmixer_bypass_rare",
                          f"RankMixer_Kuai_BypassRare_S{seed}", None))
        elif args.dataset == "merrec":
            specs.append(("trust", "config_rankmixer_sample_trust",
                          f"RankMixer_MerRec_SampleTrustShared_S{seed}", None))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise RuntimeError("Refuse to overwrite completed experiment")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    predictions, truths, provenance, flops = {}, {}, [], []
    for name, cfg, expid, override in specs:
        params = load_config(str(ROOT / cfg), expid)
        ckpt = ROOT / override if override else (
            ROOT / params["model_root"] / params["dataset_id"] / (expid + ".model"))
        params.update(gpu=0, enable_torch_compile=False, enable_bf16=True,
                      distributed=False, num_workers=2, verbose=0,
                      model_root=str(out / "unused_model_root"))
        data = Path(params["data_root"]) / params["dataset_id"]
        fmap = FeatureMap(params["dataset_id"], str(data))
        fmap.load(str(data / "feature_map.json"), params)
        model = RankMixer(fmap, **params)
        state = {k.replace("._orig_mod.", "."): v for k, v in
                 torch.load(ckpt, map_location="cpu", weights_only=True).items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        predictions[name] = {}
        for split, limit in (("valid", 128), ("test", 256)):
            if split == "valid":
                train, iterator = RankDataLoader(fmap, stage="train", **params).make_iterator()
                del train
            else:
                iterator = RankDataLoader(fmap, stage="test", **params).make_iterator()
            ys, ps = [], []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for step, batch in enumerate(iterator):
                    if step >= limit:
                        break
                    output = model(batch)
                    ps.append(torch.cat([output[t+"_pred"] for t in fmap.labels],
                                        dim=1).float().cpu().numpy())
                    ys.append(torch.cat(model.get_labels(batch), dim=1).cpu().numpy())
                    if step == 0 and split == "valid":
                        with FlopCounterMode(display=False) as counter:
                            model(batch)
                        flops.append(counter.get_total_flops()/ys[-1].shape[0]/1e9)
            y, p = np.concatenate(ys), np.concatenate(ps)
            if split in truths:
                np.testing.assert_array_equal(y, truths[split])
            else:
                truths[split] = y
            predictions[name][split] = p
            print(name, split, json.dumps(metrics(y, p)), flush=True)
            del iterator
        provenance.append({"name": name, "config": cfg, "expid": expid,
                           "checkpoint": str(ckpt),
                           "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                           "topology": params.get("loop_topology"),
                           "model_type": model.model_type,
                           "gate": params.get("use_loop_gate", False)})
        del model, state
        torch.cuda.empty_cache()

    names = [s[0] for s in specs]
    # Nonnegative simplex grid; fit only validation, separately for each task.
    weights = [np.array(w)/10 for w in itertools.product(
        range(11), repeat=len(names)) if sum(w) == 10]
    valid_stack = np.stack([predictions[n]["valid"] for n in names])
    selected = []
    for task in range(len(fmap.labels)):
        best = max(weights, key=lambda w: roc_auc_score(
            truths["valid"][:, task], w @ valid_stack[:, :, task]))
        selected.append(best.tolist())
    best_single = max(names, key=lambda n: np.mean(
        metrics(truths["valid"], predictions[n]["valid"])["auc"]))
    result = {
        "models": provenance, "labels": fmap.labels, "seed": seed,
        "dataset": params["dataset_id"],
        "ensemble_scope": ("native loop topologies with differing gates" if args.dataset == "taobao"
                           else "retained parallel/DAG/configuration ensemble, not topology-only"),
        "selection": "Task-specific nonnegative probability weights on 0.1 simplex grid; "
                     "select by validation task AUC only; no test-based selection.",
        "weights_model_order": names, "selected_task_weights": selected,
        "validation_best_single": best_single, "forward_gflops_per_model": dict(zip(names, flops)),
        "samples": {s: len(y) for s, y in truths.items()},
        "positives": {s: y.sum(axis=0).tolist() for s, y in truths.items()},
        "metrics": {},
        "note": "Exploratory prefixes, previously inspected data, same seed but gates/source "
                "provenance differ; frozen archived model implementation with strict checkpoint "
                "load, not proof of source identity; full all-task forwards, no shared-front "
                "serving optimization or isolated timing.",
    }
    for split in ("valid", "test"):
        stack = np.stack([predictions[n][split] for n in names])
        weighted = np.column_stack([
            np.array(selected[t]) @ stack[:, :, t] for t in range(len(fmap.labels))])
        values = {n: predictions[n][split] for n in names}
        values["equal_mean"] = stack.mean(axis=0)
        values["validation_weighted"] = weighted
        result["metrics"][split] = {n: metrics(truths[split], p) for n, p in values.items()}
    active = np.any(np.array(selected) > 0, axis=0)
    result["weighted_forward_gflops"] = sum(f for f, used in zip(flops, active) if used)
    result["equal_mean_forward_gflops"] = sum(flops)
    (out / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print("DONE", json.dumps(result["metrics"]["test"]), flush=True)


if __name__ == "__main__":
    main()
