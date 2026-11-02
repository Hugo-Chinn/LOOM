#!/usr/bin/env python3
"""Warm-start multi-graph ensemble and matched one-step multi-teacher KD probe."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extend_one_step_distillation import RankMixer, SingleTarget, SPECS, tokens
import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import roc_auc_score, log_loss
from unirank.utils import load_config
from unirank.features import FeatureMap
from unirank.pytorch.dataloaders import RankDataLoader

OUT = ROOT / "reports/multigraph_distillation_20261026"


def metric(y, p):
    return {"auc": [float(roc_auc_score(y[:, i], p[:, i])) for i in range(y.shape[1])],
            "logloss": [float(log_loss(y[:, i], p[:, i])) for i in range(y.shape[1])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*SPECS, "taobao"], required=True)
    ap.add_argument("--historical", action="store_true")
    args = ap.parse_args()
    study = "historical_ensemble_distillation_20261026" if args.historical else "multigraph_distillation_20261026"
    out = ROOT / "reports" / study / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise RuntimeError("Refuse completed overwrite")
    torch.set_num_threads(2)
    torch.manual_seed(20262028)
    cfg, expid, target = (SPECS[args.dataset] if args.dataset != "taobao" else
                          ("config_rankmixer_viz", "RankMixer_Taobao_Branch_S20262028", "buy"))
    historical = None
    if args.historical:
        if args.dataset == "taac":
            raise ValueError("No prior validated TAAC ensemble")
        if args.dataset == "taobao":
            prior = ROOT / "reports/topology_ensemble_taobao_20261026/results.json"
        else:
            prior = ROOT / "reports/ensemble_extension_20261026" / args.dataset / "results.json"
        historical = json.loads(prior.read_text())
        base_name = "branch" if args.dataset == "taobao" else "dag"
        base = next(m for m in historical["models"] if m["name"] == base_name)
        cfg, expid = base["config"], base["expid"]
    params = load_config(str(ROOT / cfg), expid)
    ckpt = ROOT / params["model_root"] / params["dataset_id"] / (expid+".model")
    if historical:
        ckpt = Path(base["checkpoint"])
        torch.manual_seed(historical["seed"])
    params.update(gpu=0, distributed=False, enable_torch_compile=False, enable_bf16=True,
                  num_workers=2, verbose=0, model_root=str(
                      ROOT / "runs_local" / study / args.dataset))
    save = Path(params["model_root"])
    save.mkdir(parents=True, exist_ok=True)
    fmap = FeatureMap(params["dataset_id"], str(Path(params["data_root"])/params["dataset_id"]))
    fmap.load(str(Path(fmap.data_dir)/"feature_map.json"), params)
    index = fmap.labels.index(target)
    weights = {k.replace("._orig_mod.", "."): v for k, v in
               torch.load(ckpt, map_location="cpu", weights_only=True).items()}
    teachers = [RankMixer(fmap, **params)]
    teachers[0].load_state_dict(weights, strict=True)
    teachers[0].eval().requires_grad_(False)
    labels = fmap.labels
    order = list(range(len(labels)))
    reverse = [i for i in reversed(order) if i != index]+[index]
    graph_orders = {"chain": order, "reverse_chain_target_last": reverse}
    graphs = {
        name: {labels[node]: ([] if pos == 0 else [labels[seq[pos-1]]])
               for pos, node in enumerate(seq)}
        for name, seq in graph_orders.items()
    }
    root = next(i for i in order if i != index)
    graphs["star"] = {labels[i]: ([] if i == root else [labels[root]]) for i in order}
    # Deduplicate topologies; two-task datasets have limited structural diversity.
    unique = {}
    for name, graph in graphs.items():
        if graph not in unique.values():
            unique[name] = graph
    names = ["original"]
    provenance = []
    if historical:
        teachers = []
        names = []
        unique = {}
        for member in historical["models"]:
            path = Path(member["checkpoint"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == member["sha256"], path
            p = load_config(str(ROOT / member["config"]), member["expid"])
            p.update(gpu=0, distributed=False, enable_torch_compile=False,
                     enable_bf16=True, num_workers=2, verbose=0,
                     model_root=str(save))
            m = RankMixer(fmap, **p)
            state = {k.replace("._orig_mod.", "."): v for k, v in
                     torch.load(path, map_location="cpu", weights_only=True).items()}
            m.load_state_dict(state, strict=True)
            m.eval().requires_grad_(False)
            teachers.append(m)
            names.append(member["name"])
            provenance.append(member)
        print("HISTORICAL", names, flush=True)
    for name, graph in unique.items():
        p = {**params, "loop_topology": "dag", "behavior_graph": graph,
             "dag_parent_fusion": "mean"}
        m = RankMixer(fmap, **p)
        missing, unexpected = m.load_state_dict(weights, strict=False)
        if any(k not in {"branch_fusion_logits", "dag_fusion_logits"} for k in unexpected):
            raise RuntimeError(f"Unexpected keys: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing warm-start tensors: {missing}")
        for module in (m.embedding_layer, m.unified_tokenizer_layer, m.attention_layers):
            module.requires_grad_(False)
        m._compile_optimizers()
        train, valid = RankDataLoader(fmap, stage="train", **params).make_iterator()
        del valid
        m.train()
        for step, batch in enumerate(train):
            if step >= 256:
                break
            m._optimizer_zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = m(batch)
            y = m.get_labels(batch)
            loss = sum(nn.functional.binary_cross_entropy(
                pred[t+"_pred"].float().clamp(1e-6, 1-1e-6), y[i])
                       for i, t in enumerate(labels))
            loss.backward()
            m._optimizer_step()
        m.eval().requires_grad_(False)
        torch.save(m.state_dict(), save/(name+".model"))
        teachers.append(m)
        names.append(name)
        print("ADAPTED", name, flush=True)

    def evaluate_models(models, split, limit, single=False):
        if split == "valid":
            train, iterator = RankDataLoader(fmap, stage="train", **params).make_iterator()
            del train
        else:
            iterator = RankDataLoader(fmap, stage="test", **params).make_iterator()
        ys, ps = [], [[] for _ in models]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for step, batch in enumerate(iterator):
                if step >= limit:
                    break
                ys.append(torch.cat(teachers[0].get_labels(batch), dim=1).cpu().numpy())
                for i, m in enumerate(models):
                    pred = m(batch)
                    p = pred if single else torch.cat([pred[t+"_pred"] for t in labels], dim=1)
                    ps[i].append(p.float().cpu().numpy())
        return np.concatenate(ys), np.stack([np.concatenate(p) for p in ps])

    valid_batches = 128 if historical else 64
    test_batches = 256 if historical else 128
    yv, pv = evaluate_models(teachers, "valid", valid_batches)
    yt, pt = evaluate_models(teachers, "test", test_batches)
    best_single = max(range(len(names)), key=lambda i: np.mean(metric(yv, pv[i])["auc"]))
    subsets = []
    selected_weights = None
    selected_score = -1
    for count in range(1, len(names)+1):
        for subset in itertools.combinations(range(len(names)), count):
            val = pv[list(subset)].mean(axis=0)
            test = pt[list(subset)].mean(axis=0)
            subsets.append({"members": [names[i] for i in subset],
                            "valid": metric(yv, val), "test": metric(yt, test)})
    # Only target validation selects multi-teacher weights; positive weights >=.25.
    units = 10 if historical else 4
    for w in itertools.product(range(units+1), repeat=len(names)):
        if sum(w) != units or sum(x > 0 for x in w) < 2:
            continue
        w = np.array(w)/units
        score = roc_auc_score(yv[:, index], w @ pv[:, :, index])
        if score > selected_score:
            selected_score, selected_weights = score, w
    single_teacher = max(range(len(names)), key=lambda i: roc_auc_score(yv[:, index], pv[i, :, index]))
    ensemble_test = selected_weights @ pt[:, :, index]
    np.savez_compressed(out/"teacher_predictions.npz", valid_labels=yv, test_labels=yt,
                        valid_predictions=pv, test_predictions=pt)

    students = []
    for _ in range(3):
        s = SingleTarget(fmap, **params)
        s.load_state_dict(weights, strict=True)
        s.tower = nn.ModuleList([s.tower[index]])
        s.output_activation = nn.ModuleList([s.output_activation[index]])
        s.behavior_embedding = nn.Embedding.from_pretrained(
            s.behavior_embedding.weight[index:index+1].detach().clone(), freeze=False)
        if s.loop_gate_logits is not None:
            s.loop_gate_logits = nn.Parameter(s.loop_gate_logits[index:index+1].detach().clone())
        s.num_tasks = s.num_loops = 1
        s.num_preloops = 0
        s.loop_topology = "linear"
        for key in ("branch_fusion_logits", "dag_fusion_logits", "preloop_gate_logits",
                    "base_bypass_logits"):
            s.register_parameter(key, None)
        s.base_bypass_mask = None
        for module in (s.embedding_layer, s.unified_tokenizer_layer, s.attention_layers):
            module.requires_grad_(False)
        s._compile_optimizers()
        students.append(s)
    arm_names = ["label", "single_teacher", "multi_teacher"]
    best = [-1]*3
    validations = [[] for _ in students]
    train, valid = RankDataLoader(fmap, stage="train", **params).make_iterator()
    del valid
    for step, batch in enumerate(train):
        if step >= 512:
            break
        y = teachers[0].get_labels(batch)[index]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            soft = torch.stack([m(batch)[target+"_pred"].float() for m in teachers])
            multi = sum(float(w)*soft[i] for i, w in enumerate(selected_weights))
        for i, s in enumerate(students):
            s.train()
            s._optimizer_zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = s(batch).float().clamp(1e-6, 1-1e-6)
            loss = nn.functional.binary_cross_entropy(p, y)
            if i:
                loss = .1*loss+.9*nn.functional.binary_cross_entropy(
                    p, (soft[single_teacher] if i == 1 else multi).clamp(1e-6, 1-1e-6))
            loss.backward()
            s._optimizer_step()
        if step+1 in (128, 512):
            for s in students:
                s.eval()
            labels_valid, preds = evaluate_models(students, "valid", valid_batches, single=True)
            for i, s in enumerate(students):
                row = {"step": step+1, "auc": float(roc_auc_score(
                    labels_valid[:, index], preds[i, :, 0]))}
                validations[i].append(row)
                if row["auc"] > best[i]:
                    best[i] = row["auc"]
                    torch.save(s.state_dict(), save/(arm_names[i]+".model"))
            print("STUDENT_VALID", step+1, validations, flush=True)
    for i, s in enumerate(students):
        s.load_state_dict(torch.load(save/(arm_names[i]+".model"), weights_only=True))
        s.eval()
    ys, preds = evaluate_models(students, "test", test_batches, single=True)
    np.testing.assert_array_equal(ys, yt)
    iterator = RankDataLoader(fmap, stage="test", **params).make_iterator()
    batch = next(iter(iterator))
    flops = []
    for m in [*teachers, students[2]]:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with FlopCounterMode(display=False) as counter:
                m(batch)
        flops.append(counter.get_total_flops()/ys[:8192].shape[0]/1e9)
    result = {"dataset": params["dataset_id"], "target": target,
              "seed": historical["seed"] if historical else 20262028,
              "historical_members": provenance,
              "names": names, "graphs": unique, "teacher_subsets": subsets,
              "validation_best_macro_single": names[best_single],
              "validation_best_target_single": names[single_teacher],
              "target_multi_weights": selected_weights.tolist(),
              "ensemble_target_test": {"auc": float(roc_auc_score(yt[:, index], ensemble_test)),
                                      "logloss": float(log_loss(yt[:, index], ensemble_test))},
              "student_validation": dict(zip(arm_names, validations)),
              "student_test": {name: {"auc": float(roc_auc_score(yt[:, index], preds[i, :, 0])),
                                     "logloss": float(log_loss(yt[:, index], preds[i, :, 0]))}
                               for i, name in enumerate(arm_names)},
              "teacher_forward_gflops": dict(zip(names, flops[:-1])),
              "student_forward_gflops": flops[-1],
              "samples": {"valid": len(yv), "test": len(yt)},
              "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
              "note": ("Verified retained historical members; no teacher adaptation; target "
                       "validation AUC selects weights on .1 grid; at least two positive "
                       "weights for multi-teacher experimental arm, not production selection; "
                       "matched original initialization and student budgets; previously "
                       "inspected prefixes, no independent confirmation."
                       if historical else
                       "Original historical teacher plus 256-batch warm-start graph adaptations; "
                      "graph adaptations use main losses only; teacher budgets not matched; "
                      "prefixes previously explored, no independent confirmation or pure "
                      "topology attribution; student512 batches, validation selection only.")}
    (out/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print("DONE", result["student_test"], flush=True)


if __name__ == "__main__":
    main()
