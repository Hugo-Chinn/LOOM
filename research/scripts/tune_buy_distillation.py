#!/usr/bin/env python3
"""Validation-selected one-step Buy KD sweep with fixed budgets and held-out test."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_buy_one_step_distillation import RankMixer, BuyStudent
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.flop_counter import FlopCounterMode
from unirank.utils import load_config
from unirank.features import FeatureMap
from unirank.pytorch.dataloaders import RankDataLoader

OUT = ROOT / "reports/buy_distillation_tuning"
ARMS = [
    {"name": "label_only", "alpha": 0, "temperature": 1, "feature_weight": 0},
    {"name": "kd50", "alpha": 0.5, "temperature": 1, "feature_weight": 0},
    {"name": "kd90", "alpha": 0.9, "temperature": 1, "feature_weight": 0},
    {"name": "kd90_T2", "alpha": 0.9, "temperature": 2, "feature_weight": 0},
    {"name": "kd90_T4", "alpha": 0.9, "temperature": 4, "feature_weight": 0},
    {"name": "kd90_repr", "alpha": 0.9, "temperature": 1, "feature_weight": 0.1},
    {"name": "kd90_frozen_front", "alpha": 0.9, "temperature": 1,
     "feature_weight": 0, "freeze_front": True},
    {"name": "label_frozen_front", "alpha": 0, "temperature": 1,
     "feature_weight": 0, "freeze_front": True},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=int, required=True)
    parser.add_argument("--evaluate-selected", action="store_true")
    args = parser.parse_args()
    arm = ARMS[args.arm]
    OUT.mkdir(exist_ok=True)
    out = OUT / (arm["name"] + ".json")
    if out.exists() and not args.evaluate_selected:
        raise RuntimeError(f"Refuse to overwrite {out}")
    torch.set_num_threads(2)
    torch.manual_seed(20262028)
    params = load_config(str(ROOT / "config_rankmixer_viz"),
                         "RankMixer_Taobao_Branch_S20262028")
    params.update(gpu=0, enable_torch_compile=False, distributed=False,
                  enable_bf16=True, num_workers=2, verbose=0,
                  model_root=str(ROOT / "runs_local/buy_distillation_tuning"))
    ds = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(ds))
    fmap.load(str(ds / "feature_map.json"), params)
    checkpoint = ROOT / "runs_local/checkpoints_rankmixer_viz/Taobao_Action/RankMixer_Taobao_Branch_S20262028.model"
    state = {k.replace("._orig_mod.", "."): v for k,v in
             torch.load(checkpoint, map_location="cpu", weights_only=True).items()}
    teacher = RankMixer(fmap, **params)
    teacher.load_state_dict(state, strict=True)
    teacher.eval().requires_grad_(False)
    index = fmap.labels.index("buy")
    student = BuyStudent(fmap, **params)
    student.load_state_dict(state, strict=True)
    student.tower = nn.ModuleList([student.tower[index]])
    student.behavior_embedding = nn.Embedding.from_pretrained(
        student.behavior_embedding.weight[index:index+1].detach().clone(), freeze=False)
    student.num_tasks = student.num_loops = 1
    student.loop_topology = "linear"
    student.supervision_mode = "progressive"
    student.register_parameter("branch_fusion_logits", None)
    if arm.get("freeze_front"):
        for module in (student.embedding_layer, student.unified_tokenizer_layer,
                       student.attention_layers):
            module.requires_grad_(False)
    student._compile_optimizers()
    captured = {}
    teacher.tower[index].register_forward_pre_hook(
        lambda module, inputs: captured.update(teacher=inputs[0].detach()))
    student.tower[0].register_forward_pre_hook(
        lambda module, inputs: captured.update(student=inputs[0]))

    def evaluate(iterator, limit):
        labels, p_t, p_s = [], [], []
        student.eval()
        with torch.inference_mode():
            for step, batch in enumerate(iterator):
                if step >= limit:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    a, b = teacher(batch)["buy_pred"], student(batch)
                labels.extend(teacher.get_labels(batch)[index].cpu().flatten().tolist())
                p_t.extend(a.float().cpu().flatten().tolist())
                p_s.extend(b.float().cpu().flatten().tolist())
        return {
            "n": len(labels), "positives": sum(labels),
            "teacher_auc": roc_auc_score(labels, p_t),
            "teacher_logloss": log_loss(labels, p_t),
            "student_auc": roc_auc_score(labels, p_s),
            "student_logloss": log_loss(labels, p_s),
        }

    best_path = ROOT / "runs_local/buy_distillation_tuning" / (arm["name"] + ".model")
    best_path.parent.mkdir(exist_ok=True)
    if args.evaluate_selected:
        student.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True), strict=True)
        test = RankDataLoader(fmap, stage="test", **params).make_iterator()
        metrics = evaluate(test, 256)
        (OUT / (arm["name"] + "_heldout_test.json")).write_text(json.dumps({
            "arm": arm, "metrics": metrics,
            "note": "Exploratory expanded test prefix; original probe already inspected part of test.",
        }, indent=2)+"\n")
        print(json.dumps(metrics), flush=True)
        return
    train, valid = RankDataLoader(fmap, stage="train", **params).make_iterator()
    validations = []
    best_auc = -1
    start = time.monotonic()
    samples = 0
    for step, batch in enumerate(train):
        if step >= 512:
            break
        student.train()
        targets = teacher.get_labels(batch)[index]
        student._optimizer_zero_grad()
        if arm["alpha"] or arm["feature_weight"]:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p_t = teacher(batch)["buy_pred"].float().clamp(1e-6, 1-1e-6)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p_s = student(batch).float().clamp(1e-6, 1-1e-6)
        loss = (1-arm["alpha"])*nn.functional.binary_cross_entropy(p_s, targets)
        if arm["alpha"]:
            temperature = arm["temperature"]
            q_t = torch.sigmoid(torch.logit(p_t)/temperature)
            q_s = torch.sigmoid(torch.logit(p_s)/temperature)
            # Bernoulli KL differs from BCE by a teacher-only constant.
            loss = loss + arm["alpha"]*temperature**2*nn.functional.binary_cross_entropy(q_s, q_t)
        if arm["feature_weight"]:
            a = nn.functional.normalize(captured["teacher"].float(), dim=-1)
            b = nn.functional.normalize(captured["student"].float(), dim=-1)
            loss = loss + arm["feature_weight"]*(1-(a*b).sum(-1)).mean()
        loss.backward()
        student._optimizer_step()
        samples += targets.numel()
        if (step+1) % 64 == 0:
            print(f"{arm['name']} step={step+1}/512", flush=True)
        if step+1 in (128, 512):
            metrics = evaluate(valid, 128)
            metrics["train_batches"] = step+1
            validations.append(metrics)
            if metrics["student_auc"] > best_auc:
                best_auc = metrics["student_auc"]
                torch.save(student.state_dict(), best_path)
            print("VALID", json.dumps(metrics), flush=True)
    result = {
        "arm": arm, "seed": 20262028, "train_samples": samples,
        "train_plus_validation_wall_seconds": time.monotonic()-start,
        "validation": validations,
        "selection": "highest validation Buy AUC of fixed 128/512-batch checkpoints",
        "test_touched": False, "student_checkpoint": str(best_path),
        "teacher_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out.write_text(json.dumps(result, indent=2)+"\n")
    print("DONE", arm["name"], flush=True)


if __name__ == "__main__":
    main()
