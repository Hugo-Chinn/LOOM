#!/usr/bin/env python3
"""Small real-data Buy-only one-step distillation probe, not a formal benchmark."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runs_local/source_snapshots/loop_strategy_20261021"
sys.path.insert(0, str(RUNTIME))
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.flop_counter import FlopCounterMode
from unirank.utils import load_config
from unirank.features import FeatureMap
from unirank.pytorch.dataloaders import RankDataLoader
from model_zoo.RankMixer import RankMixer


class BuyStudent(RankMixer):
    def forward(self, inputs):
        output = super().forward(inputs)
        return output[f"{self.feature_map.labels[0]}_pred"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20262028)
    output_dir = ROOT / "reports/buy_distillation_probe"
    output_dir.mkdir(exist_ok=True)
    params = load_config(str(ROOT / "config_rankmixer_viz"),
                         "RankMixer_Taobao_Branch_S20262028")
    params.update(gpu=0, enable_torch_compile=False, distributed=False,
                  enable_bf16=True, num_workers=2, verbose=0,
                  model_root=str(ROOT / "runs_local/buy_distillation_probe"))
    ds = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(ds))
    fmap.load(str(ds / "feature_map.json"), params)
    teacher = RankMixer(fmap, **params)
    checkpoint = ROOT / "runs_local/checkpoints_rankmixer_viz/Taobao_Action/RankMixer_Taobao_Branch_S20262028.model"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    index = fmap.labels.index("buy")
    students = []
    for _ in range(2):
        student = BuyStudent(fmap, **params)
        student.load_state_dict(teacher.state_dict(), strict=True)
        student.tower = nn.ModuleList([student.tower[index]])
        student.behavior_embedding = nn.Embedding.from_pretrained(
            student.behavior_embedding.weight[index:index+1].detach().clone(), freeze=False)
        student.num_tasks = student.num_loops = 1
        student.loop_topology = "linear"
        student.supervision_mode = "progressive"
        student.register_parameter("branch_fusion_logits", None)
        student._compile_optimizers()
        students.append(student)
    teacher.requires_grad_(False)
    names = ["one_step_label_only", "one_step_KD"]
    train, validation = RankDataLoader(fmap, stage="train", **params).make_iterator()
    # Validation is not used for model selection in this small fixed-budget probe.
    del validation
    start = time.monotonic()
    train_samples = 0
    for step, batch in enumerate(train):
        if step >= args.batches:
            break
        targets = teacher.get_labels(batch)[index]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            soft = teacher(batch)["buy_pred"].float()
        for kind, student in enumerate(students):
            student.train()
            student._optimizer_zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = student(batch).float()
            hard_loss = nn.functional.binary_cross_entropy(prediction, targets)
            loss = hard_loss if kind == 0 else (
                0.5 * hard_loss + 0.5 * nn.functional.binary_cross_entropy(prediction, soft))
            loss.backward()
            student._optimizer_step()
        train_samples += targets.numel()
        if step % 16 == 0:
            print(f"TRAIN {step}/{args.batches} samples={train_samples}", flush=True)
    elapsed = time.monotonic() - start
    del train
    test = RankDataLoader(fmap, stage="test", **params).make_iterator()
    if isinstance(test, tuple):
        test = test[0]
    labels, predictions = [], [[], [], []]
    representative = None
    for step, batch in enumerate(test):
        if step >= args.eval_batches:
            break
        representative = batch
        labels.extend(teacher.get_labels(batch)[index].flatten().cpu().tolist())
        for model in students:
            model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = [teacher(batch)["buy_pred"], students[0](batch), students[1](batch)]
        for values, prediction in zip(predictions, outputs):
            values.extend(prediction.float().flatten().cpu().tolist())
    result = {
        "scope": "fixed small train/test iterator prefixes, no validation tuning, one seed",
        "teacher_checkpoint": str(checkpoint),
        "teacher_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "warm_start": "Both students inherit teacher front-end, shared one-layer block and Buy tower",
        "student": "Tokenizer -> one shared Mixer call conditioned on Buy -> single Buy tower",
        "kd": "0.5 BCE(real Buy) + 0.5 BCE(teacher Buy probability), temperature=1",
        "train_samples": train_samples, "test_samples": len(labels),
        "test_positive_buy": sum(labels), "joint_train_wall_seconds": elapsed,
        "timing_note": "GPUs shared with unrelated training; no clean latency/speedup claim",
        "models": {},
    }
    for name, model, values in zip(["teacher_branch"] + names, [teacher] + students, predictions):
        counts = sum(p.numel() for p in model.parameters())
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with FlopCounterMode(display=False) as counter:
                model(representative)
        batch_size = representative[2].shape[0]
        result["models"][name] = {
            "buy_auc": roc_auc_score(labels, values), "buy_logloss": log_loss(labels, values),
            "parameters": counts, "forward_counted_gflops_per_sample": counter.get_total_flops()/batch_size/1e9,
        }
    result["flops_note"] = "Registered PyTorch operators, MAC=2; excludes lookup and some elementwise operations"
    def teacher_buy_only(batch):
        # Same graph states, but no unused task towers or auxiliary predictions.
        context, items, mask = teacher.get_inputs(batch)
        batch_size = mask.shape[0]
        item_embeddings = teacher.embedding_layer(items, flatten_emb=True).view(
            batch_size, -1, teacher.item_info_dim)
        target = item_embeddings[:, -1, :]
        pooled = teacher.attention_layers(target, item_embeddings[:, :-1, :], mask)
        features = torch.cat([
            teacher.embedding_layer(context, flatten_emb=True), target, pooled], dim=-1)
        tokens = teacher.unified_tokenizer_layer(features)
        states, _, base, _ = teacher._run_progressive_encoder(tokens)
        return teacher.output_activation[index](teacher._task_logit(
            teacher._head_representation(states[index], index, base), index))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        direct = teacher_buy_only(representative)
        original = teacher(representative)["buy_pred"]
        torch.testing.assert_close(direct, original, rtol=0, atol=0)
        with FlopCounterMode(display=False) as counter:
            teacher_buy_only(representative)
    result["teacher_buy_only_pruned_gflops_per_sample"] = counter.get_total_flops()/batch_size/1e9
    result["teacher_buy_only_prediction_equivalence"] = True
    (output_dir / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
