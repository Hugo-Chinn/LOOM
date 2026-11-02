#!/usr/bin/env python3
"""Fixed-recipe single-target compression on retained historical teachers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_buy_one_step_distillation import RankMixer
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import roc_auc_score, log_loss
from unirank.utils import load_config
from unirank.features import FeatureMap
from unirank.pytorch.dataloaders import RankDataLoader

SPECS = {
    "kuai": ("config_rankmixer_viz", "RankMixer_Kuai_DAGMeanPre2_S20262028", "is_forward"),
    "merrec": ("config_rankmixer_viz", "RankMixer_MerRec_DAGGated_S20262028", "Purchase"),
    "qk": ("config_rankmixer_viz", "RankMixer_QK_DAGGated_S20262028", "share"),
    "taac": ("config_rankmixer_bypass_rare", "RankMixer_TAAC_BypassRare_S20262028", "is_conversion"),
}
OUT = ROOT / "reports/one_step_distillation_extension_20261026"


def tokens(model, batch):
    context, items, mask = model.get_inputs(batch)
    embeddings = model.embedding_layer(items, flatten_emb=True).view(
        mask.shape[0], -1, model.item_info_dim)
    target = embeddings[:, -1]
    pooled = model.attention_layers(target, embeddings[:, :-1], mask)
    return model.unified_tokenizer_layer(torch.cat([
        model.embedding_layer(context, flatten_emb=True), target, pooled], dim=-1))


class SingleTarget(RankMixer):
    def forward(self, batch):
        base = tokens(self, batch)
        state, _ = self._loop_update(base, 0)
        return self.output_activation[0](self._task_logit(
            self._state_representation(state, 0), 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SPECS, required=True)
    parser.add_argument("--arm", choices=["kd90", "label"], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / f"{args.dataset}_{args.arm}.json"
    if result_path.exists():
        raise RuntimeError(f"Refuse to overwrite {result_path}")
    torch.set_num_threads(2)
    torch.manual_seed(20262028)
    cfg, expid, target = SPECS[args.dataset]
    params = load_config(str(ROOT / cfg), expid)
    checkpoint = Path(params["model_root"]) / params["dataset_id"] / (expid + ".model")
    params.update(gpu=0, enable_torch_compile=False, distributed=False,
                  enable_bf16=True, num_workers=2, verbose=0,
                  model_root=str(ROOT / "runs_local/one_step_distillation_extension_20261026"))
    data = Path(params["data_root"]) / params["dataset_id"]
    fmap = FeatureMap(params["dataset_id"], str(data))
    fmap.load(str(data / "feature_map.json"), params)
    index = fmap.labels.index(target)
    weights = {k.replace("._orig_mod.", "."): v for k, v in
               torch.load(checkpoint, map_location="cpu", weights_only=True).items()}
    teacher = RankMixer(fmap, **params)
    teacher.load_state_dict(weights, strict=True)
    teacher.eval().requires_grad_(False)
    student = SingleTarget(fmap, **params)
    student.load_state_dict(weights, strict=True)
    assert student.task_token_mode == "none"
    assert student.loop_block is not None
    assert student.iteration_depth_embedding is None
    assert student.depth_channel_scale is None and student.depth_modulation is None
    student.tower = nn.ModuleList([student.tower[index]])
    student.output_activation = nn.ModuleList([student.output_activation[index]])
    student.behavior_embedding = nn.Embedding.from_pretrained(
        student.behavior_embedding.weight[index:index+1].detach().clone(), freeze=False)
    if student.loop_gate_logits is not None:
        student.loop_gate_logits = nn.Parameter(
            student.loop_gate_logits[index:index+1].detach().clone())
    student.num_tasks = student.num_loops = 1
    student.num_preloops = 0
    student.loop_topology = "linear"
    student.supervision_mode = "progressive"
    for name in ("preloop_gate_logits", "dag_fusion_logits", "branch_fusion_logits",
                 "base_bypass_logits"):
        student.register_parameter(name, None)
    student.base_bypass_mask = None
    for module in (student.embedding_layer, student.unified_tokenizer_layer,
                   student.attention_layers):
        module.requires_grad_(False)
    student._compile_optimizers()
    alpha = 0.9 if args.arm == "kd90" else 0.0

    def teacher_target(batch):
        base = teacher._run_preloops(tokens(teacher, batch))
        if teacher.loop_topology == "dag":
            needed = set()
            def include(node):
                if node not in needed:
                    needed.add(node)
                    for parent in teacher.graph_parents[node]:
                        include(parent)
            include(index)
            states = [None] * teacher.num_tasks
            for node in teacher.graph_topological_order:
                if node not in needed:
                    continue
                parent = (teacher._fuse_dag_parents(node, states, base)
                          if teacher.graph_parents[node] else base)
                states[node], _ = teacher._loop_update(parent, node)
            state = states[index]
        else:
            assert teacher.loop_topology == "linear"
            state = base
            for node in range(index+1):
                state, _ = teacher._loop_update(state, node)
        return teacher.output_activation[index](teacher._task_logit(
            teacher._head_representation(state, index, base), index))

    def evaluate(iterator, limit):
        ys, ts, ss = [], [], []
        student.eval()
        with torch.inference_mode():
            for step, batch in enumerate(iterator):
                if step >= limit:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pt, ps = teacher_target(batch), student(batch)
                ys.extend(teacher.get_labels(batch)[index].flatten().cpu().tolist())
                ts.extend(pt.float().flatten().cpu().tolist())
                ss.extend(ps.float().flatten().cpu().tolist())
        return dict(n=len(ys), positives=sum(ys), teacher_auc=roc_auc_score(ys, ts),
                    student_auc=roc_auc_score(ys, ss), teacher_logloss=log_loss(ys, ts),
                    student_logloss=log_loss(ys, ss))

    train, valid = RankDataLoader(fmap, stage="train", **params).make_iterator()
    best_path = Path(params["model_root"]) / f"{args.dataset}_{args.arm}.model"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    validations, best_auc, samples = [], -1, 0
    for step, batch in enumerate(train):
        if step >= 512:
            break
        if step == 0:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                torch.testing.assert_close(teacher_target(batch), teacher(batch)[target+"_pred"],
                                           atol=0, rtol=0)
        student.train()
        y = teacher.get_labels(batch)[index]
        student._optimizer_zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ps = student(batch).float().clamp(1e-6, 1-1e-6)
            if alpha:
                with torch.no_grad():
                    pt = teacher_target(batch).float().clamp(1e-6, 1-1e-6)
        loss = (1-alpha)*nn.functional.binary_cross_entropy(ps, y)
        if alpha:
            loss += alpha*nn.functional.binary_cross_entropy(ps, pt)
        loss.backward()
        student._optimizer_step()
        samples += y.numel()
        if (step+1) % 64 == 0:
            print(f"{args.dataset}/{args.arm} step={step+1}", flush=True)
        if step+1 in (128, 512):
            metrics = evaluate(valid, 128)
            metrics["train_batches"] = step+1
            validations.append(metrics)
            if metrics["student_auc"] > best_auc:
                best_auc = metrics["student_auc"]
                torch.save(student.state_dict(), best_path)
            print("VALID", json.dumps(metrics), flush=True)
    student.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    test = RankDataLoader(fmap, stage="test", **params).make_iterator()
    metrics = evaluate(test, 256)
    representative = next(iter(test))
    counts = {}
    for name, forward in (("student", student), ("teacher_target_only", teacher_target),
                          ("teacher_all_tasks", teacher)):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with FlopCounterMode(display=False) as counter:
                forward(representative)
        counts[name] = counter.get_total_flops()/representative[2].shape[0]/1e9
    result = dict(dataset=params["dataset_id"], target=target, arm=args.arm,
                  seed=20262028, alpha=alpha, train_samples=samples,
                  validation=validations, test=metrics, forward_gflops_per_sample=counts,
                  student_parameters=sum(p.numel() for p in student.parameters()),
                  student_trainable_parameters=sum(p.numel() for p in student.parameters()
                                                   if p.requires_grad),
                  teacher_parameters=sum(p.numel() for p in teacher.parameters()),
                  teacher_checkpoint=str(checkpoint),
                  teacher_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  student_checkpoint=str(best_path),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  teacher_pruned_equivalence=True, student_mixer_calls=1,
                  note="Historical retained teacher; frozen front, no student preloops/bypass; "
                       "fixed validation selection, one seed, iterator prefixes; "
                       "operator-count GFLOPs MAC=2, incomplete elementwise coverage.")
    result_path.write_text(json.dumps(result, indent=2)+"\n")
    print("DONE", json.dumps(metrics), counts, flush=True)


if __name__ == "__main__":
    main()
