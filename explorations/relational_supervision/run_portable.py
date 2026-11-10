"""Bounded RelSup study on NPZ data with the portable core.

    python -m explorations.relational_supervision.run_portable \
        --config configs/portable/default.yaml --data runs_local/demo-data \
        --output runs_local/relsup-portable --device cuda:0

Output layout matches `loom.report`: <output>/<arm>/S<seed>/metrics.json.
This runner is a pipeline smoke check. It writes test CSVs immediately, so it
is outside the PLAN.md selection protocol: never select an arm from its output.
Synthetic demo data is not method evidence.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from loom.cli import evaluate, load_split
from loom.config import LOOMConfig
from loom.report import summarize

from . import arms as A
from .environment import write as write_environment
from .losses import RelSupConfig
from .portable_model import RelationalTabular



def cross_state_auc(model, x, y, device, batch_size=1024):
    """AUC of head t read from state u, for every (t, u). Diagonal = main task.

    Row t, column u far below the diagonal means state u lacks what head t uses.
    """
    core = model.core
    model.eval()
    chunks = []
    with torch.inference_mode():
        for batch in DataLoader(torch.from_numpy(x), batch_size=batch_size):
            out = model(batch.to(device), return_states=True)
            chunks.append(torch.stack([
                torch.stack([core._readout(state, task, out["base"])
                             for state in out["states"]], 1)
                for task in range(len(core.tasks))], 1).float().cpu().numpy())
    logits = np.concatenate(chunks)  # [N, task, state]
    matrix = {}
    for t, task in enumerate(core.tasks):
        mask = np.isfinite(y[:, t])
        truth = y[mask, t]
        defined = len(np.unique(truth)) == 2
        matrix[task] = {
            state: float(roc_auc_score(truth, logits[mask, t, u])) if defined else None
            for u, state in enumerate(core.tasks)}
    return matrix


def train_one(config, relsup, settings, splits, seed, device, output, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    (x, y), (vx, vy), (tx, ty) = splits
    model = RelationalTabular(x.shape[1], config, relsup).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=0.0)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=settings["batch_size"], shuffle=True)
    output.mkdir(parents=True)
    best, history = -float("inf"), []
    for epoch in range(settings["epochs"]):
        model.train()
        losses, terms = [], {}
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.core.loss(model(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            for name, value in model.core.last_relational.items():
                terms.setdefault(name, []).append(value)
        metrics, _ = evaluate(model, vx, vy, device)
        aucs = [m["auc"] for m in metrics.values() if m["auc"] is not None]
        if not aucs:
            raise ValueError("validation needs at least one task with both label classes")
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "train_relational": {k: float(np.mean(v)) for k, v in terms.items()},
                        "validation": metrics})
        print(json.dumps({"arm": output.parent.name, "seed": seed, **history[-1]}),
              flush=True)
        if float(np.mean(aucs)) > best:
            best = float(np.mean(aucs))
            torch.save({"model": model.state_dict(), "config": config.to_dict(),
                        "relsup": relsup.to_dict(), "input_dim": x.shape[1],
                        "seed": seed, "epoch": epoch + 1}, output / "best.pt")
    saved = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    test, _ = evaluate(model, tx, ty, device)
    result = {
        "seed": seed, "selected_epoch": saved["epoch"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "test": test, "history": history,
        "cross_state_auc": {"validation": cross_state_auc(model, vx, vy, device),
                            "test": cross_state_auc(model, tx, ty, device)},
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (output / "resolved_config.json").write_text(json.dumps(
        {"model": config.to_dict(), "relsup": relsup.to_dict(), "training": settings},
        indent=2))
    return history[saved["epoch"] - 1]["validation"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(A.phase1()),
                        choices=list(A.phase1()))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(A.DEV_SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-runs", type=int, default=20)
    args = parser.parse_args()
    arms = {name: spec["overrides"] for name, spec in A.phase1().items()
            if name in args.arms}
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    count = len(arms) * len(args.seeds)
    if count > args.max_runs:
        parser.error(f"{count} runs exceeds budget {args.max_runs}")
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    document = yaml.safe_load(args.config.read_text())
    config = LOOMConfig(**document["model"])
    settings = {"epochs": 3, "batch_size": 128, "learning_rate": 0.001}
    settings.update(document.get("training", {}))
    x, y = load_split(args.data / "train.npz", config.tasks)
    vx, vy = load_split(args.data / "valid.npz", config.tasks, x.shape[1])
    tx, ty = load_split(args.data / "test.npz", config.tasks, x.shape[1])
    mean, scale = x.mean(0), x.std(0).clip(1e-6)  # training split only
    splits = [((s - mean) / scale, labels) for s, labels in ((x, y), (vx, vy), (tx, ty))]
    device = torch.device(args.device)
    args.output.mkdir(parents=True)
    write_environment(args.output / "environment.json",
                      {"seeds": args.seeds, "arms": arms, "training": settings})
    scores = {}
    for arm, overrides in arms.items():
        relsup = RelSupConfig(**overrides)
        scores[arm] = []
        for seed in args.seeds:
            validation = train_one(config, relsup, settings, splits, seed, device,
                                   args.output / arm / f"S{seed}", args.threads)
            aucs = [m["auc"] for m in validation.values() if m["auc"] is not None]
            scores[arm].append(sum(aucs) / len(aucs))
    means = {arm: sum(v) / len(v) for arm, v in scores.items()}
    (args.output / "validation_selection.json").write_text(json.dumps({
        "validation_mean_auc": means, "arms": arms, "seeds": args.seeds,
        "warning": "Selection uses validation only. Read per-task test metrics per run.",
    }, indent=2))
    summarize(args.output)
    print(f"Wrote per_seed_task.csv and per_task_summary.csv to {args.output}")


if __name__ == "__main__":
    main()
