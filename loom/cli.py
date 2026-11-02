"""CPU/GPU reference workflow for numeric NPZ datasets."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .config import LOOMConfig
from .core import LOOM


class TabularLOOM(nn.Module):
    """Example encoder, not the historical RankMixer feature/sequence tokenizer."""

    def __init__(self, input_dim, config):
        super().__init__()
        self.encoder = nn.Linear(input_dim, config.num_tokens * config.token_dim)
        self.core = LOOM(config)

    def forward(self, x):
        c = self.core.config
        return self.core(self.encoder(x).reshape(-1, c.num_tokens, c.token_dim))


def load_split(path, tasks, input_dim=None):
    with np.load(path, allow_pickle=False) as data:
        if not {"x", "y", "tasks"}.issubset(data.files):
            raise ValueError(f"{path}: required arrays are x, y, tasks")
        x, y = data["x"].astype("float32"), data["y"].astype("float32")
        if data["tasks"].tolist() != tasks:
            raise ValueError(f"{path}: task order differs from model config")
    if (x.ndim != 2 or y.shape != (len(x), len(tasks)) or len(x) == 0
            or x.shape[1] == 0 or not np.isfinite(x).all()):
        raise ValueError(f"{path}: invalid feature/label shapes or nonfinite features")
    if input_dim is not None and x.shape[1] != input_dim:
        raise ValueError(f"{path}: input_dim differs between splits")
    observed = np.isfinite(y)
    if np.isinf(y).any() or ((y[observed] != 0) & (y[observed] != 1)).any():
        raise ValueError(f"{path}: labels must be 0, 1, or NaN")
    if not observed.any(1).all():
        raise ValueError(f"{path}: each sample needs at least one observed label")
    return x, y


def evaluate(model, x, y, device, batch_size=256):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in DataLoader(torch.from_numpy(x), batch_size=batch_size):
            predictions.append(model(batch.to(device))["logits"].sigmoid().cpu().numpy())
    predictions = np.concatenate(predictions)
    metrics = {}
    for i, task in enumerate(model.core.tasks):
        mask = np.isfinite(y[:, i])
        truth, pred = y[mask, i], predictions[mask, i]
        metrics[task] = {
            "auc": float(roc_auc_score(truth, pred)) if len(np.unique(truth)) == 2 else None,
            "logloss": float(log_loss(truth, pred, labels=[0, 1])) if len(truth) else None,
            "observed": int(mask.sum()),
        }
    return metrics, predictions


def demo(directory):
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{split}.npz" for split in ("train", "valid", "test")]
    if any(path.exists() for path in paths):
        raise ValueError("demo destination already contains a split; will not overwrite")
    rng = np.random.default_rng(17)
    weights = rng.normal(size=(12, 4))
    for path, count in zip(paths, (1024, 256, 256)):
        x = rng.normal(size=(count, 12)).astype("float32")
        logits = x @ weights / 3
        y = (rng.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype("float32")
        np.savez_compressed(path, x=x, y=y, tasks=np.array(["click", "cart", "fav", "buy"]))
    print(f"Synthetic pipeline-check data written to {directory}; not research evidence")


def train(args):
    document = yaml.safe_load(args.config.read_text())
    if not isinstance(document, dict) or set(document) - {"model", "training"}:
        raise ValueError("config sections must be model and training")
    config = LOOMConfig(**document["model"])
    settings = {"epochs": 3, "batch_size": 128, "learning_rate": 0.001}
    supplied = document.get("training", {})
    if set(supplied) - set(settings):
        raise ValueError("unknown training configuration")
    settings.update(supplied)
    if (not isinstance(settings["epochs"], int) or settings["epochs"] < 1
            or not isinstance(settings["batch_size"], int) or settings["batch_size"] < 1
            or not np.isfinite(settings["learning_rate"]) or settings["learning_rate"] <= 0):
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output must be empty; use a separate directory for each seed")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    x, y = load_split(args.data / "train.npz", config.tasks)
    vx, vy = load_split(args.data / "valid.npz", config.tasks, x.shape[1])
    tx, ty = load_split(args.data / "test.npz", config.tasks, x.shape[1])
    # Fit normalization exclusively on the training split.
    mean, scale = x.mean(0), x.std(0).clip(1e-6)
    x, vx, tx = [(split - mean) / scale for split in (x, vx, tx)]
    model = TabularLOOM(x.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=0.0)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=settings["batch_size"], shuffle=True)
    args.output.mkdir(parents=True, exist_ok=True)
    best, history = -float("inf"), []
    for epoch in range(settings["epochs"]):
        model.train()
        losses = []
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.core.loss(model(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        metrics, _ = evaluate(model, vx, vy, device)
        valid_aucs = [m["auc"] for m in metrics.values() if m["auc"] is not None]
        if not valid_aucs:
            raise ValueError("validation needs at least one task with both label classes")
        score = float(np.mean(valid_aucs))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation": metrics})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "config": config.to_dict(),
                        "input_dim": x.shape[1], "mean": torch.tensor(mean),
                        "scale": torch.tensor(scale), "seed": args.seed,
                        "training": settings, "epoch": epoch + 1},
                       args.output / "best.pt")
    saved = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    metrics, _ = evaluate(model, tx, ty, device)
    result = {"seed": args.seed, "selected_epoch": saved["epoch"],
              "parameters": sum(p.numel() for p in model.parameters()),
              "test": metrics, "history": history}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.output / "resolved_config.json").write_text(
        json.dumps({"model": config.to_dict(), "training": settings}, indent=2))
    print(json.dumps({"seed": args.seed, "test": metrics}), flush=True)


def predict(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = LOOMConfig(**saved["config"])
    model = TabularLOOM(saved["input_dim"], config)
    model.load_state_dict(saved["model"])
    model.to(args.device)
    with np.load(args.input, allow_pickle=False) as data:
        x = data["x"].astype("float32")
    if x.ndim != 2 or x.shape[1] != saved["input_dim"] or not len(x) or not np.isfinite(x).all():
        raise ValueError("inference x has invalid shape or nonfinite values")
    if args.output.exists():
        raise ValueError("prediction output already exists")
    x = (x - saved["mean"].numpy()) / saved["scale"].numpy()
    empty_labels = np.full((len(x), len(config.tasks)), np.nan, dtype="float32")
    _, probs = evaluate(model, x, empty_labels, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, probabilities=probs, tasks=np.array(config.tasks))
    print(f"Predictions written to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("demo-data")
    generate.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=20262027)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--threads", type=int, default=2)
    inference = commands.add_parser("predict")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "demo-data":
        demo(args.output)
    elif args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
