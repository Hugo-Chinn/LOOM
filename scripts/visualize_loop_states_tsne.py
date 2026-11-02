#!/usr/bin/env python3
"""tSNE visualisation of RankMixer hidden states across loop stages.

For every dumped model this writes:

  * `<tag>_stages.png`      one panel per loop stage, points coloured by the
                            rarest task's label, annotated with a per-stage
                            linear-probe AUC for that task
  * `<tag>_evolution.png`   all stages in one shared tSNE space, coloured by
                            stage, with the positive/negative centroid
                            trajectory drawn on top
  * `<tag>_probe_auc.png`   per-task probe AUC as a function of loop depth

and a JSON with the underlying numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def probe_auc(features, targets, seed=0):
    """Linear-probe AUC with a single stratified split."""
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives < 20 or negatives < 20:
        return None
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.4, random_state=seed, stratify=targets
    )
    clf = LogisticRegression(max_iter=2000)
    scaler = StandardScaler().fit(x_train)
    clf.fit(scaler.transform(x_train), y_train)
    score = clf.decision_function(scaler.transform(x_test))
    return float(roc_auc_score(y_test, score))


def build_embedding(flat, seed, perplexity):
    n = flat.shape[0]
    perplexity = float(min(perplexity, max(5, n / 20)))
    pca = PCA(n_components=min(50, flat.shape[1]), random_state=seed)
    reduced = pca.fit_transform(flat)
    tsne = TSNE(
        n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
        random_state=seed, method="barnes_hut",
    )
    return tsne.fit_transform(reduced), float(pca.explained_variance_ratio_.sum())


def process_one(npz_path: Path, outdir: Path, seed: int, perplexity: int,
                max_samples: int, probe_task: str | None):
    data = np.load(npz_path, allow_pickle=True)
    reps = data["representations"]
    labels = np.asarray(data["labels"]).reshape(data["representations"].shape[0], -1)
    stage_names = [str(s) for s in data["stage_names"]]
    task_names = [str(s) for s in data["task_names"]]
    meta = json.loads(str(data["meta"]))
    rng = np.random.default_rng(seed)
    n_total, n_stages, dim = reps.shape
    # Probe AUCs use every dumped sample (rare tasks need the positives);
    # tSNE runs on a subsample so the embedding stays tractable.
    if n_total > max_samples:
        idx = rng.choice(n_total, max_samples, replace=False)
        plot_reps = reps[idx]
        plot_labels = labels[idx]
    else:
        plot_reps = reps
        plot_labels = labels
    n_samples = plot_reps.shape[0]

    positive_rate = labels.mean(axis=0)
    if probe_task is None:
        task_index = len(task_names) - 1  # deepest conversion-like task
    else:
        task_index = task_names.index(probe_task)
    task_name = task_names[task_index]
    target = plot_labels[:, task_index]
    full_target = labels[:, task_index]

    flat = plot_reps.reshape(-1, dim)
    coords, pca_var = build_embedding(flat, seed, perplexity)
    coords = coords.reshape(n_samples, n_stages, 2)

    tag = npz_path.stem
    metrics = {
        "npz": str(npz_path),
        "probe_samples": int(n_total),
        "tsne_samples": int(n_samples),
        "meta": meta,
        "samples": int(n_samples),
        "stages": stage_names,
        "tasks": task_names,
        "probe_task": task_name,
        "positive_rate": float(positive_rate[task_index]),
        "pca_variance_kept": pca_var,
        "per_stage": [],
        "per_task_probe_auc": {},
    }

    # Fig 1: per-stage panels coloured by the probe task label
    cmap = {0.0: "#b0b7c3", 1.0: "#d1495b"}
    n_cols = min(3, n_stages)
    n_rows = int(np.ceil(n_stages / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows),
                             squeeze=False)
    for stage in range(n_stages):
        ax = axes[stage // n_cols][stage % n_cols]
        xy = coords[:, stage, :]
        auc = probe_auc(reps[:, stage, :], full_target, seed=seed)
        pos_centroid = xy[target == 1].mean(axis=0) if (target == 1).any() else None
        neg_centroid = xy[target == 0].mean(axis=0) if (target == 0).any() else None
        sep = (float(np.linalg.norm(pos_centroid - neg_centroid))
               if pos_centroid is not None and neg_centroid is not None else None)
        drift = float(np.linalg.norm(
            plot_reps[:, stage, :].mean(axis=0) - plot_reps[:, 0, :].mean(axis=0)
        ))
        metrics["per_stage"].append({
            "stage": stage_names[stage],
            "probe_auc": auc,
            "centroid_separation_tsne": sep,
            "representation_drift": drift,
        })
        colors = [cmap[float(v)] for v in target]
        ax.scatter(xy[:, 0], xy[:, 1], s=4, c=colors, alpha=0.45, linewidths=0)
        title = stage_names[stage] if stage > 0 else "base (pre-loop)"
        sub = f"probe AUC={auc:.4f}" if auc is not None else "probe AUC=n/a"
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for stage in range(n_stages, n_rows * n_cols):
        axes[stage // n_cols][stage % n_cols].axis("off")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=cmap[1.0],
                   label=f"{task_name}=1 (n={int(target.sum())})"),
        plt.Line2D([], [], marker="o", linestyle="", color=cmap[0.0],
                   label=f"{task_name}=0"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"{meta['dataset_id']} · {meta['expid']}\n"
                 f"tSNE of pooled hidden states, coloured by {task_name}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(outdir / f"{tag}_stages.png", dpi=140)
    plt.close(fig)

    # Fig 2: shared space coloured by stage + centroid trajectory
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    stage_colors = plt.cm.viridis(np.linspace(0, 0.9, n_stages))
    for stage in range(n_stages):
        ax.scatter(coords[:, stage, 0], coords[:, stage, 1], s=3,
                   color=stage_colors[stage], alpha=0.18, linewidths=0,
                   label=stage_names[stage] if stage > 0 else "base")
    for cls, style, marker in ((1, "-", "o"), (0, "--", "s")):
        mask = target == cls
        if not mask.any():
            continue
        path = coords[mask].mean(axis=0)
        ax.plot(path[:, 0], path[:, 1], style, color="#2b2d42",
                linewidth=1.6, alpha=0.9)
        ax.scatter(path[:, 0], path[:, 1], s=45, marker=marker,
                   facecolor="#ffffff" if cls == 1 else "#2b2d42",
                   edgecolor="#2b2d42", zorder=5,
                   label=f"{task_name}={int(cls)} centroid path")
        for stage in range(n_stages):
            ax.annotate(stage_names[stage] if stage > 0 else "base",
                        (path[stage, 0], path[stage, 1]), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_title(f"{meta['dataset_id']} · {meta['expid']}\n"
                 f"hidden-state evolution across loop stages (shared tSNE space)",
                 fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_evolution.png", dpi=140)
    plt.close(fig)

    # Fig 3: per-task probe AUC vs stage
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for task_idx, task_label in enumerate(task_names):
        aucs = []
        for stage in range(n_stages):
            auc = probe_auc(reps[:, stage, :], labels[:, task_idx], seed=seed)  # full sample
            aucs.append(np.nan if auc is None else auc)
        metrics["per_task_probe_auc"][task_label] = [None if np.isnan(a) else a for a in aucs]
        ax.plot(range(n_stages), aucs, marker="o", linewidth=1.6, label=task_label)
    ax.set_xticks(range(n_stages))
    ax.set_xticklabels([s if i > 0 else "base" for i, s in enumerate(stage_names)],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("linear-probe AUC")
    ax.set_title(f"{meta['dataset_id']} · {meta['expid']}\n"
                 "probe AUC per task across loop depth", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_probe_auc.png", dpi=140)
    plt.close(fig)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=int, default=35)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--probe-task", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for path in args.inputs:
        metrics = process_one(Path(path), outdir, args.seed, args.perplexity,
                              args.max_samples, args.probe_task)
        all_metrics.append(metrics)
        print("done", path, flush=True)
    Path(args.out_json).write_text(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
