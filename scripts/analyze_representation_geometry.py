#!/usr/bin/env python3
"""Representation geometry: effective rank, probe AUC, class separation, t-SNE.

Works for any dumped model: slices are loop stages when available, otherwise
per-task head inputs. Produces per-model panels/evolution/metric figures plus a
cross-method comparison against the logged 4-seed macro AUC.
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


def effective_rank(x: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def probe_auc(features: np.ndarray, target: np.ndarray, seed: int = 0):
    pos = int(target.sum())
    neg = len(target) - pos
    if pos < 20 or neg < 20:
        return None
    xtr, xte, ytr, yte = train_test_split(features, target, test_size=0.4,
                                          random_state=seed, stratify=target)
    scaler = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(xtr), ytr)
    return float(roc_auc_score(yte, clf.decision_function(scaler.transform(xte))))


def slice_stats(reps: np.ndarray, y: np.ndarray, labels, seed: int):
    n, dim = reps.shape
    out = {
        "effective_rank": effective_rank(reps),
        "effective_rank_ratio": effective_rank(reps) / dim,
        "mean_norm": float(np.linalg.norm(reps, axis=1).mean()),
        "feature_std_mean": float(reps.std(axis=0).mean()),
        "probe_auc": {},
        "centroid_gap": {},
    }
    for i, label in enumerate(labels):
        auc = probe_auc(reps, y[:, i], seed=seed)
        out["probe_auc"][label] = auc
        pos = reps[y[:, i] == 1]
        neg = reps[y[:, i] == 0]
        if len(pos) > 5 and len(neg) > 5:
            gap = np.linalg.norm(pos.mean(0) - neg.mean(0))
            spread = np.sqrt(((pos - pos.mean(0)) ** 2).sum(1).mean() + ((neg - neg.mean(0)) ** 2).sum(1).mean())
            out["centroid_gap"][label] = {"gap": float(gap), "separation_ratio": float(gap / spread) if spread > 0 else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--macro-json", default=None, help="optional {arm: macro} for cross-method plots")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=3000)
    ap.add_argument("--perplexity", type=int, default=35)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    macro = json.loads(Path(args.macro_json).read_text()) if args.macro_json else {}
    summary = []

    for path in args.inputs:
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        labels = [str(v) for v in data["labels"]]
        y = data["y"]
        reps = None
        names = []
        kind = "none"
        if "stage_reps" in data:
            reps = data["stage_reps"]
            names = [str(v) for v in data["stage_names"]]
            kind = "stage"
        elif "head_inputs" in data:
            reps = data["head_inputs"]
            names = labels
            kind = "head_input"
        if reps is None:
            print("skip (no representations):", path)
            continue
        reps = np.squeeze(reps)
        if reps.ndim == 2:
            reps = reps[:, None, :]
        y = np.squeeze(y)
        if y.ndim == 1:
            y = y[:, None]
        n, n_slices, dim = reps.shape
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[1] != len(labels):
            k = min(y.shape[1], len(labels))
            labels = labels[:k]
            y = y[:, :k]
        if len(names) < n_slices:
            names = names + [f"slice_{i}" for i in range(len(names), n_slices)]
        idx = np.random.default_rng(args.seed).choice(n, min(args.max_samples, n), replace=False)
        sub = reps[idx]
        ysub = y[idx]
        tag = Path(path).stem
        per_slice = []
        for s in range(n_slices):
            st = slice_stats(sub[:, s, :], ysub, labels, args.seed)
            st["name"] = names[s] if s < len(names) else f"slice_{s}"
            per_slice.append(st)

        # joint t-SNE across slices
        flat = sub.reshape(-1, dim)
        pca = PCA(n_components=min(50, dim), random_state=args.seed).fit_transform(flat)
        tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                    learning_rate="auto", random_state=args.seed, method="barnes_hut")
        emb = tsne.fit_transform(pca).reshape(sub.shape[0], n_slices, 2)

        focus = labels[-1]
        focus_idx = len(labels) - 1
        cols = min(3, n_slices)
        rows = int(np.ceil(n_slices / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
        for s in range(n_slices):
            ax = axes[s // cols][s % cols]
            xy = emb[:, s, :]
            colors = ["#d1495b" if v == 1 else "#b0b7c3" for v in ysub[:, focus_idx]]
            ax.scatter(xy[:, 0], xy[:, 1], s=4, c=colors, alpha=0.45, linewidths=0)
            st = per_slice[s]
            pa = st["probe_auc"].get(focus)
            pa_txt = "n/a" if pa is None else f"{pa:.4f}"
            ax.set_title(f"{st['name']}\nprobe({focus})={pa_txt} "
                         f"effrank={st['effective_rank']:.1f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        for s in range(n_slices, rows * cols):
            axes[s // cols][s % cols].axis("off")
        fig.suptitle(f"{meta['dataset_id']} · {meta['expid']}\n"
                     f"{'loop stages' if kind == 'stage' else 'head inputs'} · coloured by {focus}", fontsize=11)
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        fig.savefig(outdir / f"{tag}_panels.png", dpi=140)
        plt.close(fig)

        # metric curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        xs = list(range(n_slices))
        axes[0].plot(xs, [st["effective_rank"] for st in per_slice], marker="o")
        axes[0].set_title("effective rank per slice"); axes[0].set_xlabel("slice")
        for i, label in enumerate(labels):
            vals = [st["probe_auc"].get(label) for st in per_slice]
            axes[1].plot(xs, [np.nan if v is None else v for v in vals], marker="o", label=label)
        axes[1].set_title("probe AUC per task"); axes[1].legend(fontsize=8)
        for i, label in enumerate(labels):
            vals = [st["centroid_gap"].get(label, {}).get("separation_ratio") for st in per_slice]
            axes[2].plot(xs, [np.nan if v is None else v for v in vals], marker="o", label=label)
        axes[2].set_title("class separation ratio"); axes[2].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.25)
            if n_slices <= 12:
                ax.set_xticks(xs)
                ax.set_xticklabels([st["name"] for st in per_slice], rotation=25, ha="right", fontsize=7)
        fig.suptitle(f"{meta['dataset_id']} · {meta['expid']}", fontsize=11)
        fig.tight_layout()
        fig.savefig(outdir / f"{tag}_metrics.png", dpi=140)
        plt.close(fig)

        summary.append({
            "npz": str(path), "tag": tag, "meta": meta, "slice_kind": kind,
            "n_samples": int(sub.shape[0]), "dim": int(dim), "per_slice": per_slice,
            "macro_auc": macro.get(meta["expid"].split("_S")[0], None),
        })
        print("analyzed", tag, flush=True)

    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    if macro:
        xs, ys, tags = [], [], []
        for s in summary:
            key = s["meta"]["expid"].split("_S")[0]
            if key in macro:
                best = max([v for v in s["per_slice"][0]["probe_auc"].values() if v is not None], default=None)
                er = s["per_slice"][0]["effective_rank"]
                if best is not None:
                    xs.append(best); ys.append(macro[key]); tags.append(key)
        if xs:
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            ax.scatter(xs, ys, s=35)
            for x, y_, t in zip(xs, ys, tags):
                ax.annotate(t[:26], (x, y_), fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("probe AUC (first slice)")
            ax.set_ylabel("4-seed macro AUC")
            ax.set_title("representation quality vs task accuracy (Taobao)")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(outdir / "cross_method_probe_vs_macro.png", dpi=140)
            plt.close(fig)
    print("summary entries:", len(summary))


if __name__ == "__main__":
    main()
