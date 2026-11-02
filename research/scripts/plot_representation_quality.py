#!/usr/bin/env python3
"""Publication-style representation plots: t-SNE panels + cross-method curves.

Reads the dumped npz files and the geometry metrics json. Produces, per model,
a t-SNE panel figure, and cross-method comparison curves (effective rank,
probe AUC, class separation) plus a ranked quality-vs-accuracy chart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


NEG = "#cfd6e0"
POS = "#c0392b"
BACKBONE_COLORS = {"rankmixer": "#2b6cb0", "onetrans": "#d97706", "mtlbases": "#2f855a"}


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
    })


def embed(reps: np.ndarray, seed: int, perplexity: int):
    n, s, d = reps.shape
    flat = reps.reshape(-1, d)
    pca = PCA(n_components=min(50, d), random_state=seed).fit_transform(flat)
    pp = min(perplexity, max(5, flat.shape[0] // 20))
    ts = TSNE(n_components=2, perplexity=pp, init="pca", learning_rate="auto",
              random_state=seed, method="barnes_hut")
    return ts.fit_transform(pca).reshape(n, s, 2)


def panels(tag, reps, y, names, labels, meta, metrics, outdir, seed, perplexity, max_samples):
    n = reps.shape[0]
    idx = np.random.default_rng(seed).choice(n, min(max_samples, n), replace=False)
    sub = reps[idx]
    ysub = y[idx]
    emb = embed(sub, seed, perplexity)
    n_slices = sub.shape[1]
    focus = len(labels) - 1
    cols = min(4, n_slices)
    rows = int(np.ceil(n_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.1 * rows), squeeze=False)
    for s in range(n_slices):
        ax = axes[s // cols][s % cols]
        xy = emb[:, s, :]
        neg = ysub[:, focus] == 0
        ax.scatter(xy[neg, 0], xy[neg, 1], s=3, c=NEG, alpha=0.35, linewidths=0, rasterized=True)
        ax.scatter(xy[~neg, 0], xy[~neg, 1], s=5, c=POS, alpha=0.75, linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        st = metrics["per_slice"][s]
        pa = st["probe_auc"].get(labels[focus])
        ax.set_title(f"{st['name']}\nprobe {'-' if pa is None else format(pa, '.3f')} · "
                     f"rank {st['effective_rank']:.0f}", fontsize=8.5, pad=6)
    for s in range(n_slices, rows * cols):
        axes[s // cols][s % cols].axis("off")
    handles = [Line2D([], [], marker="o", ls="", color=POS, label=f"{labels[focus]}=1"),
               Line2D([], [], marker="o", ls="", color=NEG, label=f"{labels[focus]}=0")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"{meta['dataset_id']} · {meta['expid']}\n"
                 f"{metrics['slice_kind']} · t-SNE (PCA-50 → BH, perp={perplexity})", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(outdir / f"{tag}_tsne.png", dpi=180)
    plt.close(fig)


def curves(entries, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for e in entries:
        bb = e["tag"].split("__")[0]
        color = BACKBONE_COLORS.get(bb, "#666666")
        ps = e["per_slice"]
        xs = list(range(len(ps)))
        axes[0].plot(xs, [p["effective_rank"] for p in ps], color=color, alpha=0.75, lw=1.3,
                     marker="o", ms=3, label=None)
        vals = []
        for p in ps:
            a = [v for v in p["probe_auc"].values() if v is not None]
            vals.append(np.nan if not a else float(np.mean(a)))
        axes[1].plot(xs, vals, color=color, alpha=0.75, lw=1.3, marker="o", ms=3)
        sep = []
        for p in ps:
            s = [v["separation_ratio"] for v in p["centroid_gap"].values()
                 if v.get("separation_ratio") is not None]
            sep.append(np.nan if not s else float(np.mean(s)))
        axes[2].plot(xs, sep, color=color, alpha=0.75, lw=1.3, marker="o", ms=3)
    titles = ["effective rank", "mean probe AUC", "mean class separation ratio"]
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xlabel("slice index")
    handles = [Line2D([], [], color=c, lw=2, label=n) for n, c in BACKBONE_COLORS.items()]
    axes[2].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="backbone")
    fig.suptitle("Representation quality across methods (Taobao)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "cross_method_quality_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def ranking(entries, macro, outdir):
    rows = []
    for e in entries:
        ps = e["per_slice"]
        er = np.mean([p["effective_rank"] for p in ps])
        seps = [v["separation_ratio"] for p in ps for v in p["centroid_gap"].values()
                if v.get("separation_ratio") is not None]
        m = macro.get(e["meta"]["expid"].split("_S")[0])
        if m is None or not seps:
            continue
        rows.append((e["tag"], float(er), float(np.mean(seps)), float(m)))
    rows.sort(key=lambda r: -r[2])
    if not rows:
        return
    tags = [r[0] for r in rows]
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.bar(x - 0.2, [r[1] for r in rows], width=0.4, color="#7f9fc4", label="effective rank")
    ax.set_ylabel("effective rank")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [r[2] for r in rows], width=0.4, color="#c0392b", alpha=0.75, label="separation ratio")
    ax2.set_ylabel("class separation ratio")
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.plot(x, [r[3] for r in rows], color="#2f855a", marker="o", ms=4, lw=1.5, label="macro AUC")
    ax3.set_ylabel("4-seed macro AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_Taobao_Action_S20262027", "") for t in tags], rotation=60, ha="right", fontsize=7)
    ax.set_title("Representation quality vs task accuracy (Taobao, sorted by separation ratio)", fontsize=12)
    hs = [Line2D([], [], color="#7f9fc4", lw=6, label="effective rank"),
          Line2D([], [], color="#c0392b", lw=6, label="separation ratio"),
          Line2D([], [], color="#2f855a", marker="o", ms=4, lw=1.5, label="macro AUC")]
    ax.legend(handles=hs, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "cross_method_quality_ranking.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--geometry-json", required=True)
    ap.add_argument("--macro-json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=2500)
    ap.add_argument("--perplexity", type=int, default=35)
    args = ap.parse_args()
    style()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    geo = json.loads(Path(args.geometry_json).read_text())
    macro = json.loads(Path(args.macro_json).read_text())
    by_tag = {e["tag"]: e for e in geo}
    for path in sorted(Path(args.npz_dir).glob("*__*.npz")):
        tag = path.stem
        e = by_tag.get(tag)
        if e is None:
            continue
        d = np.load(path, allow_pickle=True)
        if "stage_reps" in d:
            reps = d["stage_reps"]; names = [str(v) for v in d["stage_names"]]
        elif "head_inputs" in d:
            reps = d["head_inputs"]; names = [str(v) for v in d["labels"]]
        else:
            continue
        reps = np.squeeze(reps)
        if reps.ndim == 2:
            reps = reps[:, None, :]
        y = np.squeeze(d["y"])
        if y.ndim == 1:
            y = y[:, None]
        labels = [str(v) for v in d["labels"]]
        if y.shape[1] != len(labels):
            k = min(y.shape[1], len(labels)); labels = labels[:k]; y = y[:, :k]
        panels(tag, reps, y, names, labels, e["meta"], e, outdir, args.seed, args.perplexity, args.max_samples)
        print("plotted", tag, flush=True)
    curves(geo, outdir)
    ranking(geo, macro, outdir)
    print("curves + ranking written")


if __name__ == "__main__":
    main()
