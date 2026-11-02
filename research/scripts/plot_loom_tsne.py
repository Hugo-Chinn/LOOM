#!/usr/bin/env python3
"""Create the paper t-SNE comparison for LOOM, MMoE, and STEM.

The script loads real Taobao checkpoints, freezes one shared ordered subset of
the test set, extracts the four task-head input representations, and performs
one joint PCA+t-SNE per method.  No synthetic points or geometric post-hoc
manipulation are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_zoo  # noqa: E402
from unirank.features import FeatureMap  # noqa: E402
from unirank.preprocess import FeatureProcessor, build_dataset  # noqa: E402
from unirank.pytorch.dataloaders import RankDataLoader  # noqa: E402
from unirank.utils import load_config  # noqa: E402


METHODS = (
    {
        "key": "loom",
        "title": "LOOM",
        "expid": "LoopMT_Taobao_Action_S20262027",
        "hidden_state": "progressive task representation read by each task tower",
    },
    {
        "key": "mmoe",
        "title": "MMoE",
        "expid": "MMoE_Taobao_Action_S20262027",
        "hidden_state": "task-gated mixture-of-experts output read by each task tower",
    },
    {
        "key": "stem",
        "title": "STEM",
        "expid": "STEM_Taobao_Action_S20262027",
        "hidden_state": "AFTB task-routed expert output read by each task tower",
    },
)

DISPLAY_NAMES = ("Click", "Cart", "Favor", "Buy")
COLORS = ("#3477C6", "#D98B35", "#289672", "#BF607F")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config_viz_taobao_mtl_baselines"),
        help="Config directory containing the three aligned experiments.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ROOT / "runs_local/checkpoints_config_viz_taobao_mtl_baselines"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "reports"))
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--density-k", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--cache-npz", default=None)
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _tree_slice(value, stop):
    if torch.is_tensor(value):
        return value[:stop].detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _tree_slice(item, stop) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_slice(item, stop) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_slice(item, stop) for item in value)
    raise TypeError(f"Unsupported batch node: {type(value)!r}")


def _tree_concat(values):
    first = values[0]
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        return {key: _tree_concat([value[key] for value in values]) for key in first}
    if isinstance(first, list):
        return [_tree_concat([value[index] for value in values]) for index in range(len(first))]
    if isinstance(first, tuple):
        return tuple(_tree_concat([value[index] for value in values]) for index in range(len(first)))
    raise TypeError(f"Unsupported batch node: {type(first)!r}")


def _tree_hash(value, digest, path="root"):
    digest.update(path.encode("utf-8"))
    if torch.is_tensor(value):
        tensor = value.contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            _tree_hash(value[key], digest, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _tree_hash(item, digest, f"{path}[{index}]")
    else:
        raise TypeError(f"Unsupported batch node: {type(value)!r}")


def _tree_batch_slice(value, start, stop):
    if torch.is_tensor(value):
        return value[start:stop]
    if isinstance(value, dict):
        return {key: _tree_batch_slice(item, start, stop) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_batch_slice(item, start, stop) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_batch_slice(item, start, stop) for item in value)
    raise TypeError(f"Unsupported batch node: {type(value)!r}")


def freeze_test_samples(feature_map, params, samples: int):
    loader_params = dict(params)
    loader_params.update(
        batch_size=min(int(loader_params.get("batch_size", samples)), samples),
        shuffle=False,
        num_workers=0,
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
    )
    test_loader = RankDataLoader(feature_map, stage="test", **loader_params).make_iterator()
    pieces = []
    count = 0
    for batch in test_loader:
        batch_rows = int(batch[2].shape[0])
        take = min(samples - count, batch_rows)
        pieces.append(_tree_slice(batch, take))
        count += take
        if count >= samples:
            break
    if count < samples:
        raise RuntimeError(f"Requested {samples} test samples, but only found {count}")
    frozen = _tree_concat(pieces)
    digest = hashlib.sha256()
    _tree_hash(frozen, digest)
    return frozen, digest.hexdigest()


def iter_frozen_batches(frozen, batch_size):
    total = int(frozen[2].shape[0])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        yield _tree_batch_slice(frozen, start, stop)


def extract_task_representations(model, batch):
    """Return B x 4 x D representations immediately before task towers."""
    if isinstance(model, model_zoo.LoopMT):
        tokens = model._token_representations(batch)
        states = model._loop_states(tokens)
        representations = []
        stages = []
        for task_idx in range(model.num_tasks):
            stage = task_idx if model.readout == "progressive" else model.num_loops - 1
            stage = min(stage, len(states) - 1)
            representations.append(model._task_representation(states[stage], task_idx))
            stages.append(stage)
        return torch.stack(representations, dim=1), stages
    shared = model._shared_representation(batch)
    representations = model._task_representations(shared)
    return torch.stack(representations, dim=1), None


def load_and_extract(method, feature_map, common_params, frozen, args):
    params = load_config(args.config, method["expid"])
    params.update(
        gpu=args.gpu,
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        num_workers=0,
        enable_bf16=False,
    )
    model_class = getattr(model_zoo, params["model"])
    model = model_class(feature_map, **params)
    model.model_to_device()
    checkpoint = Path(args.checkpoint_dir) / params["dataset_id"] / f"{method['expid']}.model"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model.load_weights(str(checkpoint))
    model.eval()

    arrays = []
    readout_stages = None
    with torch.inference_mode():
        for batch in iter_frozen_batches(frozen, args.batch_size):
            representations, stages = extract_task_representations(model, batch)
            arrays.append(representations.float().cpu().numpy())
            if stages is not None:
                readout_stages = stages
    result = np.concatenate(arrays, axis=0)
    if result.shape[1] != 4:
        raise ValueError(f"{method['title']} produced {result.shape[1]} tasks, expected four")
    info = {
        "setting": method["title"],
        "model_class": params["model"],
        "experiment_id": method["expid"],
        "dataset_id": params["dataset_id"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "hidden_state": method["hidden_state"],
        "representation_shape": list(result.shape),
        "representation_dtype": str(result.dtype),
        "task_labels": list(feature_map.labels),
        "readout_stage_zero_based": readout_stages,
        "config": {
            key: params.get(key)
            for key in (
                "seed", "bottom_type", "embedding_dim", "token_dim", "num_layers",
                "num_ns_token", "num_loops", "readout", "deep_supervision",
                "refinement_layers", "num_experts", "num_shared_experts",
                "num_task_experts", "task_embedding", "private_rank",
            )
            if key in params
        },
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, info


def reduce_jointly(representations, args):
    samples, tasks, dim = representations.shape
    flat = representations.reshape(samples * tasks, dim)
    n_components = min(args.pca_components, flat.shape[1], flat.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=args.random_state)
    pca_values = pca.fit_transform(flat)
    tsne_kwargs = dict(
        n_components=2,
        perplexity=args.perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
        method="barnes_hut",
        angle=0.5,
    )
    # scikit-learn renamed n_iter to max_iter in 1.5.
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
        tsne_kwargs["max_iter"] = args.max_iter
    else:
        tsne_kwargs["n_iter"] = args.max_iter
    coordinates = TSNE(**tsne_kwargs).fit_transform(pca_values)
    return coordinates.reshape(samples, tasks, 2), {
        "input_dim": int(dim),
        "pca_components": int(n_components),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "tsne": {
            "perplexity": args.perplexity,
            "init": "pca",
            "learning_rate": "auto",
            "max_iter": args.max_iter,
            "random_state": args.random_state,
            "method": "barnes_hut",
            "angle": 0.5,
        },
        "joint_across_four_tasks": True,
    }


def legacy_effective_rank_stats(x, projection_dim=64, seed=20262027):
    """Match scripts/head_hidden_analysis.py for historical comparability."""
    values = torch.from_numpy(np.asarray(x)).float()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if values.size(1) > projection_dim:
        projection = torch.randn(
            values.size(1), projection_dim, generator=generator,
            dtype=torch.float32,
        ) / np.sqrt(projection_dim)
        values = values @ projection
    values = values - values.mean(dim=0, keepdim=True)
    covariance = values.T @ values / max(1, values.size(0) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    energy = eigenvalues.sum().clamp_min(1e-12)
    probabilities = eigenvalues / energy
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    return {
        "effective_rank": float(effective_rank),
        "effective_rank_ratio": float(effective_rank / values.size(1)),
        "top1_variance_fraction": float(eigenvalues[-1] / energy),
        "projection_dimension": int(values.size(1)),
        "projection_seed": int(seed),
        "samples": int(values.size(0)),
    }


def density_style(points, base_color, k):
    effective_k = min(k + 1, len(points))
    neighbors = NearestNeighbors(n_neighbors=effective_k).fit(points)
    distances, _ = neighbors.kneighbors(points)
    radius = np.maximum(distances[:, -1], np.finfo(np.float32).eps)
    density = 1.0 / np.square(radius)
    low, high = np.quantile(density, [0.05, 0.95])
    if high <= low:
        normalized = np.full_like(density, 0.5)
    else:
        normalized = np.clip((density - low) / (high - low), 0.0, 1.0)
    rgb = np.asarray(mcolors.to_rgb(base_color))
    strength = 0.34 + 0.66 * normalized
    facecolors = 1.0 - strength[:, None] * (1.0 - rgb[None, :])
    alpha = 0.38 + 0.57 * normalized
    rgba = np.column_stack([facecolors, alpha])
    order = np.argsort(normalized)
    return rgba, order, {
        "k": int(effective_k - 1),
        "density_definition": "inverse squared distance to kth nearest neighbor in 2D t-SNE space",
        "normalization_quantiles": [0.05, 0.95],
        "density_q05": float(low),
        "density_q95": float(high),
    }


def centered_square_limits(points, padding=0.075):
    """Fit all real points in a centered, isotropic square viewport.

    This changes only the axes shown for a panel. It does not transform,
    translate, shrink, or discard any t-SNE point.
    """
    x_low, y_low = np.min(points, axis=0)
    x_high, y_high = np.max(points, axis=0)
    x_center = float((x_low + x_high) / 2.0)
    y_center = float((y_low + y_high) / 2.0)
    span = max(float(x_high - x_low), float(y_high - y_low), 1e-6)
    half = 0.5 * span * (1.0 + 2.0 * padding)
    return (x_center - half, x_center + half), (y_center - half, y_center + half)


def draw_figure(all_coordinates, method_infos, args):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.45,
    })
    fig = plt.figure(figsize=(7.16, 3.12), facecolor="white")
    outer = fig.add_gridspec(
        1, 3, left=0.026, right=0.994, bottom=0.055, top=0.865, wspace=0.105
    )
    density_metadata = {}
    group_axes = []
    for method_idx, (method, coordinates) in enumerate(zip(METHODS, all_coordinates)):
        inner = outer[method_idx].subgridspec(2, 2, wspace=0.055, hspace=0.16)
        axes = []
        density_metadata[method["key"]] = {}
        for task_idx in range(4):
            row, col = divmod(task_idx, 2)
            ax = fig.add_subplot(inner[row, col])
            axes.append(ax)
            points = coordinates[:, task_idx, :]
            xlim, ylim = centered_square_limits(points)
            rgba, order, density_info = density_style(
                points, COLORS[task_idx], args.density_k
            )
            ax.scatter(
                points[order, 0], points[order, 1], s=2.35,
                c=rgba[order], edgecolors="none", linewidths=0, rasterized=False,
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("white")
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_color("#AEB3B7")
                spine.set_linewidth(0.42)
            ax.set_title(
                DISPLAY_NAMES[task_idx], fontsize=10.8, fontweight="bold",
                color="#050607", pad=2.0,
            )
            density_metadata[method["key"]][DISPLAY_NAMES[task_idx]] = density_info
            density_metadata[method["key"]][DISPLAY_NAMES[task_idx]]["viewport"] = {
                "xlim": [float(xlim[0]), float(xlim[1])],
                "ylim": [float(ylim[0]), float(ylim[1])],
                "fit": "per-panel full-range centered isotropic square",
                "padding_fraction": 0.075,
                "points_clipped": 0,
            }
        group_axes.append(axes)
        left = min(ax.get_position().x0 for ax in axes)
        right = max(ax.get_position().x1 for ax in axes)
        setting_title = fig.text(
            (left + right) / 2, 0.947, method["title"], ha="center", va="center",
            fontsize=12.2, fontweight="bold", color="#050607",
        )
        # DejaVu Sans has no separate 900-weight face in the experiment image;
        # a very fine same-colour stroke gives the requested black visual weight
        # while keeping the text editable in SVG/PDF.
        setting_title.set_path_effects([
            path_effects.withStroke(linewidth=0.22, foreground="#050607")
        ])

    # Fine separators between method groups, constrained to panel area.
    for index in (0, 1):
        right = max(ax.get_position().x1 for ax in group_axes[index])
        left = min(ax.get_position().x0 for ax in group_axes[index + 1])
        x = (right + left) / 2
        fig.add_artist(Line2D(
            [x, x], [0.055, 0.982], transform=fig.transFigure,
            color="#D7DADD", linewidth=0.48, solid_capstyle="butt",
        ))
    return fig, density_metadata


def main():
    args = parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    if args.perplexity >= args.samples * 4:
        raise ValueError("t-SNE perplexity must be smaller than the joint point count")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_npz) if args.cache_npz else output_dir / "loom_tsne_representations.npz"

    reference_params = load_config(args.config, METHODS[0]["expid"])
    reference_params.update(num_workers=0, batch_size=args.batch_size)
    processor = FeatureProcessor(**reference_params)
    train_data, valid_data, test_data = build_dataset(processor, **reference_params)
    reference_params.update(train_data=train_data, valid_data=valid_data, test_data=test_data)
    data_dir = Path(reference_params["data_root"]) / reference_params["dataset_id"]
    feature_map = FeatureMap(reference_params["dataset_id"], str(data_dir))
    feature_map.load(str(data_dir / "feature_map.json"), reference_params)
    if list(feature_map.labels) not in (
        ["is_click", "cart", "fav", "buy"],
        ["is_click", "is_cart", "is_fav", "is_buy"],
    ):
        raise ValueError(f"Unexpected Taobao task order: {feature_map.labels}")

    frozen, sample_hash = freeze_test_samples(feature_map, reference_params, args.samples)
    method_infos = []
    representations = []
    cache_used = False
    if cache_path.is_file() and not args.force_extract:
        cache = np.load(cache_path, allow_pickle=False)
        cached_hash = str(cache["sample_hash"].item())
        if cached_hash != sample_hash:
            raise ValueError(
                f"Cached sample hash {cached_hash} does not match current test subset {sample_hash}; "
                "rerun with --force-extract"
            )
        for method in METHODS:
            representations.append(cache[method["key"]])
        method_infos = json.loads(str(cache["method_infos"].item()))
        cache_used = True
    else:
        for method in METHODS:
            values, info = load_and_extract(
                method, feature_map, reference_params, frozen, args
            )
            representations.append(values)
            method_infos.append(info)
        np.savez_compressed(
            cache_path,
            **{method["key"]: values for method, values in zip(METHODS, representations)},
            sample_hash=np.array(sample_hash),
            method_infos=np.array(json.dumps(method_infos)),
        )

    all_coordinates = []
    effective_rank_rows = []
    for method, values, info in zip(METHODS, representations, method_infos):
        coordinates, reduction_info = reduce_jointly(values, args)
        all_coordinates.append(coordinates)
        info["dimensionality_reduction"] = reduction_info
        task_rank_stats = {}
        for task_idx, (display, label) in enumerate(zip(DISPLAY_NAMES, feature_map.labels)):
            stats = legacy_effective_rank_stats(values[:, task_idx, :])
            task_rank_stats[display] = stats
            effective_rank_rows.append({
                "setting": method["title"],
                "experiment_id": method["expid"],
                "behavior": display,
                "label": label,
                **stats,
            })
        info["effective_rank"] = {
            "definition": "exp(Shannon entropy of projected covariance eigenvalue distribution)",
            "historical_implementation": "scripts/head_hidden_analysis.py::_summary",
            "projection_dimension": 64,
            "projection_seed": 20262027,
            "per_behavior": task_rank_stats,
            "mean_across_behaviors": float(np.mean([
                task_rank_stats[name]["effective_rank"] for name in DISPLAY_NAMES
            ])),
        }

    figure, density_metadata = draw_figure(all_coordinates, method_infos, args)
    pdf_path = output_dir / "loom_tsne_acm.pdf"
    png_path = output_dir / "loom_tsne_acm.png"
    svg_path = output_dir / "loom_tsne_acm.svg"
    figure.savefig(pdf_path, format="pdf", facecolor="white")
    figure.savefig(png_path, format="png", dpi=args.dpi, facecolor="white")
    figure.savefig(svg_path, format="svg", facecolor="white")
    plt.close(figure)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure_style": "Minimal ACM V3",
        "figure_size_inches": [7.16, 3.12],
        "outputs": {
            "pdf": str(pdf_path.resolve()),
            "png": str(png_path.resolve()),
            "svg": str(svg_path.resolve()),
        },
        "source": {
            "config_dir": str(Path(args.config).resolve()),
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
            "split": "test",
            "selection": f"first {args.samples} rows in deterministic test-loader order",
            "sample_count": args.samples,
            "sample_order_shared_across_settings": True,
            "sample_tensor_sha256": sample_hash,
            "representation_cache": str(cache_path.resolve()),
            "representation_cache_reused": cache_used,
        },
        "behavior_order": [
            {"display": display, "label": label, "color": color}
            for display, label, color in zip(DISPLAY_NAMES, feature_map.labels, COLORS)
        ],
        "settings": method_infos,
        "comparability": {
            "within_setting": (
                "All four task representations have equal dimensionality and are inputs to "
                "the task towers, so each setting uses one joint PCA+t-SNE. For readable paper "
                "layout, each panel uses a centered full-range square viewport; all points are "
                "retained, but absolute panel position and scale must not be compared visually."
            ),
            "across_settings": (
                "Each setting is reduced independently. Absolute 2D position, rotation, scale, "
                "and inter-setting distances are not comparable."
            ),
            "semantic_limit": (
                "LOOM columns are progressive recurrent stages. MMoE and STEM columns are "
                "task-conditioned routing outputs for the four behaviors, not temporal loop stages."
            ),
        },
        "density_rendering": density_metadata,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "random_seed": args.random_state,
        "synthetic_data_used": False,
        "point_coordinates_modified_after_tsne": False,
        "viewport_only_rescaled_per_panel": True,
    }
    metadata_path = output_dir / "loom_tsne_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    rank_json_path = output_dir / "loom_mmoe_stem_effective_rank.json"
    rank_json_path.write_text(json.dumps({
        "protocol": {
            "dataset": reference_params["dataset_id"],
            "split": "test",
            "samples": args.samples,
            "sample_tensor_sha256": sample_hash,
            "definition": "exp(Shannon entropy of projected covariance eigenvalue distribution)",
            "projection_dimension": 64,
            "projection_seed": 20262027,
            "source_implementation": "scripts/head_hidden_analysis.py::_summary",
        },
        "rows": effective_rank_rows,
    }, indent=2, ensure_ascii=False) + "\n")
    rank_csv_path = output_dir / "loom_mmoe_stem_effective_rank.csv"
    with rank_csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(effective_rank_rows[0]))
        writer.writeheader()
        writer.writerows(effective_rank_rows)
    print(json.dumps({
        "pdf": str(pdf_path),
        "png": str(png_path),
        "svg": str(svg_path),
        "metadata": str(metadata_path),
        "effective_rank_json": str(rank_json_path),
        "effective_rank_csv": str(rank_csv_path),
        "samples": args.samples,
        "sample_hash": sample_hash,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
