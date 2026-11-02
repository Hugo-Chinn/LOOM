"""Diagnostics for the representations immediately before multi-task heads."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def _summary(x: torch.Tensor, y: torch.Tensor, projection_dim: int, seed: int):
    x = x.float()
    y = y.float().view(-1)
    norms = x.norm(dim=1)
    feature_std = x.std(dim=0, unbiased=False)
    mean_vector = x.mean(dim=0)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if x.size(1) > projection_dim:
        projection = torch.randn(
            x.size(1), projection_dim, generator=generator, dtype=torch.float32
        ) / math.sqrt(projection_dim)
        z = x @ projection
    else:
        z = x
    z = z - z.mean(dim=0, keepdim=True)
    covariance = z.T @ z / max(1, z.size(0) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    energy = eigenvalues.sum().clamp_min(1e-12)
    probabilities = eigenvalues / energy
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )

    positive = y > 0.5
    negative = ~positive
    fisher = None
    centroid_distance = None
    if positive.any() and negative.any():
        pos = z[positive]
        neg = z[negative]
        pos_mean = pos.mean(dim=0)
        neg_mean = neg.mean(dim=0)
        centroid_distance = (pos_mean - neg_mean).norm()
        within = pos.var(dim=0, unbiased=False).mean() + neg.var(dim=0, unbiased=False).mean()
        fisher = centroid_distance.square() / within.clamp_min(1e-12)

    return {
        "samples": int(x.size(0)),
        "dimension": int(x.size(1)),
        "global_mean": float(x.mean()),
        "global_std": float(x.std(unbiased=False)),
        "mean_abs": float(x.abs().mean()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std(unbiased=False)),
        "mean_vector_norm": float(mean_vector.norm()),
        "feature_std_mean": float(feature_std.mean()),
        "near_zero_fraction": float((x.abs() < 1e-6).float().mean()),
        "large_value_fraction": float((x.abs() > 5).float().mean()),
        "effective_rank": float(effective_rank),
        "top1_variance_fraction": float(eigenvalues[-1] / energy),
        "positive_rate": float(positive.float().mean()),
        "centroid_distance_projected": None if centroid_distance is None else float(centroid_distance),
        "fisher_ratio_projected": None if fisher is None else float(fisher),
    }


def analyze_head_inputs(
    model,
    data_generator,
    output_path,
    max_batches=8,
    max_samples=2048,
    projection_dim=64,
    seed=20262027,
):
    """Capture the first invocation of each task tower on each validation batch."""
    labels = list(model.feature_map.labels)
    captures = {label: [] for label in labels}
    targets = {label: [] for label in labels}
    current = {}
    handles = []

    def make_hook(index):
        def hook(_module, inputs):
            if index not in current:
                current[index] = inputs[0].detach().float().cpu()
        return hook

    taps = getattr(model, "task_head_input_taps", None)
    if taps is not None:
        for index, tap in enumerate(taps):
            handles.append(tap.register_forward_pre_hook(make_hook(index)))
    else:
        for index, tower in enumerate(model.tower):
            handles.append(tower.register_forward_pre_hook(make_hook(index)))

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_generator):
            if batch_index >= max_batches:
                break
            current.clear()
            with model._get_amp_context():
                model(batch)
            batch_labels = model.get_labels(batch)
            for index, label in enumerate(labels):
                remaining = max_samples - sum(t.size(0) for t in captures[label])
                if remaining <= 0:
                    continue
                hidden = current[index][:remaining]
                captures[label].append(hidden)
                targets[label].append(batch_labels[index].detach().float().cpu()[:remaining])
            if all(sum(t.size(0) for t in captures[label]) >= max_samples for label in labels):
                break

    for handle in handles:
        handle.remove()

    hidden = {label: torch.cat(captures[label], dim=0) for label in labels}
    target = {label: torch.cat(targets[label], dim=0) for label in labels}
    task_stats = {
        # Use the same random projection for every task so projected covariance
        # statistics are directly comparable across progressive stages.
        label: _summary(hidden[label], target[label], projection_dim, seed)
        for index, label in enumerate(labels)
    }

    pair_stats = {}
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1:]:
            count = min(hidden[left].size(0), hidden[right].size(0))
            x = hidden[left][:count]
            y = hidden[right][:count]
            delta = y - x
            pair_stats[f"{left}->{right}"] = {
                "cosine_mean": float(F.cosine_similarity(x, y, dim=1).mean()),
                "cosine_std": float(F.cosine_similarity(x, y, dim=1).std(unbiased=False)),
                "delta_norm_mean": float(delta.norm(dim=1).mean()),
                "normalized_delta_mean": float(
                    (delta.norm(dim=1) / x.norm(dim=1).clamp_min(1e-12)).mean()
                ),
            }

    # Keep the legacy Taobao chain view, but also expose topology-aware edges.
    # The latter is required for cross-dataset DAG analysis where label order is
    # not the computation graph (e.g. Like->{Cart,Offer,Checkout}->Purchase).
    adjacent = {
        key: pair_stats[key]
        for key in ("is_click->cart", "cart->fav", "fav->buy")
        if key in pair_stats
    }
    graph_edges = {}
    graph_parents = getattr(model, "graph_parents", None)
    if graph_parents is not None:
        for child_index, parent_indices in enumerate(graph_parents):
            child = labels[child_index]
            for parent_index in parent_indices:
                parent = labels[parent_index]
                key = f"{parent}->{child}"
                # Compute in the true parent->child direction. pair_stats only
                # stores label-order pairs, while DAG edges may point against
                # that order (QK labels list Follow before Like).
                count = min(hidden[parent].size(0), hidden[child].size(0))
                x = hidden[parent][:count]
                y = hidden[child][:count]
                delta = y - x
                graph_edges[key] = {
                    "cosine_mean": float(F.cosine_similarity(x, y, dim=1).mean()),
                    "cosine_std": float(F.cosine_similarity(x, y, dim=1).std(unbiased=False)),
                    "delta_norm_mean": float(delta.norm(dim=1).mean()),
                    "normalized_delta_mean": float(
                        (delta.norm(dim=1) / x.norm(dim=1).clamp_min(1e-12)).mean()
                    ),
                }
    elif getattr(model, "model_type", None) != "parallel":
        for index in range(len(labels) - 1):
            key = f"{labels[index]}->{labels[index + 1]}"
            if key in pair_stats:
                graph_edges[key] = pair_stats[key]
    result = {
        "model_id": model.model_id,
        "dataset_id": model.feature_map.dataset_id,
        "max_batches": int(max_batches),
        "max_samples": int(max_samples),
        "task_stats": task_stats,
        "pair_stats": pair_stats,
        "adjacent_progression": adjacent,
        "graph_edge_progression": graph_edges,
        "loop_metadata": {
            "topology": getattr(model, "loop_topology", None),
            "gates": (
                None if getattr(model, "loop_gate_logits", None) is None
                else torch.sigmoid(model.loop_gate_logits.detach()).float().cpu().tolist()
            ),
            "branch_fusion_gates": (
                None if getattr(model, "branch_fusion_logits", None) is None
                else torch.sigmoid(model.branch_fusion_logits.detach()).float().cpu().tolist()
            ),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
