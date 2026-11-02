#!/usr/bin/env python3
"""Build behaviour-graph structure ablations for a dataset.

Variants keep the graph *shape* (same in/out degree profile) and only change
which behaviour occupies which node, plus chain-shaped controls.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "config_rankmixer_multiseed_confirmation"
SEEDS = [20262027, 20262028, 20262029, 20262030]


def chain(labels, order=None):
    labels = order or labels
    g = {labels[0]: []}
    for i, lab in enumerate(labels[1:], start=1):
        g[lab] = [labels[i - 1]]
    return g


def permuted_same_shape(graph, swap_a, swap_b):
    """Swap the roles of two labels while keeping every edge pattern."""
    labels = list(graph)
    mapping = {l: l for l in labels}
    mapping[swap_a], mapping[swap_b] = swap_b, swap_a
    out = {}
    for lab in labels:
        src = mapping[lab]
        parents = [mapping[p] for p in graph[src]]
        out[lab] = parents
    return out


def main():
    conf = yaml.safe_load((SRC / "model_config.yaml").read_text())
    specs = {
        "kuai": ("RankMixer_Kuai_DAGMeanPre2_S20262028", "KuaiRand_Video_Action", "Kuai",
                 [("RootLike", "is_click", "is_like"),
                  ("RootLongView", "is_click", "long_view"),
                  ("SinkSwap", "is_like", "is_follow")]),
        "merrec": ("RankMixer_MerRec_DAGGated_S20262028", "MerRec_Action", "MerRec",
                   [("RootSwap", "Like", "Cart"),
                    ("SinkSwap", "Cart", "Purchase")]),
        "qk": ("RankMixer_QK_DAGGated_S20262028", "QK_Video_Action", "QK",
               [("RootSwap", "click", "like"),
                ("SinkSwap", "like", "share")]),
    }
    for key, (base_name, dataset, prefix, swaps) in specs.items():
        base = conf[base_name]
        graph = base["behavior_graph"]
        labels = list(graph)
        DST = ROOT / f"config_rankmixer_structure_{key}"
        DST.mkdir(exist_ok=True)
        shutil.copy(SRC / "dataset_config.yaml", DST / "dataset_config.yaml")
        base_cfg = dict(conf["Base"])
        base_cfg["model_root"] = f"./runs_local/checkpoints_rankmixer_structure_{key}/"
        out = {"Base": base_cfg}

        def add(name, g):
            for seed in SEEDS:
                arm = dict(base)
                arm["seed"] = seed
                arm["behavior_graph"] = g
                out[f"Struct_{prefix}_{name}_S{seed}"] = arm

        add("True", graph)
        for name, a, b in swaps:
            add(name, permuted_same_shape(graph, a, b))
        # chain controls: natural label order, reversed, and one shuffled order
        add("ChainOrder", chain(labels))
        add("ChainReverse", chain(labels, list(reversed(labels))))
        perm = list(range(len(labels)))
        perm = perm[1:] + perm[:1]          # rotate, valid for any label count
        if len(labels) > 2:
            perm[0], perm[-1] = perm[-1], perm[0]
        rng_order = [labels[i] for i in perm]
        add("ChainShuffle", chain(labels, rng_order))
        (DST / "model_config.yaml").write_text(
            yaml.safe_dump(out, sort_keys=True, default_flow_style=False))
        print(key, "arms", len(out) - 1)


if __name__ == "__main__":
    main()
