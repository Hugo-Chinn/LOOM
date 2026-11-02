#!/usr/bin/env python3
"""Build deterministic matched case-control pairs from transition-aware rows."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EDGE_SPECS = {
    "edge_click_cart": {
        "child": "cart",
        "risk": lambda frame: (frame["is_click"] > 0) | (frame["cart"] > 0),
    },
    "edge_click_fav": {
        "child": "fav",
        "risk": lambda frame: (frame["is_click"] > 0) | (frame["fav"] > 0),
    },
    "edge_cart_buy": {
        "child": "buy",
        "risk": lambda frame: (frame["cart"] > 0),
    },
    "edge_fav_buy": {
        "child": "buy",
        "risk": lambda frame: (frame["fav"] > 0),
    },
    "edge_click_buy_direct": {
        "child": "buy",
        "risk": lambda frame: (frame["is_click"] > 0),
    },
}


def build_transition_pairs(
    frame: pd.DataFrame,
    edge: str,
    controls_per_positive: int = 1,
    seed: int = 20262027,
    strata_columns: tuple[str, ...] = ("hour",),
) -> pd.DataFrame:
    """Match observed edge positives to non-transition controls in the same risk set.

    Controls are sampled without replacement within each positive when possible.
    The same control may be reused for different positives, which keeps the
    construction deterministic and avoids dropping rare positive strata.
    """
    if edge not in EDGE_SPECS:
        raise ValueError(f"Unsupported edge={edge}")
    if controls_per_positive <= 0:
        raise ValueError("controls_per_positive must be positive")
    required = {"source_row_id", edge, EDGE_SPECS[edge]["child"], *strata_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing transition-pair columns: {sorted(missing)}")

    work = frame.copy()
    risk = EDGE_SPECS[edge]["risk"](work).to_numpy(dtype=bool, copy=False)
    positive = risk & (work[edge].to_numpy() > 0)
    # Controls must be in the same parent-risk state and have no observed edge
    # and no child event. This avoids treating a different path to the same child
    # as a negative transition.
    child = EDGE_SPECS[edge]["child"]
    control = risk & (work[edge].to_numpy() <= 0) & (work[child].to_numpy() <= 0)
    positives = work.loc[positive].copy()
    controls = work.loc[control].copy()
    if len(positives) == 0 or len(controls) == 0:
        return pd.DataFrame(columns=[
            "edge", "positive_source_row_id", "control_source_row_id",
            "pair_rank", *strata_columns,
        ])

    rng = np.random.default_rng(seed)
    if strata_columns:
        control_groups = {
            key if isinstance(key, tuple) else (key,): group
            for key, group in controls.groupby(list(strata_columns), sort=False)
        }
    else:
        control_groups = {(): controls}

    rows = []
    for _, positive_row in positives.sort_values("source_row_id").iterrows():
        key = tuple(positive_row[column] for column in strata_columns)
        candidates = control_groups.get(key)
        if candidates is None or len(candidates) == 0:
            continue
        candidate_ids = candidates["source_row_id"].to_numpy(dtype=np.int64, copy=False)
        take = min(controls_per_positive, len(candidate_ids))
        selected = rng.choice(candidate_ids, size=take, replace=False)
        for rank, control_id in enumerate(selected):
            row = {
                "edge": edge,
                "positive_source_row_id": int(positive_row["source_row_id"]),
                "control_source_row_id": int(control_id),
                "pair_rank": int(rank),
            }
            row.update({column: positive_row[column] for column in strata_columns})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edge", choices=sorted(EDGE_SPECS), required=True)
    parser.add_argument("--controls_per_positive", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20262027)
    parser.add_argument("--strata", nargs="*", default=["hour"])
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    pairs = build_transition_pairs(
        frame,
        edge=args.edge,
        controls_per_positive=args.controls_per_positive,
        seed=args.seed,
        strata_columns=tuple(args.strata),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.output, index=False)
    print(f"wrote {len(pairs):,} pairs to {args.output}")


if __name__ == "__main__":
    main()
