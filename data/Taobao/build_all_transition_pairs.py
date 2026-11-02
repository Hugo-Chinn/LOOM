#!/usr/bin/env python3
"""Build and audit matched transition pairs for every Taobao split/block."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_transition_pairs import EDGE_SPECS, build_transition_pairs


def build_all_pairs(
    dataset_root: Path,
    output_root: Path,
    controls_per_positive: int,
    seed: int,
    strata_columns: tuple[str, ...],
) -> dict:
    audit = {
        "dataset_root": str(dataset_root.resolve()),
        "output_root": str(output_root.resolve()),
        "controls_per_positive": controls_per_positive,
        "seed": seed,
        "strata_columns": list(strata_columns),
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        transition_dir = dataset_root / split / "transition_info"
        split_out = output_root / split
        split_audit = {"blocks": {}, "totals": {}}
        for input_path in sorted(transition_dir.glob("part-*.parquet")):
            block = input_path.stem
            frame = pd.read_parquet(input_path)
            block_audit = {
                "rows": len(frame),
                "edges": {},
                "delay_seconds": {},
            }
            for delay_column in ("time_to_cart", "time_to_fav", "time_to_buy"):
                valid = frame.loc[frame[delay_column] >= 0, delay_column]
                if len(valid):
                    block_audit["delay_seconds"][delay_column] = {
                        "count": int(len(valid)),
                        "p50": float(valid.quantile(0.50)),
                        "p90": float(valid.quantile(0.90)),
                        "p99": float(valid.quantile(0.99)),
                    }
            for edge in EDGE_SPECS:
                positives = int((frame[edge] > 0).sum())
                pairs = build_transition_pairs(
                    frame,
                    edge=edge,
                    controls_per_positive=controls_per_positive,
                    seed=seed,
                    strata_columns=strata_columns,
                )
                edge_dir = split_out / edge
                edge_dir.mkdir(parents=True, exist_ok=True)
                pairs.to_parquet(edge_dir / input_path.name, index=False)
                paired_positives = int(
                    pairs["positive_source_row_id"].nunique()) if len(pairs) else 0
                block_audit["edges"][edge] = {
                    "positives": positives,
                    "pairs": int(len(pairs)),
                    "paired_positives": paired_positives,
                    "positive_coverage": (
                        paired_positives / positives if positives else 0.0
                    ),
                }
            split_audit["blocks"][block] = block_audit

        totals = {}
        for edge in EDGE_SPECS:
            edge_rows = [
                block["edges"][edge]
                for block in split_audit["blocks"].values()
            ]
            positives = sum(row["positives"] for row in edge_rows)
            pairs = sum(row["pairs"] for row in edge_rows)
            paired_positives = sum(row["paired_positives"] for row in edge_rows)
            totals[edge] = {
                "positives": positives,
                "pairs": pairs,
                "paired_positives": paired_positives,
                "positive_coverage": (
                    paired_positives / positives if positives else 0.0
                ),
            }
        split_audit["totals"] = totals
        audit["splits"][split] = split_audit
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--controls_per_positive", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20262027)
    parser.add_argument("--strata", nargs="*", default=["hour"])
    args = parser.parse_args()

    audit = build_all_pairs(
        args.dataset_root,
        args.output_root,
        controls_per_positive=args.controls_per_positive,
        seed=args.seed,
        strata_columns=tuple(args.strata),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "transition_pair_audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
