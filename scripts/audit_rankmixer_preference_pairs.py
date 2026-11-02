#!/usr/bin/env python3
"""Estimate same-user Buy-vs-click-only pair coverage under shuffled batches."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/preprocessed/Taobao_Action/train/data")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20262027)
    parser.add_argument("--group-aware", action="store_true")
    parser.add_argument("--output", default="../reports/rankmixer_preference_pair_audit.json")
    args = parser.parse_args()

    table = ds.dataset(args.data, format="parquet").to_table(
        columns=["user_id", "is_click", "cart", "fav", "buy"]
    )
    values = {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}
    rng = np.random.default_rng(args.seed)
    total = len(values["user_id"])
    batch_stats = []
    if args.group_aware:
        users, inverse = np.unique(values["user_id"], return_inverse=True)
        group_rank = np.empty(len(users), dtype=np.int64)
        group_rank[rng.permutation(len(users))] = np.arange(len(users))
        stream = np.argsort(group_rank[inverse], kind="stable")
    for batch_index in range(args.batches):
        if args.group_aware:
            start = batch_index * args.batch_size
            if start + args.batch_size > total:
                break
            index = stream[start:start + args.batch_size]
        else:
            index = rng.choice(total, size=args.batch_size, replace=False)
        users = values["user_id"][index]
        buy = values["buy"][index] > 0.5
        click_only = (
            (values["is_click"][index] > 0.5)
            & ~(values["cart"][index] > 0.5)
            & ~(values["fav"][index] > 0.5)
            & ~buy
        )
        positive_users, positive_counts = np.unique(users[buy], return_counts=True)
        negative_users, negative_counts = np.unique(users[click_only], return_counts=True)
        common, pos_at, neg_at = np.intersect1d(
            positive_users, negative_users, return_indices=True
        )
        pair_count = int((positive_counts[pos_at] * negative_counts[neg_at]).sum())
        batch_stats.append({
            "buy_count": int(buy.sum()),
            "click_only_count": int(click_only.sum()),
            "paired_users": int(len(common)),
            "pair_count": pair_count,
        })
    result = {
        "data": str(Path(args.data).resolve()),
        "batch_size": args.batch_size,
        "batches": args.batches,
        "seed": args.seed,
        "group_aware": args.group_aware,
        "nonzero_pair_batch_rate": float(np.mean([x["pair_count"] > 0 for x in batch_stats])),
        "means": {
            key: float(np.mean([x[key] for x in batch_stats]))
            for key in batch_stats[0]
        },
        "percentiles_pair_count": {
            str(p): float(np.percentile([x["pair_count"] for x in batch_stats], p))
            for p in (0, 25, 50, 75, 100)
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
