#!/usr/bin/env python3
"""Preserve the complete MerRec event stream in chronological blocked Parquet.

Unlike the benchmark preprocessing script, this command never collapses events to
one view anchor. Every raw event remains one output row and keeps its event type,
timestamp, user/item/session identifiers, and a one-hot feedback projection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


EVENTS = (
    "item_view",
    "item_like",
    "item_add_to_cart_tap",
    "offer_make",
    "buy_start",
    "buy_comp",
)
FEEDBACK_COLUMNS = {
    "item_like": "Like",
    "item_add_to_cart_tap": "Cart",
    "offer_make": "Offer",
    "buy_start": "Checkout",
    "buy_comp": "Purchase",
}
RAW_COLUMNS = [
    "user_id", "stime", "session_id", "sequence_id", "sequence_length",
    "event_id", "item_id", "product_id", "price", "c0_name", "c0_id",
    "c1_name", "c1_id", "c2_name", "c2_id", "brand_name", "brand_id",
    "item_condition_id", "item_condition_name", "size_name", "size_id",
    "color", "shipper_id", "shipper_name",
]
OUTPUT_COLUMNS = RAW_COLUMNS + ["timestamp_ms", "date", "event_code"] + list(FEEDBACK_COLUMNS.values())


def files_for(root: Path) -> List[Path]:
    root = root / "20231001" if (root / "20231001").is_dir() else root
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw MerRec parquet files found under {root}")
    return files


def parse_time(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    valid = dt.notna()
    timestamp_ms = pd.Series(np.zeros(len(series), dtype=np.int64), index=series.index)
    timestamp_ms.loc[valid] = (dt.loc[valid].astype("int64") // 1_000_000).astype(np.int64)
    date = pd.Series(np.zeros(len(series), dtype=np.int32), index=series.index)
    date.loc[valid] = dt.loc[valid].dt.strftime("%Y%m%d").astype(np.int32)
    return valid, timestamp_ms, date


def normalize(batch: pa.RecordBatch) -> pd.DataFrame:
    df = batch.to_pandas()
    for column in RAW_COLUMNS:
        if column not in df:
            df[column] = None
    valid, timestamp_ms, date = parse_time(df["stime"])
    df = df.loc[valid].copy()
    timestamp_ms = timestamp_ms.loc[valid]
    date = date.loc[valid]
    df["timestamp_ms"] = timestamp_ms.to_numpy(dtype=np.int64)
    df["date"] = date.to_numpy(dtype=np.int32)
    df["event_id"] = df["event_id"].astype("string")
    df = df[df["event_id"].isin(EVENTS)].copy()
    df["event_code"] = pd.Categorical(df["event_id"], categories=EVENTS).codes.astype(np.int8)
    for event, column in FEEDBACK_COLUMNS.items():
        df[column] = (df["event_id"] == event).astype(np.float32)
    return df[OUTPUT_COLUMNS]


def split_for_date(date: int, boundaries: Dict[str, int]) -> str:
    if date < boundaries["valid_start"]:
        return "train"
    if date < boundaries["test_start"]:
        return "valid"
    return "test"


def date_boundaries(files: Iterable[Path], batch_size: int) -> Dict[str, Any]:
    dates = set()
    for path in files:
        parquet = pq.ParquetFile(path)
        if "stime" not in parquet.schema_arrow.names:
            raise ValueError(f"Missing stime in {path}")
        for batch in parquet.iter_batches(columns=["stime"], batch_size=batch_size):
            _, _, date = parse_time(batch.to_pandas()["stime"])
            dates.update(int(x) for x in date.unique() if int(x) > 0)
    ordered = sorted(dates)
    if len(ordered) < 3:
        raise ValueError(f"Need at least three dates, found {ordered}")
    n_train = max(1, int(round(len(ordered) * 0.8)))
    n_valid = max(1, int(round(len(ordered) * 0.1)))
    if n_train + n_valid >= len(ordered):
        n_train, n_valid = len(ordered) - 2, 1
    return {
        "dates": ordered,
        "train_start": ordered[0],
        "train_end": ordered[n_train - 1],
        "valid_start": ordered[n_train],
        "valid_end": ordered[n_train + n_valid - 1],
        "test_start": ordered[n_train + n_valid],
        "test_end": ordered[-1],
    }


class WriterSet:
    def __init__(self, root: Path, rows_per_file: int):
        self.root = root
        self.rows_per_file = rows_per_file
        self.buffers: Dict[str, List[pd.DataFrame]] = {split: [] for split in ("train", "valid", "test")}
        self.counts = Counter()
        self.parts = Counter()

    def append(self, split: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.buffers[split].append(frame)
        self.counts[split] += len(frame)
        buffered = sum(len(x) for x in self.buffers[split])
        if buffered >= self.rows_per_file:
            self.flush(split)

    def flush(self, split: str) -> None:
        frames = self.buffers[split]
        if not frames:
            return
        frame = pd.concat(frames, ignore_index=True)
        out_dir = self.root / split / "events"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"part-{self.parts[split]:05d}.parquet"
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, path, compression="zstd")
        self.parts[split] += 1
        self.buffers[split] = []

    def close(self) -> None:
        for split in self.buffers:
            self.flush(split)


def preprocess(input_root: Path, output_root: Path, batch_size: int, rows_per_file: int) -> Dict[str, Any]:
    files = files_for(input_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    boundaries = date_boundaries(files, batch_size)
    writers = WriterSet(output_root, rows_per_file)
    event_counts = Counter()
    schema_names = set()
    raw_rows = 0
    kept_rows = 0
    for index, path in enumerate(files, 1):
        parquet = pq.ParquetFile(path)
        raw_rows += int(parquet.metadata.num_rows)
        schema_names.update(parquet.schema_arrow.names)
        for batch in parquet.iter_batches(columns=[c for c in RAW_COLUMNS if c in parquet.schema_arrow.names], batch_size=batch_size):
            frame = normalize(batch)
            kept_rows += len(frame)
            event_counts.update(frame["event_id"].astype(str).tolist())
            if frame.empty:
                continue
            for split, group in frame.groupby(frame["date"].map(lambda x: split_for_date(int(x), boundaries)), sort=False):
                writers.append(split, group.reset_index(drop=True))
        print(f"processed={index}/{len(files)} raw_rows={raw_rows:,} kept_rows={kept_rows:,}", flush=True)
    writers.close()
    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(input_root.resolve()),
        "raw_file_count": len(files),
        "raw_rows": raw_rows,
        "kept_rows": kept_rows,
        "event_types": list(EVENTS),
        "feedback_columns": list(FEEDBACK_COLUMNS.values()),
        "event_counts": dict(event_counts),
        "source_schema_columns": sorted(schema_names),
        "split_boundaries": boundaries,
        "split_rows": dict(writers.counts),
        "output_parts": dict(writers.parts),
        "preserves_atomic_events": True,
        "aggregation": None,
    }
    (output_root / "meta_data.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--rows-per-file", type=int, default=1_000_000)
    args = parser.parse_args()
    metadata = preprocess(args.input_root, args.output_root, args.batch_size, args.rows_per_file)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
