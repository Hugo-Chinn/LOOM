#!/usr/bin/env python3
"""Merge selected Hugging Face TAAC parquet shards without materializing all rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def merge(source_dir: Path, output_file: Path) -> dict:
    shards = sorted(source_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards in {source_dir}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        raise FileExistsError(output_file)
    writer = None
    rows = 0
    try:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            if writer is None:
                writer = pq.ParquetWriter(output_file, parquet.schema_arrow,
                                          compression="snappy")
            elif parquet.schema_arrow != writer.schema:
                raise ValueError(f"Schema mismatch: {shard}")
            for batch in parquet.iter_batches(batch_size=65536):
                table = __import__("pyarrow").Table.from_batches([batch])
                writer.write_table(table)
                rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    return {
        "source_shards": [str(path) for path in shards],
        "source_bytes": sum(path.stat().st_size for path in shards),
        "rows": rows,
        "output": str(output_file),
        "output_bytes": output_file.stat().st_size,
        "sha256": sha256(output_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = {}
    for name in ("seq", "user_feat", "item_feat"):
        manifest[name] = merge(args.download_root / name,
                               args.output_root / f"{name}.parquet")
        print(json.dumps({name: manifest[name]}), flush=True)
    (args.output_root / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
