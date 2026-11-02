#!/usr/bin/env python3
"""Download pinned ready-to-use UniRank datasets with bounded concurrency."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DATASETS = {
    "KuaiRand_Video_Action": "salmon1802/KuaiRand",
    "Taobao_Action": "salmon1802/Taobao",
    "MerRec_Action": "salmon1802/MerRec",
    "QK_Video_Action": "salmon1802/QK-Video",
}


def download_one(name: str, repo: str, root: Path, workers: int) -> dict:
    api = HfApi()
    info = api.dataset_info(repo, files_metadata=True, timeout=60)
    output = root / name
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=info.sha,
        local_dir=output,
        max_workers=workers,
    )
    expected = {
        f.rfilename: int(f.size or 0)
        for f in info.siblings
        if not f.rfilename.endswith("/")
    }
    missing = []
    size_mismatch = []
    for rel, size in expected.items():
        path = output / rel
        if not path.is_file():
            missing.append(rel)
        elif size and path.stat().st_size != size:
            size_mismatch.append(
                {"file": rel, "expected": size, "actual": path.stat().st_size}
            )
    manifest = output / "block_manifest.json"
    metadata = output / "meta_data.json"
    if not manifest.is_file() or not metadata.is_file():
        raise RuntimeError(f"Dataset metadata incomplete: {name}")
    result = {
        "dataset_id": name,
        "repo": repo,
        "revision": info.sha,
        "used_storage": int(info.used_storage or 0),
        "expected_files": len(expected),
        "missing": missing,
        "size_mismatch": size_mismatch,
        "output": str(output),
        "complete": not missing and not size_mismatch,
        "block_manifest": json.loads(manifest.read_text()),
        "meta_data": json.loads(metadata.read_text()),
    }
    (output / "download_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({name: {k: result[k] for k in (
        "revision", "used_storage", "expected_files", "complete")}}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parallel-datasets", type=int, default=4)
    parser.add_argument("--workers-per-dataset", type=int, default=2)
    parser.add_argument("--datasets", nargs="*", choices=sorted(DATASETS),
                        default=sorted(DATASETS))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    chosen = [(name, DATASETS[name]) for name in args.datasets]
    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.parallel_datasets
    ) as pool:
        futures = [pool.submit(download_one, name, repo, args.output_root,
                               args.workers_per_dataset)
                   for name, repo in chosen]
        for future in futures:
            results.append(future.result())
    (args.output_root / "download_summary.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
