#!/usr/bin/env python3
"""Resumable HTTP Range downloader for large public archives."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import time
import urllib.request
from pathlib import Path


CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+)", re.IGNORECASE)


def request_range(url: str, start: int, end: int, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "UniRank-range-downloader/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        content_range = response.headers.get("Content-Range", "")
        match = CONTENT_RANGE.search(content_range)
        if not match:
            raise RuntimeError(f"Missing/invalid Content-Range for {start}-{end}: {content_range!r}")
        actual_start, actual_end, total = map(int, match.groups())
        if actual_start != start or actual_end != end:
            raise RuntimeError(f"Wrong Content-Range: requested {start}-{end}, received {content_range}")
        if len(payload) != end - start + 1:
            raise RuntimeError(f"Short response for {start}-{end}: {len(payload)} bytes")
        return payload


def download_part(args: tuple[str, int, int, Path, int, int]) -> tuple[int, int, int]:
    url, start, end, target, retries, timeout = args
    expected = end - start + 1
    for attempt in range(1, retries + 1):
        try:
            payload = request_range(url, start, end, timeout)
            with target.open("r+b") as stream:
                stream.seek(start)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            return start, end, len(payload)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--part-size", type=int, default=64 << 20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.size <= 0 or args.part_size <= 0 or args.workers <= 0:
        raise SystemExit("size, part-size, and workers must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.stat().st_size != args.size:
        with args.output.open("ab") as stream:
            stream.truncate(args.size)

    jobs = []
    for start in range(0, args.size, args.part_size):
        end = min(args.size - 1, start + args.part_size - 1)
        jobs.append((args.url, start, end, args.output, args.retries, args.timeout))

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_part, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            start, end, size = future.result()
            completed += 1
            print(f"completed={completed}/{len(jobs)} range={start}-{end} bytes={size}", flush=True)
    print(f"complete={args.output} bytes={args.output.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
