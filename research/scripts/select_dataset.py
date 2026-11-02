#!/usr/bin/env python3
"""Choose a reproducible UniRank dataset, preferring TAAC/TencentGR-10M."""
import argparse
import json
from pathlib import Path


def complete_raw(root: Path) -> bool:
    return all((root / f"{name}.parquet").is_file()
               for name in ("seq", "user_feat", "item_feat"))


def complete_blocked(root: Path) -> bool:
    return (root / "block_manifest.json").is_file() and all(
        any((root / split / kind).glob("*.parquet"))
        for split in ("train", "valid", "test")
        for kind in ("data", "user_info", "item_info")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taac-raw", type=Path, required=True)
    parser.add_argument("--taac-blocked", type=Path, required=True)
    parser.add_argument("--taobao-raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = {
        "priority": ["TencentGR_10M_Action", "Taobao_Action"],
        "TencentGR_10M_Action": {
            "raw_ready": complete_raw(args.taac_raw),
            "blocked_ready": complete_blocked(args.taac_blocked),
            "labels": ["is_click", "is_conversion"],
            "reason": "Preferred TAAC dataset with chronological item/action/timestamp history",
        },
        "Taobao_Action": {
            "raw_candidate_exists": bool(args.taobao_raw and args.taobao_raw.exists()),
            "labels": ["is_click", "cart", "fav", "buy"],
            "reason": "Fallback only if TAAC cannot be acquired or preprocessed",
        },
    }
    audit["selected"] = "TencentGR_10M_Action" if audit["TencentGR_10M_Action"]["raw_ready"] else "Taobao_Action"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
