"""Command line interface for AutoAna."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .analyzer import AnalysisOptions, analyze_dataset
from .layout import discover_datasets, load_dataset_layout
from .report import write_json, write_markdown
from .graph import build_conflict_graph, write_graph


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m AutoAna",
        description="Automated analysis and validation for UniRank blocked Parquet datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one UniRank dataset")
    analyze.add_argument("--dataset-dir", type=Path, help="Root containing train/valid/test")
    analyze.add_argument("--config", type=Path, help="dataset_config.yaml or its directory")
    analyze.add_argument("--dataset-id", help="Dataset key in the YAML config")
    analyze.add_argument("--output-dir", type=Path, default=Path("AutoAna/reports"))
    analyze.add_argument("--sample-rows-per-file", type=int, default=2000)
    analyze.add_argument("--max-files-per-table", type=int)
    analyze.add_argument("--top-k", type=int, default=10)
    analyze.add_argument("--distinct-cap", type=int, default=10000)
    analyze.add_argument("--drift-warning-threshold", type=float, default=0.10)
    analyze.add_argument("--imbalance-warning-threshold", type=float, default=0.01)
    analyze.add_argument(
        "--fail-on",
        choices=("never", "error", "warning"),
        default="error",
        help="Exit non-zero at this severity (default: error)",
    )

    graph = subparsers.add_parser("graph", help="Write a feedback conflict graph from an AutoAna JSON report")
    graph.add_argument("--report", type=Path, required=True, help="AutoAna report.json")
    graph.add_argument("--output-dir", type=Path, required=True)

    discover = subparsers.add_parser("discover", help="Find blocked datasets below a directory")
    discover.add_argument("root", type=Path)
    discover.add_argument("--json", action="store_true", help="Print machine-readable output")
    return parser


def _run_analyze(args: argparse.Namespace) -> int:
    layout = load_dataset_layout(
        dataset_dir=args.dataset_dir,
        config=args.config,
        dataset_id=args.dataset_id,
    )
    options = AnalysisOptions(
        sample_rows_per_file=args.sample_rows_per_file,
        max_files_per_table=args.max_files_per_table,
        top_k=args.top_k,
        distinct_cap=args.distinct_cap,
        drift_warning_threshold=args.drift_warning_threshold,
        severe_imbalance_threshold=args.imbalance_warning_threshold,
    )
    report = analyze_dataset(layout, options)
    output_dir = args.output_dir.expanduser().resolve() / layout.dataset_id
    json_path = write_json(report, output_dir / "report.json")
    markdown_path = write_markdown(report, output_dir / "report.md")

    summary = report["summary"]
    print(f"Dataset: {layout.dataset_id}")
    print(f"Status: {summary['status'].upper()}")
    print(f"Rows: {summary['total_data_rows']:,}")
    print(f"Errors: {summary['error_count']}; warnings: {summary['warning_count']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")

    if args.fail_on == "warning" and summary["status"] in {"warning", "error"}:
        return 2
    if args.fail_on == "error" and summary["status"] == "error":
        return 2
    return 0


def _run_discover(args: argparse.Namespace) -> int:
    datasets = discover_datasets(args.root)
    if args.json:
        print(json.dumps([str(path) for path in datasets], ensure_ascii=False, indent=2))
    else:
        for path in datasets:
            print(path)
    return 0


def _run_graph(args: argparse.Namespace) -> int:
    report = json.loads(args.report.expanduser().resolve().read_text(encoding="utf-8"))
    dependency_report = report.get("feedback_dependencies", report)
    graph = report.get("feedback_conflict_graph") or build_conflict_graph(dependency_report)
    paths = write_graph(graph, args.output_dir)
    print(f"JSON: {paths['json']}")
    print(f"Markdown: {paths['markdown']}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            return _run_analyze(args)
        if args.command == "discover":
            return _run_discover(args)
        if args.command == "graph":
            return _run_graph(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    return 1
