import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from AutoAna.analyzer import AnalysisOptions, analyze_dataset
from AutoAna.dependencies import DependencyOptions, analyze_feedback_dependencies
from AutoAna.layout import discover_datasets, layout_from_config, layout_from_directory
from AutoAna.report import render_markdown, write_json
from AutoAna.graph import build_conflict_graph, render_mermaid


def _write_dataset(root: Path, invalid_item_index: bool = False, include_conversion: bool = False) -> None:
    for split, label_values in {
        "train": [0.0, 1.0, 0.0],
        "valid": [0.0, 1.0],
        "test": [1.0, 0.0],
    }.items():
        for table in ("data", "user_info", "item_info"):
            (root / split / table).mkdir(parents=True, exist_ok=True)
        item_indices = [0, 99 if invalid_item_index else 1, 2][: len(label_values)]
        data = {
                    "user_index": list(range(len(label_values))),
                    "item_index": item_indices,
                    "seq_len": list(range(len(label_values))),
                    "user_id": [10 + index for index in range(len(label_values))],
                    "is_click": label_values,
                }
        if include_conversion:
            data["is_conversion"] = [0.0, 1.0, 0.0][: len(label_values)]
        pq.write_table(pa.table(data),
            root / split / "data" / "part-00000.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "user_index": list(range(len(label_values))),
                    "full_item_seq": [list(range(index)) for index in range(len(label_values))],
                    "full_action_seq": [[1] * index for index in range(len(label_values))],
                }
            ),
            root / split / "user_info" / "part-00000.parquet",
        )
        pq.write_table(
            pa.table({"item_index": [0, 1, 2], "item_id": [100, 101, 102]}),
            root / split / "item_info" / "part-00000.parquet",
        )


def test_analyze_and_render_report(tmp_path: Path):
    root = tmp_path / "demo"
    _write_dataset(root, include_conversion=True)
    report = analyze_dataset(
        layout_from_directory(root),
        AnalysisOptions(sample_rows_per_file=10, drift_warning_threshold=0.20),
    )

    assert report["summary"]["status"] == "ok"
    assert report["summary"]["total_data_rows"] == 7
    assert report["labels"] == ["is_click", "is_conversion"]
    assert report["splits"]["train"]["block_pairing"]["paired"]
    assert report["label_analysis"]["is_click"]["train"]["positive_rate"] == 1 / 3
    assert "AutoAna report: demo" in render_markdown(report)
    graph = build_conflict_graph(report["feedback_dependencies"])
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 2
    assert "conflict_rate" in graph["edge_semantics"]
    assert "mermaid" in render_mermaid(graph)

    output = write_json(report, tmp_path / "report.json")
    assert json.loads(output.read_text())["summary"]["status"] == "ok"


def test_invalid_index_is_reported(tmp_path: Path):
    root = tmp_path / "broken"
    _write_dataset(root, invalid_item_index=True)
    report = analyze_dataset(layout_from_directory(root), AnalysisOptions(sample_rows_per_file=10))

    assert report["summary"]["status"] == "error"
    assert any(issue["code"] == "index_out_of_range" for issue in report["issues"])


def test_discovery(tmp_path: Path):
    root = tmp_path / "datasets" / "demo"
    _write_dataset(root)
    assert discover_datasets(tmp_path / "datasets") == [root]


def test_grouped_feature_names_are_expanded(tmp_path: Path):
    config = tmp_path / "dataset_config.yaml"
    config.write_text(
        """
demo:
  data_root: .
  feature_cols:
    - {name: [a, b], active: true, type: categorical, vocab_size: 5}
  label_col:
    - {name: label, dtype: float}
""".strip(),
        encoding="utf-8",
    )
    layout = layout_from_config(config, "demo")
    assert [feature.name for feature in layout.features] == ["a", "b"]
    assert layout.labels == ["label"]


def test_feedback_dependency_direction(tmp_path: Path):
    root = tmp_path / "dependency"
    _write_dataset(root, include_conversion=True)
    result = analyze_feedback_dependencies(
        layout_from_directory(root), ["is_click", "is_conversion"],
        DependencyOptions(min_support=1),
    )
    click_to_conversion = result["aggregate"]["is_click"]["is_conversion"]
    conversion_to_click = result["aggregate"]["is_conversion"]["is_click"]
    assert click_to_conversion["counts"] == {"00": 3, "01": 1, "10": 1, "11": 2}
    assert conversion_to_click["source_to_target"]["violation_rate"] == 1 / 3
