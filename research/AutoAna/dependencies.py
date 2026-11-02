"""Exact pairwise dependency analysis for binary feedback labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

from .layout import DatasetLayout, SPLITS


@dataclass(frozen=True)
class DependencyOptions:
    strict_violation_rate: float = 0.0
    approximate_violation_rate: float = 0.01
    min_support: int = 100
    batch_size: int = 65536

    def __post_init__(self) -> None:
        if not 0 <= self.strict_violation_rate <= self.approximate_violation_rate <= 1:
            raise ValueError("Expected 0 <= strict_violation_rate <= approximate_violation_rate <= 1")
        if self.min_support < 1 or self.batch_size < 1:
            raise ValueError("min_support and batch_size must be positive")


def _parquet_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    return sorted(path.rglob("*.parquet")) if path.is_dir() else []


def _binary(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in "f":
        array = np.nan_to_num(array, nan=0.0)
    return array.astype(np.float64, copy=False) > 0


def _empty_counts(size: int) -> np.ndarray:
    return np.zeros((size, size, 2, 2), dtype=np.int64)


def _update_counts(counts: np.ndarray, values: Mapping[str, Any], labels: Sequence[str]) -> int:
    binary = np.column_stack([_binary(values[label]) for label in labels])
    for left, _ in enumerate(labels):
        for right, _ in enumerate(labels):
            # Encode [A=0,B=0], [A=0,B=1], [A=1,B=0], [A=1,B=1] as 0..3.
            code = binary[:, left].astype(np.int8) * 2 + binary[:, right].astype(np.int8)
            counts[left, right] += np.bincount(code, minlength=4).reshape(2, 2)
    return int(binary.shape[0])


def _pair_result(
    source: str,
    target: str,
    table: np.ndarray,
    options: DependencyOptions,
) -> Dict[str, Any]:
    n00, n01 = int(table[0, 0]), int(table[0, 1])
    n10, n11 = int(table[1, 0]), int(table[1, 1])
    source_support = n10 + n11
    target_support = n01 + n11
    total = n00 + n01 + n10 + n11
    source_to_target_violation = n10 / source_support if source_support else None
    target_to_source_violation = n01 / target_support if target_support else None
    p_target_given_source = n11 / source_support if source_support else None
    p_source_given_target = n11 / target_support if target_support else None
    independent = (source_support / total) * (target_support / total) if total else 0.0
    joint = n11 / total if total else 0.0
    phi_denominator = ((n10 + n11) * (n00 + n01) * (n01 + n11) * (n00 + n10)) ** 0.5
    phi = ((n11 * n00) - (n10 * n01)) / phi_denominator if phi_denominator else 0.0
    lift = joint / independent if independent else None

    def classify(violation: Optional[float], support: int) -> str:
        if violation is None or support < options.min_support:
            return "insufficient_support"
        if violation <= options.strict_violation_rate:
            return "strict"
        if violation <= options.approximate_violation_rate:
            return "approximate"
        return "unsupported"

    return {
        "source": source,
        "target": target,
        "counts": {"00": n00, "01": n01, "10": n10, "11": n11},
        "support": {"source": source_support, "target": target_support, "total": total},
        "source_to_target": {
            "conditional_probability": p_target_given_source,
            "violation_rate": source_to_target_violation,
            "classification": classify(source_to_target_violation, source_support),
        },
        "target_to_source": {
            "conditional_probability": p_source_given_target,
            "violation_rate": target_to_source_violation,
            "classification": classify(target_to_source_violation, target_support),
        },
        "association": {"joint_probability": joint, "lift": lift, "phi": phi},
        "temporal_precedence": "unavailable_from_aggregated_label_columns",
    }


def _scan_split(layout: DatasetLayout, split: str, labels: Sequence[str], options: DependencyOptions) -> Dict[str, Any]:
    counts = _empty_counts(len(labels))
    row_count = 0
    files = _parquet_files(layout.table_path(split, "data"))
    analyzed_files = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        missing = [label for label in labels if label not in available]
        if missing:
            raise ValueError(f"{split}/data/{path.name} is missing feedback columns: {missing}")
        analyzed_files += 1
        for batch in parquet.iter_batches(columns=list(labels), batch_size=options.batch_size):
            row_count += _update_counts(counts, batch.to_pydict(), labels)
    pairs = {
        source: {
            target: _pair_result(source, target, counts[i, j], options)
            for j, target in enumerate(labels)
            if source != target
        }
        for i, source in enumerate(labels)
    }
    return {"row_count": row_count, "file_count": len(files), "analyzed_file_count": analyzed_files, "pairs": pairs}


def _aggregate_counts(
    split_reports: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
    options: DependencyOptions,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    # Reconstructing from pair outputs keeps the JSON self-contained and avoids exposing arrays.
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for source in labels:
        result[source] = {}
        for target in labels:
            if source == target:
                continue
            keys = ("00", "01", "10", "11")
            counts = {key: sum(report["pairs"][source][target]["counts"][key] for report in split_reports.values()) for key in keys}
            result[source][target] = _pair_result(
                source,
                target,
                np.array([[counts["00"], counts["01"]], [counts["10"], counts["11"]]]),
                options,
            )
    return result


def analyze_feedback_dependencies(
    layout: DatasetLayout,
    labels: Sequence[str],
    options: Optional[DependencyOptions] = None,
) -> Dict[str, Any]:
    options = options or DependencyOptions()
    labels = list(dict.fromkeys(labels))
    if len(labels) < 2:
        return {"labels": labels, "options": asdict(options), "available": False, "reason": "fewer_than_two_feedback_labels"}
    split_reports = {split: _scan_split(layout, split, labels, options) for split in SPLITS}
    aggregate = _aggregate_counts(split_reports, labels, options)
    candidates = []
    for source in labels:
        for target in labels:
            if source == target:
                continue
            relation = aggregate[source][target]["source_to_target"]
            support = aggregate[source][target]["support"]["source"]
            if relation["classification"] in {"strict", "approximate"} and support >= options.min_support:
                candidates.append({
                    "source": source,
                    "target": target,
                    "relation": f"{source} -> {target}",
                    "classification": relation["classification"],
                    "violation_rate": relation["violation_rate"],
                    "support": support,
                    "temporal_precedence": "unavailable_from_aggregated_label_columns",
                })
    return {
        "available": True,
        "labels": labels,
        "options": asdict(options),
        "splits": split_reports,
        "aggregate": aggregate,
        "candidate_dependencies": candidates,
        "interpretation": {
            "strict": "No sampled/full-row label violation for source=1,target=0, subject to support threshold.",
            "approximate": "Small but non-zero source=1,target=0 rate within configured threshold.",
            "unsupported": "The source-positive rows contain a substantial target-negative fraction.",
            "temporal_limit": "Label columns validate co-occurrence/implication only; event timestamps are required to establish order.",
        },
    }
