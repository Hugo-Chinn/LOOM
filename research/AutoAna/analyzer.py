"""Streaming, configuration-aware analysis for blocked UniRank datasets."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from .layout import DatasetLayout, FeatureSpec, SPLITS, TABLES
from .dependencies import DependencyOptions, analyze_feedback_dependencies
from .graph import build_conflict_graph


@dataclass(frozen=True)
class AnalysisOptions:
    sample_rows_per_file: int = 2000
    max_files_per_table: Optional[int] = None
    top_k: int = 10
    distinct_cap: int = 10000
    drift_warning_threshold: float = 0.10
    severe_imbalance_threshold: float = 0.01

    def __post_init__(self):
        if self.sample_rows_per_file <= 0:
            raise ValueError("sample_rows_per_file must be positive")
        if self.max_files_per_table is not None and self.max_files_per_table <= 0:
            raise ValueError("max_files_per_table must be positive or omitted")
        if self.top_k <= 0 or self.distinct_cap <= 0:
            raise ValueError("top_k and distinct_cap must be positive")


class ColumnAccumulator:
    def __init__(self, name: str, dtype: str, top_k: int, distinct_cap: int):
        self.name = name
        self.dtype = dtype
        self.top_k = top_k
        self.distinct_cap = distinct_cap
        self.sample_count = 0
        self.null_count = 0
        self.non_null_count = 0
        self.numeric_count = 0
        self.numeric_sum = 0.0
        self.numeric_sum_sq = 0.0
        self.numeric_min: Optional[float] = None
        self.numeric_max: Optional[float] = None
        self.list_count = 0
        self.list_length_sum = 0
        self.list_length_min: Optional[int] = None
        self.list_length_max: Optional[int] = None
        self.empty_list_count = 0
        self.list_element_count = 0
        self.list_element_sum = 0.0
        self.list_element_sum_sq = 0.0
        self.list_element_min: Optional[float] = None
        self.list_element_max: Optional[float] = None
        self.distinct: set = set()
        self.distinct_truncated = False
        self.top_values: Counter = Counter()

    def update(self, values: Iterable[Any]) -> None:
        for value in values:
            self.sample_count += 1
            if value is None:
                self.null_count += 1
                continue
            self.non_null_count += 1
            if isinstance(value, (list, tuple)):
                length = len(value)
                self.list_count += 1
                self.list_length_sum += length
                self.list_length_min = length if self.list_length_min is None else min(self.list_length_min, length)
                self.list_length_max = length if self.list_length_max is None else max(self.list_length_max, length)
                self.empty_list_count += int(length == 0)
                key = f"<length={length}>"
                for element in value:
                    if isinstance(element, (int, float, bool)) and not isinstance(element, complex):
                        number = float(element)
                        if math.isfinite(number):
                            self.list_element_count += 1
                            self.list_element_sum += number
                            self.list_element_sum_sq += number * number
                            self.list_element_min = number if self.list_element_min is None else min(self.list_element_min, number)
                            self.list_element_max = number if self.list_element_max is None else max(self.list_element_max, number)
            else:
                key = _hashable(value)
                if isinstance(value, (int, float, bool)) and not isinstance(value, complex):
                    number = float(value)
                    if math.isfinite(number):
                        self.numeric_count += 1
                        self.numeric_sum += number
                        self.numeric_sum_sq += number * number
                        self.numeric_min = number if self.numeric_min is None else min(self.numeric_min, number)
                        self.numeric_max = number if self.numeric_max is None else max(self.numeric_max, number)

            if len(self.distinct) < self.distinct_cap:
                self.distinct.add(key)
            elif key not in self.distinct:
                self.distinct_truncated = True
            self.top_values[key] += 1

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "dtype": self.dtype,
            "sample_count": self.sample_count,
            "non_null_count": self.non_null_count,
            "null_count": self.null_count,
            "null_rate": _ratio(self.null_count, self.sample_count),
            "distinct_count": len(self.distinct),
            "distinct_count_is_lower_bound": self.distinct_truncated,
            "top_values": [
                {"value": str(value), "count": count, "rate": _ratio(count, self.non_null_count)}
                for value, count in self.top_values.most_common(self.top_k)
            ],
        }
        if self.numeric_count:
            mean = self.numeric_sum / self.numeric_count
            variance = max(0.0, self.numeric_sum_sq / self.numeric_count - mean * mean)
            result["numeric"] = {
                "count": self.numeric_count,
                "min": self.numeric_min,
                "max": self.numeric_max,
                "mean": mean,
                "std": math.sqrt(variance),
            }
        if self.list_count:
            result["list_length"] = {
                "count": self.list_count,
                "min": self.list_length_min,
                "max": self.list_length_max,
                "mean": _ratio(self.list_length_sum, self.list_count),
                "empty_count": self.empty_list_count,
                "empty_rate": _ratio(self.empty_list_count, self.list_count),
            }
        if self.list_element_count:
            mean = self.list_element_sum / self.list_element_count
            variance = max(0.0, self.list_element_sum_sq / self.list_element_count - mean * mean)
            result["list_element_numeric"] = {
                "count": self.list_element_count,
                "min": self.list_element_min,
                "max": self.list_element_max,
                "mean": mean,
                "std": math.sqrt(variance),
            }
        return result


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _hashable(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return _display_value(value)


def _display_value(value: Any, limit: int = 100) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _parquet_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if not path.is_dir():
        return []
    return sorted(file for file in path.rglob("*.parquet") if file.is_file())


def _issue(issues: List[Dict[str, Any]], severity: str, code: str, message: str, **context: Any) -> None:
    item = {"severity": severity, "code": code, "message": message}
    if context:
        item["context"] = context
    issues.append(item)


def _expected_columns(layout: DatasetLayout, table: str) -> List[str]:
    active = [feature for feature in layout.features if feature.active]
    if table == "data":
        columns = [
            feature.name
            for feature in active
            if feature.source not in {"item", "action"}
        ]
        return list(dict.fromkeys(columns + layout.labels))
    if table == "item_info":
        return list(dict.fromkeys(["item_index"] + [f.name for f in active if f.source == "item"]))
    if table == "user_info":
        result = ["user_index", "full_item_seq"]
        if any(feature.source == "action" for feature in active):
            result.append("full_action_seq")
        return result
    return []


def _schema_signature(schema: pa.Schema) -> Dict[str, str]:
    return {field.name: str(field.type) for field in schema}


def _analyze_table(
    path: Path,
    expected_columns: Sequence[str],
    options: AnalysisOptions,
    issues: List[Dict[str, Any]],
    split: str,
    table: str,
) -> Dict[str, Any]:
    files = _parquet_files(path)
    summary: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "analyzed_file_count": 0,
        "total_bytes": sum(file.stat().st_size for file in files),
        "row_count": 0,
        "sampled_row_count": 0,
        "columns": {},
        "schema_variants": [],
        "files": [],
    }
    if not path.exists():
        _issue(issues, "error", "missing_table_path", f"Missing {split}/{table}: {path}", split=split, table=table)
        return summary
    if not files:
        _issue(issues, "error", "empty_table", f"No Parquet files in {split}/{table}", split=split, table=table)
        return summary

    signatures: MutableMapping[Tuple[Tuple[str, str], ...], List[str]] = {}
    file_metadata: List[Tuple[Path, pq.ParquetFile, Dict[str, str]]] = []
    for file in files:
        try:
            parquet = pq.ParquetFile(file)
            signature = _schema_signature(parquet.schema_arrow)
            rows = int(parquet.metadata.num_rows)
            summary["row_count"] += rows
            summary["files"].append({"path": str(file), "rows": rows, "bytes": file.stat().st_size})
            signatures.setdefault(tuple(signature.items()), []).append(file.name)
            file_metadata.append((file, parquet, signature))
        except Exception as exc:
            _issue(issues, "error", "unreadable_parquet", f"Cannot read Parquet metadata: {file}: {exc}", split=split, table=table)

    summary["schema_variants"] = [
        {"columns": dict(signature), "file_count": len(names), "example_files": names[:5]}
        for signature, names in signatures.items()
    ]
    if len(signatures) > 1:
        _issue(issues, "error", "schema_mismatch", f"{split}/{table} has {len(signatures)} schema variants", split=split, table=table)

    union_columns = set()
    for _, _, signature in file_metadata:
        union_columns.update(signature)
    missing = sorted(set(expected_columns) - union_columns)
    summary["expected_columns"] = list(expected_columns)
    summary["missing_expected_columns"] = missing
    if missing:
        _issue(issues, "error", "missing_columns", f"{split}/{table} is missing expected columns: {missing}", split=split, table=table, columns=missing)

    selected = file_metadata
    if options.max_files_per_table is not None:
        selected = selected[: options.max_files_per_table]
    accumulators: Dict[str, ColumnAccumulator] = {}
    for file, parquet, signature in selected:
        summary["analyzed_file_count"] += 1
        remaining = options.sample_rows_per_file
        try:
            batch_limit = 128 if table == "user_info" else 8192
            for batch in parquet.iter_batches(batch_size=min(batch_limit, remaining)):
                if remaining <= 0:
                    break
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
                payload = batch.to_pydict()
                for name, values in payload.items():
                    accumulator = accumulators.setdefault(
                        name,
                        ColumnAccumulator(name, signature.get(name, "unknown"), options.top_k, options.distinct_cap),
                    )
                    accumulator.update(values)
                summary["sampled_row_count"] += batch.num_rows
                remaining -= batch.num_rows
        except Exception as exc:
            _issue(issues, "error", "sample_read_failed", f"Cannot sample {file}: {exc}", split=split, table=table)

    summary["columns"] = {name: acc.to_dict() for name, acc in sorted(accumulators.items())}
    return summary


def _validate_file_pairing(
    layout: DatasetLayout,
    split: str,
    tables: Mapping[str, Mapping[str, Any]],
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    file_names = {
        table: {Path(item["path"]).name for item in tables[table].get("files", [])}
        for table in TABLES
    }
    data_names = file_names["data"]
    result = {"paired": True, "missing_by_table": {}}
    for table in ("user_info", "item_info"):
        missing = sorted(data_names - file_names[table])
        extra = sorted(file_names[table] - data_names)
        result["missing_by_table"][table] = missing
        result[f"extra_{table}_files"] = extra
        if missing:
            result["paired"] = False
            _issue(issues, "error", "unpaired_blocks", f"{split}/{table} is missing blocks paired with data: {missing[:10]}", split=split, table=table, missing_count=len(missing))
        if extra:
            _issue(issues, "warning", "extra_blocks", f"{split}/{table} has blocks without matching data: {extra[:10]}", split=split, table=table, extra_count=len(extra))
    return result


def _numeric_column(table: Mapping[str, Any], name: str) -> Optional[Mapping[str, Any]]:
    column = table.get("columns", {}).get(name)
    if not column:
        return None
    return column.get("numeric")


def _numeric_or_list_elements(table: Mapping[str, Any], name: str) -> Optional[Mapping[str, Any]]:
    column = table.get("columns", {}).get(name)
    if not column:
        return None
    return column.get("numeric") or column.get("list_element_numeric")


def _validate_index_ranges(split: str, tables: Mapping[str, Mapping[str, Any]], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    for index_name, side_table in (("user_index", "user_info"), ("item_index", "item_info")):
        numeric = _numeric_column(tables["data"], index_name)
        side_rows = int(tables[side_table].get("row_count", 0))
        if not numeric or not side_rows:
            checks[index_name] = {"checked": False}
            continue
        minimum = numeric.get("min")
        maximum = numeric.get("max")
        valid = minimum is not None and minimum >= 0 and maximum is not None and maximum < side_rows
        checks[index_name] = {
            "checked": True,
            "sample_min": minimum,
            "sample_max": maximum,
            "side_table_rows": side_rows,
            "valid_dense_index_range": valid,
        }
        if not valid:
            _issue(issues, "error", "index_out_of_range", f"Sampled {split}/{index_name} is outside [0, {side_rows})", split=split, column=index_name)

    history = _numeric_or_list_elements(tables["user_info"], "full_item_seq")
    item_rows = int(tables["item_info"].get("row_count", 0))
    if history and item_rows:
        minimum, maximum = history.get("min"), history.get("max")
        valid = minimum is not None and minimum >= 0 and maximum is not None and maximum < item_rows
        checks["full_item_seq"] = {
            "checked": True,
            "sample_min": minimum,
            "sample_max": maximum,
            "item_info_rows": item_rows,
            "valid_dense_index_range": valid,
        }
        if not valid:
            _issue(issues, "error", "history_index_out_of_range", f"Sampled {split}/full_item_seq contains values outside [0, {item_rows})", split=split, column="full_item_seq")
    else:
        checks["full_item_seq"] = {"checked": False}
    return checks


def _feature_value_checks(
    layout: DatasetLayout,
    splits: Mapping[str, Mapping[str, Any]],
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for split in SPLITS:
        split_checks: Dict[str, Any] = {}
        tables = splits[split]["tables"]
        for feature in layout.features:
            if not feature.active:
                continue
            if feature.source == "item":
                table_name, column_name = "item_info", feature.name
            elif feature.source == "action":
                table_name, column_name = "user_info", "full_action_seq"
            else:
                table_name, column_name = "data", feature.name
            profile = tables[table_name].get("columns", {}).get(column_name)
            if not profile:
                continue
            numeric = profile.get("numeric") or profile.get("list_element_numeric")
            check: Dict[str, Any] = {
                "table": table_name,
                "column": column_name,
                "null_rate": profile.get("null_rate", 0.0),
            }
            if profile.get("null_count", 0):
                _issue(issues, "warning", "null_feature_values", f"Feature {feature.name} has sampled null values in {split}/{table_name}", split=split, table=table_name, column=column_name, null_rate=profile.get("null_rate"))
            if numeric and feature.vocab_size is not None:
                minimum, maximum = numeric.get("min"), numeric.get("max")
                valid = minimum is not None and minimum >= 0 and maximum is not None and maximum < feature.vocab_size
                check.update({
                    "sample_min": minimum,
                    "sample_max": maximum,
                    "vocab_size": feature.vocab_size,
                    "within_vocab": valid,
                })
                if not valid:
                    _issue(issues, "error", "vocab_out_of_range", f"Feature {feature.name} has sampled values outside [0, {feature.vocab_size}) in {split}/{table_name}", split=split, table=table_name, column=column_name, vocab_size=feature.vocab_size)
            split_checks[feature.name] = check
        result[split] = split_checks
    return result


def _infer_labels(layout: DatasetLayout, train_data: Mapping[str, Any]) -> List[str]:
    if layout.labels:
        return list(layout.labels)
    inferred = []
    excluded = {"user_index", "item_index", "seq_len", "user_id", "item_id"}
    for name, profile in train_data.get("columns", {}).items():
        if name in excluded or "list_length" in profile:
            continue
        numeric = profile.get("numeric")
        if not numeric:
            continue
        if numeric.get("min") is not None and numeric["min"] >= 0 and numeric.get("max") is not None and numeric["max"] <= 1:
            lowered = name.lower()
            if lowered.startswith("is_") or any(token in lowered for token in ("click", "like", "cart", "buy", "follow", "conversion", "purchase", "checkout", "label")):
                inferred.append(name)
    return sorted(inferred)


def _label_analysis(
    labels: Sequence[str],
    splits: Mapping[str, Mapping[str, Any]],
    options: AnalysisOptions,
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for label in labels:
        per_split: Dict[str, Any] = {}
        train_rate: Optional[float] = None
        for split in SPLITS:
            profile = splits[split]["tables"]["data"].get("columns", {}).get(label)
            if not profile:
                per_split[split] = {"available": False}
                continue
            numeric = profile.get("numeric")
            if not numeric or numeric.get("count", 0) == 0:
                per_split[split] = {"available": True, "positive_rate": None}
                continue
            minimum, maximum = numeric.get("min"), numeric.get("max")
            is_binary = minimum is not None and minimum >= 0 and maximum is not None and maximum <= 1
            rate = float(numeric["mean"]) if is_binary else None
            per_split[split] = {
                "available": True,
                "is_binary": is_binary,
                "positive_rate": rate,
                "sample_count": numeric["count"],
                "min": minimum,
                "max": maximum,
            }
            if not is_binary:
                _issue(issues, "error", "non_binary_label", f"Label {label} in {split} contains values outside [0, 1]", split=split, column=label)
            if rate is not None and min(rate, 1.0 - rate) < options.severe_imbalance_threshold:
                _issue(issues, "warning", "severe_label_imbalance", f"Label {label} in {split} has sampled positive rate {rate:.6f}", split=split, column=label, positive_rate=rate)
            if split == "train":
                train_rate = rate
            elif train_rate is not None and rate is not None:
                delta = abs(rate - train_rate)
                per_split[split]["absolute_drift_from_train"] = delta
                if delta >= options.drift_warning_threshold:
                    _issue(issues, "warning", "label_drift", f"Label {label} positive rate shifts by {delta:.4f} from train to {split}", split=split, column=label, absolute_delta=delta)
        result[label] = per_split
    return result


def _sequence_analysis(splits: Mapping[str, Mapping[str, Any]], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for split in SPLITS:
        data_columns = splits[split]["tables"]["data"].get("columns", {})
        user_columns = splits[split]["tables"]["user_info"].get("columns", {})
        seq_len = data_columns.get("seq_len", {}).get("numeric")
        item_seq = user_columns.get("full_item_seq", {}).get("list_length")
        action_seq = user_columns.get("full_action_seq", {}).get("list_length")
        result[split] = {
            "target_prefix_length": seq_len,
            "full_item_sequence_length": item_seq,
            "full_action_sequence_length": action_seq,
        }
        if seq_len and seq_len.get("min") is not None and seq_len["min"] < 0:
            _issue(issues, "error", "negative_sequence_length", f"{split}/data has negative seq_len values", split=split)
        if item_seq and action_seq and item_seq.get("mean") != action_seq.get("mean"):
            _issue(issues, "warning", "sequence_length_mismatch", f"Sampled item/action sequence mean lengths differ in {split}", split=split)
    return result


def _feature_summary(layout: DatasetLayout) -> Dict[str, Any]:
    by_source = Counter(feature.source or "data" for feature in layout.features if feature.active)
    by_type = Counter(feature.type for feature in layout.features if feature.active)
    return {
        "configured_feature_count": sum(by_source.values()),
        "by_source": dict(sorted(by_source.items())),
        "by_type": dict(sorted(by_type.items())),
        "features": [asdict(feature) for feature in layout.features],
    }


def _recommendations(report: Mapping[str, Any]) -> List[str]:
    codes = Counter(issue["code"] for issue in report["issues"])
    recommendations = []
    if any(issue["severity"] == "error" for issue in report["issues"]):
        recommendations.append("Fix structural and schema errors before starting model training.")
    if codes["severe_label_imbalance"]:
        recommendations.append("Review loss weighting, sampling, and per-task AUC/PR-AUC for severely imbalanced labels.")
    if codes["label_drift"]:
        recommendations.append("Investigate temporal or preprocessing changes behind train/validation/test label drift.")
    if codes["schema_mismatch"]:
        recommendations.append("Rewrite inconsistent Parquet blocks with one canonical schema.")
    if codes["unpaired_blocks"]:
        recommendations.append("Regenerate blocked data so every data block has matching user_info and item_info blocks.")
    if not recommendations:
        recommendations.append("No blocking issue was found in the analyzed metadata and samples; proceed with a training smoke test.")
    return recommendations


def analyze_dataset(layout: DatasetLayout, options: Optional[AnalysisOptions] = None) -> Dict[str, Any]:
    options = options or AnalysisOptions()
    issues: List[Dict[str, Any]] = []
    split_reports: Dict[str, Any] = {}

    for split in SPLITS:
        tables: Dict[str, Any] = {}
        for table in TABLES:
            tables[table] = _analyze_table(
                layout.table_path(split, table),
                _expected_columns(layout, table),
                options,
                issues,
                split,
                table,
            )
        split_reports[split] = {
            "tables": tables,
            "block_pairing": _validate_file_pairing(layout, split, tables, issues),
            "index_ranges": _validate_index_ranges(split, tables, issues),
        }

    labels = _infer_labels(layout, split_reports["train"]["tables"]["data"])
    report: Dict[str, Any] = {
        "schema_version": 1,
        "module_version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_id": layout.dataset_id,
            "root": str(layout.root) if layout.root else None,
            "config_path": str(layout.config_path) if layout.config_path else None,
            "metadata": layout.metadata,
        },
        "analysis_options": asdict(options),
        "feature_summary": _feature_summary(layout),
        "labels": labels,
        "splits": split_reports,
        "label_analysis": _label_analysis(labels, split_reports, options, issues),
        "sequence_analysis": _sequence_analysis(split_reports, issues),
        "feature_value_checks": _feature_value_checks(layout, split_reports, issues),
        "feedback_dependencies": analyze_feedback_dependencies(
            layout,
            labels,
            DependencyOptions(min_support=100),
        ),
        "issues": issues,
    }
    severity_counts = Counter(issue["severity"] for issue in issues)
    report["summary"] = {
        "status": "error" if severity_counts["error"] else ("warning" if severity_counts["warning"] else "ok"),
        "error_count": severity_counts["error"],
        "warning_count": severity_counts["warning"],
        "total_rows": {
            split: split_reports[split]["tables"]["data"]["row_count"]
            for split in SPLITS
        },
        "total_data_rows": sum(split_reports[split]["tables"]["data"]["row_count"] for split in SPLITS),
    }
    report["recommendations"] = _recommendations(report)
    report["feedback_conflict_graph"] = build_conflict_graph(report["feedback_dependencies"])
    return report
