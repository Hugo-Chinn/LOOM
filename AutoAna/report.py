"""Report serialization for AutoAna."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_json(report: Mapping[str, Any], path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def _number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    dataset = report["dataset"]
    lines = [
        f"# AutoAna report: {dataset['dataset_id']}",
        "",
        f"- Status: **{summary['status'].upper()}**",
        f"- Generated at: `{report['generated_at']}`",
        f"- Dataset root: `{dataset.get('root') or '-'}`",
        f"- Errors: {summary['error_count']}",
        f"- Warnings: {summary['warning_count']}",
        "",
        "## Split overview",
        "",
        "| Split | Data rows | Data blocks | Sampled rows | Paired blocks |",
        "|---|---:|---:|---:|:---:|",
    ]
    for split in ("train", "valid", "test"):
        split_report = report["splits"][split]
        data = split_report["tables"]["data"]
        lines.append(
            f"| {split} | {_number(data['row_count'])} | {_number(data['file_count'])} | "
            f"{_number(data['sampled_row_count'])} | {'yes' if split_report['block_pairing']['paired'] else 'no'} |"
        )

    lines.extend(["", "## Label distribution", ""])
    if report["labels"]:
        lines.extend([
            "| Label | Train positive rate | Valid positive rate | Test positive rate |",
            "|---|---:|---:|---:|",
        ])
        for label in report["labels"]:
            values = report["label_analysis"][label]
            rates = [values[split].get("positive_rate") for split in ("train", "valid", "test")]
            lines.append(f"| {label} | {_number(rates[0])} | {_number(rates[1])} | {_number(rates[2])} |")
    else:
        lines.append("No labels were configured or inferred.")

    lines.extend(["", "## Sequence profile", "", "| Split | Mean target prefix | Max target prefix | Mean full history | Max full history |", "|---|---:|---:|---:|---:|"])
    for split in ("train", "valid", "test"):
        item = report["sequence_analysis"][split]
        prefix = item.get("target_prefix_length") or {}
        history = item.get("full_item_sequence_length") or {}
        lines.append(
            f"| {split} | {_number(prefix.get('mean'))} | {_number(prefix.get('max'))} | "
            f"{_number(history.get('mean'))} | {_number(history.get('max'))} |"
        )

    dependencies = report.get("feedback_dependencies", {})
    lines.extend(["", "## Feedback dependencies", ""])
    if not dependencies.get("available"):
        lines.append("Feedback dependency analysis is unavailable because fewer than two feedback labels were found.")
    else:
        lines.extend([
            "Pairwise results use the full data rows. `A -> B` means checking the violation pattern `A=1, B=0`; timestamp order is not established by aggregated labels.",
            "",
            r"| Source | Target | A→B violation | P(B\|A) | Classification | Support(A) |",
            "|---|---|---:|---:|---|---:|",
        ])
        for source in dependencies["labels"]:
            for target in dependencies["labels"]:
                if source == target:
                    continue
                pair = dependencies["aggregate"][source][target]
                relation = pair["source_to_target"]
                lines.append(
                    f"| {source} | {target} | {_number(relation.get('violation_rate'))} | "
                    f"{_number(relation.get('conditional_probability'))} | {relation.get('classification')} | "
                    f"{_number(pair['support'].get('source'))} |"
                )
        lines.extend(["", "Candidate relations:"])
        candidates = dependencies.get("candidate_dependencies", [])
        if candidates:
            lines.extend(f"- `{item['relation']}` — {item['classification']}, violation rate {_number(item['violation_rate'])}, support {_number(item['support'])}." for item in candidates)
        else:
            lines.append("- No strict or approximate candidate dependency met the support threshold.")

    lines.extend(["", "## Findings", ""])
    if report["issues"]:
        for issue in report["issues"]:
            lines.append(f"- **{issue['severity'].upper()} · {issue['code']}** — {issue['message']}")
    else:
        lines.append("- No issue found in the analyzed metadata and samples.")

    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in report["recommendations"])
    lines.extend([
        "",
        "## Sampling note",
        "",
        "Row counts and file sizes come from Parquet metadata and are exact. Column statistics, label rates, and index ranges are sample-based according to `analysis_options` in the JSON report.",
        "",
    ])
    return "\n".join(lines)


def write_markdown(report: Mapping[str, Any], path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path
