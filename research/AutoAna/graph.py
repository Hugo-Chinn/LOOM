"""Feedback conflict graph generation.

Each feedback is a node. A directed edge ``A -> B`` carries
``P(B=0 | A=1)``, named ``conflict_rate`` in the serialized graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional


def build_conflict_graph(dependency_report: Mapping[str, Any]) -> dict:
    labels = list(dependency_report.get("labels", []))
    nodes = [{"id": label, "label": label} for label in labels]
    edges = []
    aggregate = dependency_report.get("aggregate", {})
    for source in labels:
        for target in labels:
            if source == target:
                continue
            pair = aggregate.get(source, {}).get(target)
            if not pair:
                continue
            relation = pair.get("source_to_target", {})
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "conflict_rate": relation.get("violation_rate"),
                    "conditional_probability": relation.get("conditional_probability"),
                    "classification": relation.get("classification"),
                    "support": pair.get("support", {}).get("source"),
                    "counts": pair.get("counts", {}),
                }
            )
    return {
        "schema_version": 1,
        "graph_type": "directed_feedback_conflict_graph",
        "edge_semantics": "conflict_rate(A -> B) = P(B=0 | A=1)",
        "dataset": dependency_report.get("dataset", {}),
        "nodes": nodes,
        "edges": edges,
    }


def _rate(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.4f}"


def render_mermaid(graph: Mapping[str, Any]) -> str:
    lines = [
        "```mermaid",
        "graph LR",
    ]
    node_ids = {}
    for index, node in enumerate(graph.get("nodes", [])):
        node_id = f"n{index}"
        node_ids[node["id"]] = node_id
        label = str(node["label"]).replace('"', "'")
        lines.append(f'    {node_id}["{label}"]')
    for edge in graph.get("edges", []):
        source = node_ids.get(edge["source"])
        target = node_ids.get(edge["target"])
        if source is None or target is None:
            continue
        label = _rate(edge.get("conflict_rate"))
        lines.append(f"    {source} -->|{label}| {target}")
    lines.extend([
        "",
        "    %% Edge label = conflict_rate(A -> B) = P(B=0 | A=1)",
        "```",
    ])
    return "\n".join(lines)


def render_graph_markdown(graph: Mapping[str, Any]) -> str:
    dataset = graph.get("dataset", {})
    lines = [
        f"# Feedback conflict graph: {dataset.get('dataset_id', 'dataset')}",
        "",
        "Each feedback is a node. A directed edge `A → B` is labeled with `P(B=0 | A=1)`, the conflict rate when A is observed.",
        "",
        render_mermaid(graph),
        "",
        "## Edges",
        "",
        r"| Source | Target | Conflict rate | P(Target\|Source) | Classification | Support(source) |",
        "|---|---|---:|---:|---|---:|",
    ]
    for edge in graph.get("edges", []):
        lines.append(
            f"| {edge['source']} | {edge['target']} | {_rate(edge.get('conflict_rate'))} | "
            f"{_rate(edge.get('conditional_probability'))} | {edge.get('classification', 'NA')} | "
            f"{edge.get('support', 'NA'):,} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_graph(graph: Mapping[str, Any], output_dir: Path) -> dict:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "feedback_conflict_graph.json"
    markdown_path = output_dir / "feedback_conflict_graph.md"
    json_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_graph_markdown(graph), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
