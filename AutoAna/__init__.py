"""Automated data analysis for UniRank datasets."""

from .analyzer import AnalysisOptions, analyze_dataset
from .dependencies import DependencyOptions, analyze_feedback_dependencies
from .graph import build_conflict_graph, render_graph_markdown
from .layout import DatasetLayout, load_dataset_layout

__all__ = [
    "AnalysisOptions",
    "DependencyOptions",
    "DatasetLayout",
    "analyze_dataset",
    "analyze_feedback_dependencies",
    "build_conflict_graph",
    "render_graph_markdown",
    "load_dataset_layout",
]

__version__ = "0.1.0"
