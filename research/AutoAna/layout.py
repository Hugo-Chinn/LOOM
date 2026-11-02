"""Dataset discovery and UniRank configuration parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


SPLITS = ("train", "valid", "test")
TABLES = ("data", "user_info", "item_info")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    type: str = "unknown"
    source: str = ""
    dtype: str = ""
    active: bool = True
    vocab_size: Optional[int] = None
    max_len: Optional[int] = None


@dataclass
class DatasetLayout:
    dataset_id: str
    root: Optional[Path]
    paths: Dict[str, Dict[str, Path]]
    features: List[FeatureSpec] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    config_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def table_path(self, split: str, table: str) -> Path:
        return self.paths[split][table]


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _resolve_config_file(config: Path) -> Path:
    config = config.expanduser().resolve()
    if config.is_dir():
        config = config / "dataset_config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"Dataset config not found: {config}")
    return config


def _resolve_data_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (config_path.parent / path).resolve()


def _feature_specs(raw: Iterable[Mapping[str, Any]]) -> List[FeatureSpec]:
    result: List[FeatureSpec] = []
    for item in raw or []:
        if not isinstance(item, Mapping) or "name" not in item:
            continue
        names = item["name"] if isinstance(item["name"], list) else [item["name"]]
        for name in names:
            result.append(
                FeatureSpec(
                    name=str(name),
                    type=str(item.get("type", "unknown")),
                    source=str(item.get("source", "") or ""),
                    dtype=str(item.get("dtype", "") or ""),
                    active=bool(item.get("active", True)),
                    vocab_size=_optional_int(item.get("vocab_size")),
                    max_len=_optional_int(item.get("max_len")),
                )
            )
    return result


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _labels(raw: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in raw or []:
        if isinstance(item, Mapping) and "name" in item:
            result.append(str(item["name"]))
        elif isinstance(item, str):
            result.append(item)
    return result


def layout_from_config(config: Path, dataset_id: str) -> DatasetLayout:
    config_path = _resolve_config_file(config)
    payload = _read_yaml(config_path)
    if dataset_id not in payload:
        choices = ", ".join(sorted(str(key) for key in payload))
        raise KeyError(f"Dataset '{dataset_id}' is not in {config_path}. Available: {choices}")
    spec = payload[dataset_id]
    if not isinstance(spec, Mapping):
        raise ValueError(f"Dataset configuration must be a mapping: {dataset_id}")

    paths: Dict[str, Dict[str, Path]] = {}
    for split in SPLITS:
        paths[split] = {}
        for table in TABLES:
            key = f"{split}_{table}"
            raw_path = spec.get(key)
            if raw_path:
                paths[split][table] = _resolve_data_path(str(raw_path), config_path)
            else:
                data_root = spec.get("data_root", ".")
                root = _resolve_data_path(str(data_root), config_path) / dataset_id
                paths[split][table] = root / split / table

    roots = [path.parents[1] for split in paths.values() for path in split.values()]
    common_root = roots[0] if roots and all(path == roots[0] for path in roots) else None
    return DatasetLayout(
        dataset_id=dataset_id,
        root=common_root,
        paths=paths,
        features=_feature_specs(spec.get("feature_cols", [])),
        labels=_labels(spec.get("label_col", [])),
        config_path=config_path,
        metadata={"blocked": spec.get("blocked"), "data_format": spec.get("data_format")},
    )


def layout_from_directory(dataset_dir: Path, dataset_id: Optional[str] = None) -> DatasetLayout:
    root = dataset_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    paths = {
        split: {table: root / split / table for table in TABLES}
        for split in SPLITS
    }
    return DatasetLayout(dataset_id=dataset_id or root.name, root=root, paths=paths)


def load_dataset_layout(
    *,
    dataset_dir: Optional[Path] = None,
    config: Optional[Path] = None,
    dataset_id: Optional[str] = None,
) -> DatasetLayout:
    if config is not None:
        if not dataset_id:
            raise ValueError("--dataset-id is required when --config is used")
        layout = layout_from_config(config, dataset_id)
        if dataset_dir is not None:
            root = dataset_dir.expanduser().resolve()
            layout.root = root
            layout.paths = {
                split: {table: root / split / table for table in TABLES}
                for split in SPLITS
            }
        return layout
    if dataset_dir is None:
        raise ValueError("Provide --dataset-dir, or provide --config together with --dataset-id")
    return layout_from_directory(dataset_dir, dataset_id)


def discover_datasets(root: Path) -> List[Path]:
    """Find directories that look like UniRank blocked datasets."""
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Discovery root not found: {root}")
    candidates = []
    for path in [root, *root.rglob("*")]:
        if not path.is_dir():
            continue
        if all((path / split / "data").is_dir() for split in SPLITS):
            candidates.append(path)
    return sorted(set(candidates))
