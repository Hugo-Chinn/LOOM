import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location(
        "prepare_native", ROOT / "research/prepare_native_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relocated_full_preserves_model_configuration(tmp_path):
    module = builder()
    source = ROOT / "configs/research/config_rankmixer_structure_ablation_dag"
    original = yaml.safe_load((source / "model_config.yaml").read_text())
    data = yaml.safe_load((source / "dataset_config.yaml").read_text())["Taobao_Action"]
    root = tmp_path / "data"
    directory = root / "Taobao_Action"
    directory.mkdir(parents=True)
    meta = {"label": [c["name"] for c in data["label_col"]],
            "vocab_size": {c["name"]: c["vocab_size"] for c in data["feature_cols"]
                           if "vocab_size" in c}}
    (directory / "meta_data.json").write_text(json.dumps(meta))
    for split in ("train", "valid", "test"):
        for kind in ("data", "user_info", "item_info"):
            part = directory / split / kind
            part.mkdir(parents=True)
            (part / "part-00000.parquet").touch()
    output = tmp_path / "config"
    ids = module.prepare("taobao", root, output, tmp_path / "models")
    config = yaml.safe_load((output / "model_config.yaml").read_text())
    for expid in ids:
        assert config[expid] == original[expid]
    assert {k: v for k, v in config["Base"].items() if k != "model_root"} == {
        k: v for k, v in original["Base"].items() if k != "model_root"}
    meta["vocab_size"]["user_id"] += 1
    (directory / "meta_data.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="vocabulary"):
        module.prepare("taobao", root, tmp_path / "bad-config", tmp_path / "models")
