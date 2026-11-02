import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from loom.cli import load_split


def test_train_predict_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[1]
    def run(*args):
        subprocess.run([sys.executable, "-m", "loom.cli", *map(str, args)],
                       check=True, cwd=root, capture_output=True, text=True)
    data, output = tmp_path / "data", tmp_path / "run"
    run("demo-data", "--output", data)
    run("train", "--config", root / "configs/portable/default.yaml",
        "--data", data, "--output", output)
    result = json.loads((output / "metrics.json").read_text())
    assert set(result["test"]) == {"click", "cart", "fav", "buy"}
    assert all(0 <= m["auc"] <= 1 for m in result["test"].values())
    predictions = tmp_path / "predictions.npz"
    run("predict", "--checkpoint", output / "best.pt", "--input",
        data / "test.npz", "--output", predictions)
    with np.load(predictions) as saved:
        assert saved["probabilities"].shape == (256, 4)
        assert saved["tasks"].tolist() == ["click", "cart", "fav", "buy"]


def test_dataset_task_order_rejected(tmp_path):
    path = tmp_path / "train.npz"
    np.savez(path, x=np.ones((4, 2)), y=np.zeros((4, 2)),
             tasks=np.array(["buy", "click"]))
    with pytest.raises(ValueError, match="task order"):
        load_split(path, ["click", "buy"])
