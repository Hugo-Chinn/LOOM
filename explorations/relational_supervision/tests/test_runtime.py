import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from explorations.relational_supervision.runtime import isolate_metadata, install_runtime


def test_isolated_metadata_preserves_data_and_hyperparameters(tmp_path):
    source = tmp_path / "source" / "D"
    source.mkdir(parents=True)
    (source / "meta_data.json").write_text(json.dumps({"label": ["a"]}))
    (source / "feature_map.json").write_text("historical")
    params = dict(dataset_id="D", model_id="full_S1", rebuild_dataset=False,
                  data_root=str(source.parent), train_data="/original/train/data",
                  seed=1, batch_size=8192)
    a = isolate_metadata(params, tmp_path / "run")
    b = isolate_metadata({**params, "model_id": "other_S1"}, tmp_path / "run")
    assert a["data_root"] != b["data_root"]
    assert {k: v for k, v in a.items() if k != "data_root"} == {
        k: v for k, v in params.items() if k != "data_root"}
    assert (source / "feature_map.json").read_text() == "historical"
    assert isolate_metadata(params, tmp_path / "run") == a
    (source / "meta_data.json").write_text("changed")
    with pytest.raises(ValueError, match="metadata changed"):
        isolate_metadata(params, tmp_path / "run")


def test_gpu_uuid_mask_survives_historical_launcher(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-uuid")
    monkeypatch.delenv("RELSUP_RUN_ROOT", raising=False)
    calls = []
    utils = SimpleNamespace()
    torch_utils = SimpleNamespace(setup_visible_devices=lambda ids: calls.append(ids))
    package = ModuleType("unirank")
    package.utils = utils
    pytorch = ModuleType("unirank.pytorch")
    pytorch.torch_utils = torch_utils
    monkeypatch.setitem(sys.modules, "unirank", package)
    monkeypatch.setitem(sys.modules, "unirank.pytorch", pytorch)
    install_runtime()
    torch_utils.setup_visible_devices([0])
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-test-uuid"
    assert not calls
    with pytest.raises(ValueError, match="one visible GPU"):
        torch_utils.setup_visible_devices([0, 1])
