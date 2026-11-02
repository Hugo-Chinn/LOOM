import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_cuda_runtime_and_backward():
    assert torch.__version__ == "2.6.0+cu124"
    assert torch.cuda.is_available()
    x = torch.randn(32, 32, device="cuda", requires_grad=True)
    x.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_repository_commit_is_pinned():
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == "90bb99a60724e975ec565836266fd86dcad0039d"


def test_synthetic_end_to_end_log_exists_and_completed():
    log = ROOT / "runs_local/checkpoints/TencentGR_10M_Action_Synthetic/RankMixer_TencentGR_Synthetic_Smoke.log"
    text = log.read_text()
    assert "Training finished." in text
    assert "******** Test evaluation ********" in text
    assert "[Task: is_click][Metrics]" in text
    assert "[Task: is_conversion][Metrics]" in text


def test_taac_merge_manifest():
    manifest = json.loads(
        (ROOT / "data/TAAC2025/TencentGR-10M-merged/merge_manifest.json").read_text()
    )
    assert manifest["seq"]["rows"] == 10_139_575
    assert manifest["user_feat"]["rows"] == 10_139_388
    assert manifest["item_feat"]["rows"] == 17_487_211
    assert manifest["seq"]["sha256"] == "c324bc89dd55fe3725694c82ad669816125c45e57524aa20df2a0d32c8c95d11"


def test_downloaded_preprocessed_dataset_audits():
    report_root = ROOT.parent / "reports"
    for name in ("taobao", "kuairand", "merrec", "qk"):
        report = json.loads((report_root / f"unirank_{name}_audit.json").read_text())
        assert report["complete"]
        assert not report["errors"]
        assert all(x["paired"] for x in report["splits"].values())


def test_other_dataset_real_batch_validations():
    report_root = ROOT.parent / "reports"
    for name in ("taobao", "kuairand", "merrec", "qk"):
        report = json.loads((report_root / f"unirank_{name}_real_batch_validation.json").read_text())
        assert report["finite_loss"]
        assert report["finite_gradients"]
        assert report["batch_size"] > 0
        assert set(report["tasks"]) == {key.removesuffix("_pred") for key in report["output_shapes"]}
