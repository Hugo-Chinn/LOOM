"""Protocol tooling tests; no PyTorch needed.

    python -m pytest explorations/relational_supervision/tests/test_protocol.py -q
Log files written here are parser fixtures, not experimental results.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from explorations.relational_supervision import arms as A
from explorations.relational_supervision.make_research_configs import build
from explorations.relational_supervision.structure import relations

ROOT = Path(__file__).resolve().parents[3]
TASKS = ["is_click", "cart", "fav", "buy"]


def test_grid_names_and_targets():
    assert set(A.phase1()) == {"full", *A.TERMS}
    grid = A.phase2(["sib_margin", "sib_decor"])
    assert len(grid) == 8 + 2 and "sib_margin" not in grid
    assert grid["sib_margin_w0.03_m0.5"]["overrides"] == {
        "sibling_margin_weight": 0.03, "sibling_margin": 0.5}
    assert grid["sib_decor_w0.3"]["overrides"] == {"sibling_decor_weight": 0.3}
    rel = relations([[], [0], [0], [1, 2]])
    assert A.target_tasks(["sib_margin"], rel, TASKS) == ["cart", "fav"]
    assert A.target_tasks(["edge_gain"], rel, TASKS) == ["cart", "fav", "buy"]
    combo = A.frozen_arms({"sib_margin": {"weight": 0.03, "margin": 0.5},
                           "edge_distill": {"weight": 0.3, "margin": None}}, combine=True)
    assert combo["combo_edge_distill+sib_margin"]["overrides"] == {
        "sibling_margin_weight": 0.03, "sibling_margin": 0.5, "edge_distill_weight": 0.3}
    with pytest.raises(ValueError):
        A.frozen_arms({"sib_decor": {"weight": 0.1}}, combine=True)


@pytest.fixture
def source(tmp_path):
    family = ROOT / "configs/research/config_rankmixer_structure_ablation_dag"
    models = yaml.safe_load((family / "model_config.yaml").read_text())
    kept = {"Base": models["Base"],
            **{k: v for k, v in models.items() if k.startswith("DAGTrue_S")}}
    directory = tmp_path / "full"
    directory.mkdir()
    (directory / "model_config.yaml").write_text(yaml.safe_dump(kept, sort_keys=False))
    (directory / "dataset_config.yaml").write_text(
        (family / "dataset_config.yaml").read_text())
    return directory


def fake_log(path, validation_auc, logloss=0.2, test=True):
    lines = ["Total number of parameters: 123.", "Evaluation @epoch 1 - batch 10: "]
    lines += [f"[Task: {t}][Metrics] AUC: {validation_auc[t]:.6f} - logloss: {logloss:.6f}"
              for t in TASKS]
    lines += ["Save best model: monitor(max)=1.0", "[RelSup] step=200 sibling_gap=0.010000"]
    if test:
        lines += ["******** Test evaluation ********"]
        lines += [f"[Task: {t}][Metrics] logloss: 0.300000 - AUC: 0.900000" for t in TASKS]
    path.write_text("\n".join(lines) + "\n")


def collect(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "explorations.relational_supervision.collect_research",
         *map(str, arguments)], cwd=ROOT, capture_output=True, text=True)


def test_phases_end_to_end(source, tmp_path):
    root = tmp_path / "out"
    p1 = tmp_path / "p1"
    expids = build(source, p1, root, "1")
    assert len(expids) == 20
    models = yaml.safe_load((p1 / "model_config.yaml").read_text())
    full, arm = models["full_S20262028"], models["sib_margin_S20262028"]
    assert {k: v for k, v in arm.items() if full.get(k) != v} == {
        "relsup_sibling_margin_weight": 0.1, "relsup_sibling_margin": 0.2}
    protocol = json.loads((p1 / "protocol.json").read_text())
    assert protocol["arms"]["sib_margin"]["targets"] == ["cart", "fav"]
    with pytest.raises(ValueError, match="budget"):
        build(source, tmp_path / "x", root, "1", max_runs=5)
    with pytest.raises(ValueError, match="requires --selection"):
        build(source, tmp_path / "x", root, "2")

    logs = root / "Taobao_Action"
    logs.mkdir(parents=True)
    base = dict.fromkeys(TASKS, 0.7)
    for seed in A.DEV_SEEDS:
        fake_log(logs / f"full_S{seed}.log", base)
        fake_log(logs / f"sib_margin_S{seed}.log", {**base, "cart": 0.71, "fav": 0.71})
        fake_log(logs / f"sib_decor_S{seed}.log", {**base, "cart": 0.71, "fav": 0.69})
        fake_log(logs / f"edge_gain_S{seed}.log",
                 {**base, "cart": 0.72, "fav": 0.72, "buy": 0.72}, logloss=0.21)
    fake_log(logs / f"edge_distill_S{A.DEV_SEEDS[0]}.log", base, test=False)

    assert "refused" in collect("--stage", "test", "--config", p1,
                                "--output", tmp_path / "t0").stderr
    done = collect("--stage", "phase1", "--config", p1, "--output", tmp_path / "s1")
    assert done.returncode == 0, done.stderr
    selection = json.loads((tmp_path / "s1/selection.json").read_text())
    assert selection["survivors"] == ["sib_margin"]
    reasons = {arm: d["reason"] for arm, d in selection["decisions"].items()}
    assert reasons["sib_decor"] == "target tasks not positive in enough seeds"
    assert reasons["edge_gain"] == "validation logloss tolerance violated"
    assert "incomplete" in reasons["edge_distill"]
    assert not (tmp_path / "s1/test_summary.csv").exists()
    attempts = json.loads((tmp_path / "s1/attempts.json").read_text())
    assert attempts[f"edge_distill_S{A.DEV_SEEDS[0]}"]["status"] == "incomplete"
    assert attempts[f"edge_distill_S{A.DEV_SEEDS[1]}"]["status"] == "missing"

    p2 = tmp_path / "p2"
    assert len(build(source, p2, root, "2", selection, max_runs=40)) == 32
    refused = collect("--stage", "phase2", "--config", p1, p2, "--output", tmp_path / "s2a")
    assert "grid of sib_margin incomplete" in refused.stderr
    for name in A.phase2(["sib_margin"]):
        gain = 0.02 if name == "sib_margin_w0.03_m0.5" else 0.005
        for seed in A.DEV_SEEDS:
            fake_log(logs / f"{name}_S{seed}.log",
                     {**base, "cart": 0.7 + gain, "fav": 0.7 + gain})
    done = collect("--stage", "phase2", "--config", p1, p2, "--output", tmp_path / "s2")
    assert done.returncode == 0, done.stderr
    frozen = json.loads((tmp_path / "s2/selection.json").read_text())
    assert frozen["frozen"]["sib_margin"]["arm"] == "sib_margin_w0.03_m0.5"

    with pytest.raises(ValueError, match="at least two"):
        build(source, tmp_path / "p3", root, "3", frozen)
    confirm = tmp_path / "pc"
    names = build(source, confirm, root, "confirm", frozen)
    assert names[0] == f"full_S{A.CONFIRM_SEEDS[0]}" and len(names) == 8
    confirmed = yaml.safe_load((confirm / "model_config.yaml").read_text())
    assert confirmed[names[0]]["seed"] == A.CONFIRM_SEEDS[0]
    done = collect("--stage", "test", "--config", p1, p2, confirm, "--selection",
                   tmp_path / "s2/selection.json", "--output", tmp_path / "t1")
    assert done.returncode == 0, done.stderr
    table = (tmp_path / "t1/test_summary.csv").read_text()
    assert "exploratory" in table and "sib_decor" in table
