"""Historical configuration equality and formal-run reference gates."""
import importlib.util
from pathlib import Path
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location(
        "aligned_builder", ROOT / "scripts/build_rankmixer_mtl_aligned.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_historical_settings_and_seed_alignment():
    module = builder()
    with tempfile.TemporaryDirectory() as directory:
        config, data, manifest = module.build(Path(directory))
    assert manifest["seeds"] == [20262027, 20262028, 20262029, 20262030]
    for ds, (family, control_id, _, _) in module.REFERENCES.items():
        source = module.load(family)
        original = source[control_id]
        original_data = yaml.safe_load((ROOT / family / "dataset_config.yaml").read_text())
        assert data[ds] == original_data[ds]
        for seed in module.SEEDS:
            historical = config[f"History_Parallel_{ds}_S{seed}"]
            assert historical == {**original, "seed": seed}
            control = config[f"Head_ParallelIndependentTower_{ds}_S{seed}"]
            assert control["tokenizer_type"] == "Chunk"
            for key in module.BACKBONE_KEYS:
                if key in original:
                    assert control[key] == original[key]
            for variant in module.HEADS:
                arm = config[f"Head_{variant}_{ds}_S{seed}"]
                for key in module.BACKBONE_KEYS:
                    if key in original:
                        assert arm[key] == control[key]


def test_unverified_baselines_cannot_enter_ready_queue():
    module = builder()
    with tempfile.TemporaryDirectory() as directory:
        config, _, manifest = module.build(Path(directory))
        ready = (Path(directory) / "fixed_backbone_head_expids.txt").read_text().split()
        for variant in module.BLOCKED:
            assert not any(n.startswith(f"Head_{variant}_") for n in ready)
        assert not any(n.startswith("Head_SharedBottom_") for n in ready)
        assert len(ready) == 200
        assert sum(a["status"] == "ready" for a in manifest["arms"].values()) == 260
