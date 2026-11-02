"""OneTrans task matrix follows the native baseline and RankMixer seed set."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_onetrans_config_alignment():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "one_builder", ROOT / "scripts/build_onetrans_mtl_aligned.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as directory:
        config, data, manifest = module.build(Path(directory))
    source = yaml.safe_load(
        (ROOT / "config_onetrans_all_datasets_multiseed/model_config.yaml").read_text())
    assert manifest["seeds"] == [20262027, 20262028, 20262029, 20262030]
    assert len(manifest["arms"]) == 340
    assert sum(a["status"] == "ready" for a in manifest["arms"].values()) == 260
    for ds in data:
        baseline = source[f"OneTransAll_{ds}_Baseline_S20262027"]
        for seed in manifest["seeds"]:
            assert config[f"History_Parallel_{ds}_S{seed}"] == {**baseline, "seed": seed}
            for variant in module.HEADS:
                arm = config[f"Head_{variant}_{ds}_S{seed}"]
                assert arm["bottom_type"] == "onetrans"
                if variant == "PLE":
                    assert arm["expert_hidden_units"][-1] == (
                        baseline["token_dim"] * baseline["num_ns_token"])
                for key in ("tokenizer_type", "token_dim", "num_ns_token", "num_heads",
                            "num_layers", "reduction_ratio", "batch_size", "max_len",
                            "epochs", "embedding_dim", "tower_hidden_units"):
                    assert arm[key] == baseline[key]
