"""Keep the supported package separate from archived research assets."""
from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]


def test_configuration_assets_are_centralized():
    assert (ROOT / "configs/portable/default.yaml").is_file()
    assert (ROOT / "configs/research/config/model_config.yaml").is_file()
    assert not list(ROOT.glob("config_*"))
    assert not (ROOT / "model_zoo").exists()


def test_portable_code_does_not_import_research_framework():
    for path in (ROOT / "loom").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name.split(".")[0] in {"model_zoo", "unirank", "research"}
                           for name in names), str(path)
