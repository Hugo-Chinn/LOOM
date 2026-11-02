#!/usr/bin/env python3
"""Create a separate resume study with an actual immutable training source copy."""
import hashlib
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    runtime = ROOT / "runs_local/source_snapshots/loop_strategy_20261021"
    if runtime.exists():
        raise RuntimeError(f"Refuse to overwrite {runtime}")
    runtime.mkdir(parents=True)
    for folder in ("model_zoo", "unirank", "scripts"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            target = runtime / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    shutil.copy2(ROOT / "run_expid.py", runtime / "run_expid.py")
    (runtime / "data").symlink_to(ROOT / "data", target_is_directory=True)
    paths = sorted(runtime.rglob("*.py"))
    sources = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for backbone in ("rankmixer", "onetrans"):
        old = ROOT / f"config_{backbone}_loop_strategy_20261019"
        output = ROOT / f"config_{backbone}_loop_strategy_20261021"
        if output.exists():
            raise RuntimeError(f"Refuse to overwrite {output}")
        output.mkdir()
        manifest = json.loads((old / "manifest.json").read_text())
        models = yaml.safe_load((old / "model_config.yaml").read_text())
        models["Base"]["model_root"] = str(
            ROOT / f"runs_local/checkpoints_{backbone}_loop_strategy_20261021"
        ) + "/"
        arms = {}
        for expid, arm in manifest["arms"].items():
            new_id = expid.replace("Strategy_", "StrategyV2_", 1)
            models[new_id] = models.pop(expid)
            arms[new_id] = arm
        (output / "model_config.yaml").write_text(yaml.safe_dump(models, sort_keys=False))
        shutil.copy2(old / "dataset_config.yaml", output / "dataset_config.yaml")
        manifest.update(
            arms=arms, source_hashes=sources, runtime_root=str(runtime),
            config_sha256={name: digest(output / name)
                           for name in ("model_config.yaml", "dataset_config.yaml")},
            predecessor_snapshot=digest(old / "manifest.json"),
            resume_policy="Fresh complete matrix; no old result migration or source equivalence claim.",
        )
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(output.name, digest(output / "manifest.json")[:12], len(arms))
    print("Frozen source:", runtime)


if __name__ == "__main__":
    main()
