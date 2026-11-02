#!/usr/bin/env python3
"""Prespecified capacity/weight/bypass factorial studies on historical native loops."""
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20262027, 20262028, 20262029, 20262030]
TASKS = {
    "TencentGR_10M_Action": ["is_conversion"],
    "QK_Video_Action": ["follow", "share"],
    "KuaiRand_Video_Action": ["is_follow", "is_comment", "is_forward"],
    "MerRec_Action": ["Offer", "Checkout", "Purchase"],
    "Taobao_Action": ["fav", "buy"],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for backbone in ("rankmixer", "onetrans"):
        source = ROOT / f"config_{backbone}_mtl_aligned"
        old = yaml.safe_load((source / "model_config.yaml").read_text())
        data = yaml.safe_load((source / "dataset_config.yaml").read_text())
        output = ROOT / f"config_{backbone}_loop_strategy_20261019"
        if output.exists():
            raise RuntimeError(f"Refuse to overwrite frozen extension: {output}")
        output.mkdir()
        base = copy.deepcopy(old["Base"])
        base["model_root"] = f"./runs_local/checkpoints_{backbone}_loop_strategy_20261019/"
        models, arms = {"Base": base}, {}
        for ds, targets in TASKS.items():
            labels = [x["name"] for x in data[ds]["label_col"]]
            if not set(targets) <= set(labels):
                raise ValueError(f"Wrong target labels: {ds}")
            if backbone == "rankmixer":
                template = copy.deepcopy(old[f"History_Finalist_{ds}_S20262027"])
                if ds == "Taobao_Action":
                    template["use_loop_gate"] = False
                template["expansion_factor"] = 2
                parallel = copy.deepcopy(old[f"History_Parallel_{ds}_S20262027"])
            else:
                template = copy.deepcopy(old[f"History_FiLM_{ds}_S20262027"])
                template.update(model="OneTransCapacity", expansion_factor=4,
                                refiner_expansion_factor=4,
                                supervision_mode="deep_progressive",
                                auxiliary_loss_weight=0.5)
                parallel = copy.deepcopy(old[f"History_Parallel_{ds}_S20262027"])
            for seed in SEEDS:
                for variant in ("Parallel", "LoopControl", "Wide", "Weight", "Bypass",
                                "WideWeight", "WideBypass", "WeightBypass", "WideWeightBypass"):
                    params = copy.deepcopy(parallel if variant == "Parallel" else template)
                    params["seed"] = seed
                    if "Wide" in variant:
                        params["expansion_factor" if backbone == "rankmixer"
                               else "refiner_expansion_factor"] = 4 if backbone == "rankmixer" else 8
                    if "Weight" in variant:
                        coefficient = 4.0 if ds == "KuaiRand_Video_Action" else 2.0
                        params["task_loss_weights"] = [
                            coefficient if label in targets else 1.0 for label in labels
                        ]
                    if "Bypass" in variant:
                        if backbone == "rankmixer":
                            params.update(use_base_state_bypass=True, base_bypass_mode="residual",
                                          base_bypass_tasks=targets)
                        else:
                            params["capacity_bypass_tasks"] = targets
                    expid = f"Strategy_{backbone}_{variant}_{ds}_S{seed}"
                    models[expid] = params
                    arms[expid] = {
                        "status": "ready", "dataset": ds, "seed": seed,
                        "variant": variant, "study": "native_loop_strategy",
                    }
        for filename, payload in [("model_config.yaml", models), ("dataset_config.yaml", data)]:
            (output / filename).write_text(yaml.safe_dump(payload, sort_keys=False))
        paths = list((ROOT / "model_zoo").glob("*.py"))
        paths += [ROOT / "run_expid.py", ROOT / "scripts/build_loop_strategy_extension.py"]
        paths += list((ROOT / "unirank").rglob("*.py"))
        manifest = {
            "backbone": backbone, "seeds": SEEDS, "arms": arms,
            "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)},
            "config_sha256": {f: digest(output / f)
                              for f in ("model_config.yaml", "dataset_config.yaml")},
            "notes": [
                "New study; no legacy metric adoption.",
                "RankMixer original interaction-stage Loop; OneTrans native FiLM base_refiner.",
                "OneTrans full backbone expansion stays 4; only refiner widens from 4 to 8.",
                "Target labels prespecified; weighting coefficients heuristic, not measured rarity.",
                "Full 2x2x2 factorial plus parallel, all five datasets, canonical four seeds.",
            ],
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{output.name}: {len(arms)} runs, snapshot {digest(output / 'manifest.json')[:12]}")


if __name__ == "__main__":
    main()
