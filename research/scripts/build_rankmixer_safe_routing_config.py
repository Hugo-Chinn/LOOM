#!/usr/bin/env python3
"""Build zero-perturbation node/edge conditioning screens."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config_rankmixer_safe_routing"


def clone(value):
    return yaml.safe_load(yaml.safe_dump(value))


def main():
    taobao = yaml.safe_load(
        (ROOT / "config_rankmixer_branch_gate_ablation/model_config.yaml").read_text()
    )
    kuai = yaml.safe_load(
        (ROOT / "config_rankmixer_multiseed_confirmation/model_config.yaml").read_text()
    )
    config = {"Base": clone(kuai["Base"])}
    config["Base"].update({
        "model_root": "./runs_local/checkpoints_rankmixer_safe_routing/",
        "keep_checkpoint": False,
    })

    taobao_base = clone(taobao["RankMixer_Branch_Seed20262028"])
    taobao_dynamic = clone(taobao_base)
    taobao_dynamic["branch_parent_fusion"] = "dynamic"
    config["RankMixer_Taobao_DynamicBranch_S20262028"] = taobao_dynamic

    kuai_base = clone(kuai["RankMixer_Kuai_DAGMeanPre2_S20262028"])
    adapter = clone(kuai_base)
    adapter["child_adapter_dim"] = 16
    dynamic_softmax = clone(kuai_base)
    dynamic_softmax["dag_parent_fusion"] = "dynamic_softmax"
    dynamic_softmax_adapter = clone(dynamic_softmax)
    dynamic_softmax_adapter["child_adapter_dim"] = 16
    residual_bypass = clone(kuai_base)
    residual_bypass.update({
        "use_base_state_bypass": True,
        "base_bypass_mode": "residual",
    })
    combined = clone(residual_bypass)
    combined["child_adapter_dim"] = 16
    config["RankMixer_Kuai_AdapterOnly_S20262028"] = adapter
    config["RankMixer_Kuai_DynamicSoftmax_S20262028"] = dynamic_softmax
    config["RankMixer_Kuai_DynamicSoftmaxAdapter_S20262028"] = dynamic_softmax_adapter
    config["RankMixer_Kuai_ResidualBypass_S20262028"] = residual_bypass
    config["RankMixer_Kuai_AdapterResidualBypass_S20262028"] = combined

    local_data = yaml.safe_load((ROOT / "config_local/dataset_config.yaml").read_text())
    datasets = {
        dataset_id: local_data[dataset_id]
        for dataset_id in ("Taobao_Action", "KuaiRand_Video_Action")
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "dataset_config.yaml").write_text(yaml.safe_dump(datasets, sort_keys=False))
    print(OUT)


if __name__ == "__main__":
    main()
