"""Relocate an exact historical Full configuration without changing its model."""
import argparse
import copy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {
    "taobao": ("structure_ablation_dag", "DAGTrue", "Taobao_Action"),
    "merrec": ("structure_merrec", "Struct_MerRec_True", "MerRec_Action"),
    "kuairand": ("structure_kuai", "Struct_Kuai_True", "KuaiRand_Video_Action"),
    "qk": ("structure_qk", "Struct_QK_True", "QK_Video_Action"),
}
SEEDS = (20262027, 20262028, 20262029, 20262030)


def prepare(dataset, data_root, output, model_root):
    suffix, prefix, dataset_id = REFERENCES[dataset]
    source = ROOT / "configs/research" / f"config_rankmixer_{suffix}"
    models = yaml.safe_load((source / "model_config.yaml").read_text())
    data = yaml.safe_load((source / "dataset_config.yaml").read_text())[dataset_id]
    data = copy.deepcopy(data)
    directory = data_root.resolve() / dataset_id
    meta = json.loads((directory / "meta_data.json").read_text())
    labels = [column["name"] for column in data["label_col"]]
    if meta["label"] != labels:
        raise ValueError("prepared data task order differs from historical reference")
    for feature in data["feature_cols"]:
        name = feature["name"]
        if "vocab_size" in feature and meta["vocab_size"].get(name) != feature["vocab_size"]:
            raise ValueError(f"vocabulary differs for {name}; not the historical dataset")
    data["data_root"] = str(data_root.resolve())
    for split in ("train", "valid", "test"):
        for key, child in ((f"{split}_data", "data"),
                           (f"{split}_user_info", "user_info"),
                           (f"{split}_item_info", "item_info")):
            path = directory / split / child
            if not path.is_dir() or not any(path.glob("*.parquet")):
                raise ValueError(f"missing blocked parquet: {path}")
            data[key] = str(path)
    resolved = {"Base": copy.deepcopy(models["Base"])}
    resolved["Base"]["model_root"] = str(model_root.resolve())
    for seed in SEEDS:
        expid = f"{prefix}_S{seed}"
        resolved[expid] = copy.deepcopy(models[expid])
    if output.exists():
        raise ValueError("config output already exists; refusing overwrite")
    output.mkdir(parents=True)
    (output / "model_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    (output / "dataset_config.yaml").write_text(
        yaml.safe_dump({dataset_id: data}, sort_keys=False))
    (output / "provenance.json").write_text(json.dumps({
        "dataset": dataset_id, "source": str(source.relative_to(ROOT)),
        "seeds": SEEDS, "labels": labels,
        "changes": ["absolute dataset paths", "absolute checkpoint root"],
        "warning": "Vocabulary/schema checks are not proof of identical rows or splits.",
    }, indent=2))
    return [f"{prefix}_S{seed}" for seed in SEEDS]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(REFERENCES))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    for expid in prepare(args.dataset, args.data_root, args.output, args.model_root):
        print(expid)
