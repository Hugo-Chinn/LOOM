"""Run-local metadata and single-visible-GPU handling for Native experiments."""
import copy
import os
from pathlib import Path
import shutil


def isolate_metadata(params, run_root):
    """Keep Parquet paths fixed; redirect historical feature-map writes per run."""
    if params.get("rebuild_dataset", True):
        raise ValueError("RelSup isolated runs require rebuild_dataset=false")
    result = copy.deepcopy(params)
    dataset = result["dataset_id"]
    expid = result["model_id"]
    for name in (dataset, expid):
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("dataset/model IDs must be single path components")
    source = Path(result["data_root"]) / dataset
    target_root = Path(run_root).resolve() / "metadata" / dataset / expid
    target = target_root / dataset
    target.mkdir(parents=True, exist_ok=True)
    meta = source / "meta_data.json"
    destination = target / meta.name
    if destination.exists():
        if destination.read_bytes() != meta.read_bytes():
            raise ValueError("metadata changed since this run was prepared")
    else:
        shutil.copy2(meta, destination)
    result["data_root"] = str(target_root)
    return result


def install_runtime():
    """Install launch-only wrappers, leaving historical source files unchanged."""
    from unirank import utils
    from unirank.pytorch import torch_utils

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    original_devices = torch_utils.setup_visible_devices

    def setup_devices(ids):
        if visible and visible != "-1":
            if "," in visible or ids != [0]:
                raise ValueError("RelSup needs one visible GPU and --gpu 0")
            # The historical helper overwrites UUID/physical masks with '0'.
            # Preserve the mask set before importing CUDA.
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        else:
            original_devices(ids)

    torch_utils.setup_visible_devices = setup_devices
    run_root = os.environ.get("RELSUP_RUN_ROOT")
    if run_root:
        original_load = utils.load_config

        def load_config(*args, **kwargs):
            return isolate_metadata(original_load(*args, **kwargs), run_root)

        utils.load_config = load_config
