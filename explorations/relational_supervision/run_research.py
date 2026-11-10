"""Launch the historical runner with RankMixerRelSup injected into model_zoo.

    CUDA_VISIBLE_DEVICES=0 python -m explorations.relational_supervision.run_research \
        --config configs/local/taobao-relsup-p1 --expid sib_margin_S20262027 --gpu 0

Arguments are those of `research/run_expid.py`, which is executed unmodified.
Run from the repository root. Before training, <model_root>/<dataset>/
<expid>.env.json records commit, versions, GPU and the resolved configuration.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys

RESEARCH = Path(__file__).resolve().parents[2] / "research"


def record_environment():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--expid")
    known, _ = parser.parse_known_args()
    if not known.config or not known.expid or os.environ.get("RANK", "0") != "0":
        return
    from unirank.utils import load_config
    from .environment import write
    params = load_config(os.path.abspath(known.config), known.expid)
    resolved = {k: v for k, v in params.items() if k != "feature_cols"}
    root = Path(params.get("model_root", "./checkpoints")).expanduser()
    root = root if root.is_absolute() else RESEARCH / root  # run_expid chdirs there
    write(root / params["dataset_id"] / f"{known.expid}.env.json",
          {"expid": known.expid, "seed": params.get("seed"),
           "num_workers": params.get("num_workers"), "resolved_config": resolved})


def main():
    sys.path.insert(0, str(RESEARCH))
    from .runtime import install_runtime
    install_runtime()
    import model_zoo
    from .research_model import RankMixerRelSup
    model_zoo.RankMixerRelSup = RankMixerRelSup
    record_environment()
    entry = str(RESEARCH / "run_expid.py")
    sys.argv[0] = entry
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
