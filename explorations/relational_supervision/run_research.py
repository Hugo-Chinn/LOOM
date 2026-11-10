"""Launch the historical runner with RankMixerRelSup injected into model_zoo.

    CUDA_VISIBLE_DEVICES=0 python -m explorations.relational_supervision.run_research \
        --config configs/local/taobao-relsup --expid sib_margin_S20262027 --gpu 0

Arguments are those of `research/run_expid.py`, which is executed unmodified.
Run from the repository root.
"""
from pathlib import Path
import runpy
import sys

RESEARCH = Path(__file__).resolve().parents[2] / "research"


def main():
    sys.path.insert(0, str(RESEARCH))
    import model_zoo
    from .research_model import RankMixerRelSup
    model_zoo.RankMixerRelSup = RankMixerRelSup
    entry = str(RESEARCH / "run_expid.py")
    sys.argv[0] = entry
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
