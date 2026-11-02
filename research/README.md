# Historical Research Archive

Start with the [Native Full dataset index and Taobao reproduction guide](../docs/HISTORICAL_REPRODUCTION.md).
It includes a helper for relocating historical configs without changing model settings.

This directory preserves the original UniRank-based research code, separate
from the supported portable package in `../loom/`.

| Path | Purpose |
|---|---|
| `model_zoo/RankMixer.py` | Native RankMixer LOOM recurrence and historical extras |
| `model_zoo/OneTrans.py` | OneTrans experiments |
| `model_zoo/LoopMT.py` | Head-LoopMT, a different post-backbone architecture |
| `unirank/` | Historical training, feature processing and loading framework |
| `data/` | Dataset-specific preprocessing code, not datasets |
| `scripts/` | Exploratory training, analysis and experiment-generation tools |
| `tests/` | Historical tests, not the portable CI suite |
| `AutoAna/` | Original model analysis utility |
| `README_UNIRANK.md` | Upstream documentation and attribution |

## Historical Training Entry

From the repository root, after preparing the original data and installing
`research/requirements.txt` in a suitable environment:

```bash
python research/run_expid.py \
  --config configs/research/config_rankmixer_structure_merrec \
  --expid Struct_MerRec_True_S20262027 --gpu 0
```

The entry resolves the config relative to your invocation directory, then uses
`research/` as its working directory. Relative data/checkpoint paths in YAML are
therefore relative to `research/`; absolute machine-local paths need adjustment.

The archive is not a second supported portable API. Exploratory builders,
schedulers, old shell commands and upstream documentation may refer to the
original layout or excluded local artifacts. These are retained for traceability,
not advertised as turnkey commands. Update those paths before using an archived
tool. The root README describes the supported fresh-dataset workflow.
