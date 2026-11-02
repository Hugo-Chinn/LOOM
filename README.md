# LOOM: Behavior-Conditioned Recurrent Multi-Task Modeling

LOOM refines shared token states along a behavior graph using a shared Mixer,
behavior-conditioned updates, progressive task readouts and optional auxiliary
supervision. The portable PyTorch core accepts tokens from your own backbone.
The historical UniRank research implementation is retained separately.

## From Zero To Training And Inference

Python 3.10+ is required. This quick start works on CPU:

```bash
git clone https://github.com/Hugo-Chinn/LOOM.git
cd LOOM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"

loom demo-data --output runs_local/demo-data
loom train --config examples/portable.yaml --data runs_local/demo-data \
  --output runs_local/demo/S20262027 --seed 20262027
loom predict --checkpoint runs_local/demo/S20262027/best.pt \
  --input runs_local/demo-data/test.npz --output runs_local/demo/predictions.npz
python -m pytest tests_portable -q
```

For GPU training, install an appropriate PyTorch CUDA build and pass
`--device cuda:0`. Synthetic demo results check the pipeline only, not superiority.
Each run saves `best.pt`, `resolved_config.json`, and `metrics.json` containing
**AUC/logloss per task**, observed label counts, seed and selected epoch.

## New Dataset And Backbone

Use NPZ arrays `x [N,F]`, `y [N,K]`, `tasks [K]` for the numeric reference runner,
or attach `loom.LOOM` to an encoder returning `[B,T,D]` tokens. Dataset-specific
encoding and label semantics remain your responsibility. The numeric encoder is
not the historical RankMixer feature/sequence tokenizer.

- [New dataset, custom backbone and inference](docs/NEW_DATASET.md)
- [Core equations and compatibility boundary](docs/METHOD.md)
- [Configuration schema and bounded search space](docs/CONFIGURATION.md)
- [Coding-agent instructions](AGENTS.md)

## Four-Seed Component Exploration

```bash
python -m loom.sweep --config examples/portable.yaml --data runs_local/demo-data \
  --output runs_local/demo-ablation \
  --variants full no_bcu no_mas no_cbm untied \
  --seeds 20262027 20262028 20262029 20262030 --max-runs 20
```

Selection uses validation only; each seed's per-task test metrics are retained.
The sweep also exports `per_seed_task.csv` and `per_task_summary.csv`, with
sample standard deviations and paired deltas against Full over shared seeds.
Keep backbone, data and training fixed. No-CBM routes the first parent at merge
nodes; it is N/A without merges. Untied updates increase parameter capacity.

GitHub Actions tests the API and end-to-end CPU workflow. Manual `Portable LOOM`
dispatch with `explore=true` runs a bounded synthetic ablation study and uploads
artifacts. This is not a research benchmark or autonomous AGI service.

## Repository Layout

| Path | Purpose |
|---|---|
| `loom/` | Installable backbone-independent core and CLI |
| `examples/portable.yaml` | Complete small reference configuration |
| `tests_portable/` | Core, ablation and end-to-end tests |
| `docs/` | Method, configuration and dataset integration |
| `model_zoo/RankMixer.py` | Historical Native RankMixer recurrence |
| `model_zoo/OneTrans.py` | Historical OneTrans experiments |
| `model_zoo/LoopMT.py` | Head-LoopMT, distinct from Native recurrence |
| `config_*`, `scripts/`, `unirank/` | Original exploratory research snapshot |

Original benchmark reproduction uses `run_expid.py`, `requirements.txt`,
original preprocessing and dataset-specific YAML. See [UniRank documentation](README_UNIRANK.md).
Some historical configs contain local paths; analysis scripts may need excluded
artifacts. Portable checkpoints are not compatible with every historical extension.
Datasets, weights, logs and reports are excluded. A historical heatmap pilot
script contains explicitly labeled synthetic placeholders, not measured results.

## Attribution And License

Built on [UniRank](https://github.com/salmon1802/UniRank). Mixer operators follow
its implementations. Original copyright notices, upstream documentation and
the Apache-2.0 [LICENSE](LICENSE) are retained.
