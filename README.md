# LOOM / LoopMT

Research implementation of behavior-conditioned recurrent multi-task ranking, built on [UniRank](https://github.com/salmon1802/UniRank).

## Implementation

- Native RankMixer recurrence: `model_zoo/RankMixer.py`.
- OneTrans experiments: `model_zoo/OneTrans.py`.
- Head-level LoopMT: `model_zoo/LoopMT.py` (distinct from Native backbone recurrence).
- Multi-task baselines: `model_zoo/MTLBaseline.py` and related models.
- Experiment configurations: `config_rankmixer_*`, `config_onetrans_*`.
- Training entry point: `run_expid.py`.

## Setup And Training

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare the datasets using scripts under `data/` and set dataset paths in the chosen `dataset_config.yaml`. Some research configurations contain machine-specific paths that must be adjusted. Datasets are not bundled.

```bash
python run_expid.py --config config_rankmixer_structure_merrec --expid Struct_MerRec_True_S20262027 --gpu 0
```

## Scope And Reproducibility

This is a source snapshot of the research workspace, not a validated minimal release. It includes exploratory configurations and scripts. Model weights, raw datasets, experiment logs, reports, virtual environments, and synthetic heatmap values are excluded. No experimental performance claims are established by this release. Some analysis scripts require local experiment artifacts that are not included. Native component ablations use four seeds: 20262027, 20262028, 20262029, 20262030.

## Attribution And License

The underlying framework is UniRank; its original documentation is preserved in `README_UNIRANK.md`. Original copyright notices and the Apache-2.0 `LICENSE` are retained.
