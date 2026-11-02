# Repository Architecture

```text
LOOM/
  loom/                   supported library, CLI and bounded experiments
    config.py             validated algorithm configuration
    core.py               token Mixer and recurrent behavior graph
    cli.py                example encoder, training, evaluation, inference
    sweep.py              bounded component comparisons
    report.py             per-seed/task metrics and paired statistics
  configs/
    portable/default.yaml supported example
    research/             historical experiment families
  tests/                  supported core and end-to-end tests
  docs/                   equations, schema, data contract and validation
  research/               historical UniRank code and exploratory tools
  .github/workflows/      CPU tests and manual bounded exploration
  pyproject.toml          portable installation
```

## Reading Order

1. Root README: run the minimal workflow.
2. `configs/portable/default.yaml`: inspect a complete configuration.
3. `docs/METHOD.md`, then `loom/core.py`: understand the algorithm.
4. `docs/NEW_DATASET.md`: attach your data or encoder.
5. `docs/CONFIGURATION.md`: explore component choices.
6. `research/README.md`: consult original experimental implementation only
   when historical benchmark reproduction is required.

Portable dependencies do not import the research framework. Historical models
are not part of the installed `loom-mtl` package. All YAML configuration assets
are centralized under `configs/`, with explicit portable/research separation.

## Migration From The Initial Snapshot

| Old Path | New Path |
|---|---|
| `examples/portable.yaml` | `configs/portable/default.yaml` |
| `config/`, `config_*` | `configs/research/config/`, `configs/research/config_*` |
| `tests_portable/` | `tests/` |
| `model_zoo/`, `unirank/` | `research/model_zoo/`, `research/unirank/` |
| `scripts/`, `data/`, `AutoAna/` | Corresponding directories in `research/` |
| `tests/` (historical) | `research/tests/` |
| `run_expid.py` and old root runners | `research/` |
| `README_UNIRANK.md`, `assets/` | Corresponding paths in `research/` |

The experiment workspace is unchanged. This organization applies only to the
GitHub release repository.
