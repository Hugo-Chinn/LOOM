# Coding Agent Guide

Read `docs/METHOD.md` and `docs/CONFIGURATION.md` before algorithm changes.
Supported portable API is `loom/`, `configs/portable/`, `tests/`. Historical
code is isolated in `research/`; its configurations are in `configs/research/`.
Do not confuse Head-LoopMT and Native recurrence or modify historical runs.

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
loom demo-data --output /tmp/loom-agent-data
python -m loom.sweep --config configs/portable/default.yaml \
  --data /tmp/loom-agent-data --output /tmp/loom-agent-study \
  --variants full no_bcu no_mas no_cbm untied \
  --seeds 20262027 20262028 --max-runs 10
```

Use fresh output directories and explicit budgets. Synthetic CI data checks
execution, not method superiority. Never invent per-seed experimental values.

- Freeze backbone, splits, labels, optimizer and seeds across component ablations.
- Change one component per arm; persist resolved config and parameter count.
- Report every task, not only macro AUC or improved tasks.
- Pair per-task differences over common seeds; tune on validation only.
- No-CBM is first-parent routing, retaining other branch computations.
- Graphs with no multi-parent nodes have CBM=N/A.
- Untied changes capacity; report its parameter overhead.
- Do not launch unbounded external/GPU experiments.

GitHub `Portable LOOM` CI tests Python 3.10/3.12 on CPU. Manual dispatch with
`explore=true` runs ten synthetic smoke-study runs and uploads JSON artifacts.
Permissions are read-only and no external agent credentials are supplied.
External coding agents can follow this guide to propose PRs; CI is not an
autonomous AGI training service.
