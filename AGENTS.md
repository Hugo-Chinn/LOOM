# Coding Agent Guide

Read `docs/METHOD.md` and `docs/CONFIGURATION.md` before algorithm changes.
Supported portable API is `loom/`, `configs/portable/`, `tests/`. Historical
code is isolated in `research/`; its configurations are in `configs/research/`.
Do not confuse Head-LoopMT and Native recurrence or modify historical runs.
For new architectures read `docs/ARCHITECTURE_EXPLORATION.md` and fill
`docs/AGENT_TASK_TEMPLATE.md`. Register a fresh update factory in `loom/updates.py`
instead of editing the graph/loss pipeline for an update-only experiment.
Built-in `attention` is exploratory, not a proven improvement.

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

## Server GPU operational memory

Verified on 2026-11-10; recheck live state before every experiment:

- CUDA error 304, NVML initialization failure or `nvidia-smi` driver errors
  can result from an execution sandbox hiding `/dev/nvidiactl`,
  `/dev/nvidia-uvm` and `/dev/nvidiaN`. Compare device visibility and the same
  interpreter in a permitted GPU-enabled context before diagnosing a driver
  fault. Follow the current tool permission policy: request escalation only
  when supported and required; unrestricted sessions run commands directly.
- This server's 8 H20s and driver 575.57.08 passed the GPU audit. FP32/BF16
  forward/backward passed on GPUs 0–3. These are dated checks, not guarantees
  of current health or availability.
- The verified Native environment is
  `/home/HugoChinn/LoopMT/UniRank/.venv-unirank/bin/python`
  (Python 3.12.16, PyTorch 2.6.0+cu124). Select it explicitly; execute current
  study code from this checkout. The historical UniRank worktree is separate.
- `rsi_holder.py` can occupy an otherwise idle card with ~429 MiB and ~65%
  utilization, and reacquire it later. Match UUIDs and NVML PIDs against
  `/jizhi/jizhi2/worker/trainer/rsi_holder/rsi_holder.py status`.
  `[Not Found]` process names can reflect host/container PID namespaces.
  Never assume yesterday's free GPU is available now.
- For an authorized experiment, coordinate per-UUID holder `pause`/`resume`,
  verify the target card is released, and restore its prior state afterward.
  Do not stop the entire holder or unrelated jobs. Run a bounded CUDA
  preflight before training.

The RelSup branch records the audit in
`explorations/relational_supervision/ENVIRONMENT.md` and provides
`python -m explorations.relational_supervision.gpu_preflight`
(requires exactly one explicitly selected `CUDA_VISIBLE_DEVICES`). Carry
these operational notes forward when preparing other research branches.
