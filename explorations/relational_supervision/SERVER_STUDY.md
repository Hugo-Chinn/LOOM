# Taobao / MerRec Phase 1, 2026-11-10

User allocation: GPUs 0–5, with concurrent processes on each card allowed.
GPU 6–7 are outside this allocation. This run treats both datasets as
independent exploratory screens; MerRec is not labelled held-out transfer.
No dataset's test scores select any arm or follow-up.

- Component: supervision only; existing RelSup single-term arms and Full.
- Controls: unchanged Native width-2 DAG references, labels, splits, model,
  loss settings, optimizer, batch size 8192, one epoch, BF16 and compilation.
- Seeds: 20262027, 20262028, 20262029, 20262030 for every arm and dataset.
- Budget: 20 runs per dataset, 40 total; 24 hours queue deadline, 6 hours per
  run. No automatic retries or Phase-2 expansion. Failed attempts remain in
  the queue journal and block further launches for that dataset.
- GPU pools: 0–2 Taobao, 3–5 MerRec; at most two processes per card. Complete
  Full S20262027 first on each dataset before opening the remaining slots.
- Selection: the existing validation-only 3/4-seed per-target rule and 0.5%
  relative mean logloss tolerance. Report all tasks and common seeds.
- Environment: Python 3.12.16 / PyTorch 2.6.0+cu124, explicitly using
  `/home/HugoChinn/LoopMT/UniRank/.venv-unirank/bin/python`.

The current runtime wrapper preserves the external GPU UUID mask (the
historical setup helper would overwrite it with `0`), and redirects only
feature metadata writes to per-run directories. Original Parquet paths and
model hyperparameters are unchanged. The resolved per-run configuration
records the metadata path. Historical source files are not edited.

## Verification before formal launch

53 core/exploration tests passed and all 10 three-epoch synthetic smoke runs
finished. Each real dataset completed a batch-8192 BF16/compiled check with
all five arm losses finite, finite gradients, and zero-weight base-loss
parity. These checks used training data only and produced no reusable model.

| Dataset | Tasks | Parameters | Peak allocated GiB | Peak reserved GiB |
|---|---|---|---|---|
| Taobao_Action | is_click, cart, fav, buy | 33,236,036 | 4.58 | 5.83 |
| MerRec_Action | Like, Cart, Offer, Checkout, Purchase | 803,849,962 | 15.61 | 20.16 |

These are one-batch execution measurements, not full-epoch peak memory or
throughput. The model parameter count differs across datasets because their
vocabularies differ; every arm within a dataset keeps the same parameter set.

## Outputs and operation

Run root: `runs_local/relsup-20261110T1810/`.
Configs: `configs/local/relsup-20261110T1810/{taobao,merrec}-p1/`.
Data: `/home/HugoChinn/LoopMT/UniRank/data/preprocessed/`.

`queue.json` records every pending/running/completed/failed attempt, physical
GPU, PID, stdout path and elapsed time. `resources.json` records previous
holder state and assigned UUIDs. `stdout/` holds raw logs; the Native model
logs and environment records are under `taobao/` and `merrec/`. The historical
runner computes test metrics; keep those blocks out of selection and inspect
only the validation reports until settings are frozen.

The scheduler restores only the holder pause flags it acquired on exit. A
SIGTERM requests orderly termination of its own workers and restoration.
After SIGKILL or host failure, inspect child PIDs before using `--restore-only`.
It does not reserve against unrelated schedulers.

```bash
/home/HugoChinn/LoopMT/UniRank/.venv-unirank/bin/python \
  -m explorations.relational_supervision.server_queue \
  --root runs_local/relsup-20261110T1810 \
  --configs configs/local/relsup-20261110T1810/taobao-p1 \
            configs/local/relsup-20261110T1810/merrec-p1 \
  --gpus 0 1 2 3 4 5 --slots 2 --max-runs 40 --hours 24 --run-hours 6
```

All formal jobs must use the launch commit; do not switch the active checkout
or edit its code while this queue is active. No superiority claim exists at
launch. Full-epoch time, memory and every per-task paired result remain to be
reported from completed runs.
