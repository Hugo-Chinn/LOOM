# RelSup: Relational Supervision For Sibling And Parent/Child States

Status: **code written, nothing executed with PyTorch yet.** The authoring
environment had no PyTorch, so the tests in `tests/` and every training path
below are unrun. Only the torch-free helpers (graph relations, arm expansion,
config generation, log collection) were executed. Run step 0 first.

## Hypothesis

With one shared update block, siblings (for example cart and fav under click)
receive the same parent state and differ only by an additive behavior embedding
and a scalar gate. Nothing in the current objective asks a state to be *more
useful for its own task* than a neighbouring state:

- DAG deep supervision (MAS) only ties a task head to its descendants.
- Siblings have no loss term between them.
- A child is never required to improve on its parent for the child's task.

RelSup adds label-aware relational terms. All are training-only: no new
parameters, identical inference graph, identical checkpoints.

Target component: supervision. Unchanged: encoder, update block, graph, fusion,
gates, heads, primary loss, MAS, optimizer, data. Parameter change: none.

## Loss Terms

`l(t,u) = f_t(readout(H_u))` is head `t` on state `u`; `sg` stops gradient
through the *state* (the head stays trainable); `s_t = 2y_t - 1`; `O_t` is the
set of rows with an observed label for `t`. Each term is a mean over graph
pairs of a per-pair mean.

| Arm | Relation | Term |
|---|---|---|
| `sib_margin` | siblings (t,s), both directions | `softplus(m - s_t (l(t,t) - l(t,sg s)))` over `O_t` |
| `sib_decor` | siblings (t,s), shared parent p | mean squared cross-correlation of batch-standardized `pool(H_t)-pool(H_p)` and `pool(H_s)-pool(H_p)` |
| `edge_gain` | edge p -> c | `softplus(m - s_c (l(c,c) - l(c,sg p)))` over `O_c` |
| `edge_distill` | edge p -> c | `KL(sigmoid(sg l(p,p)) || sigmoid(l(p,c)))` |

Design choices and why:

- **Label-aware margins, not instance discrimination.** Pushing `H_cart` away
  from `H_fav` per sample is solved by a constant offset in the behavior
  embedding and adds no information. The margin can only be met in a direction
  correlated with the task label.
- **Stop-gradient on the reference state.** Otherwise the cheapest solution is
  to damage the sibling/parent state.
- **Centering and standardizing in `sib_decor`.** Removes the constant-offset
  and shrink-the-increment shortcuts. It guarantees "different", not "useful",
  so it is a regularizer and is tested separately.
- **`edge_gain` is the parent/child counterpart of `sib_margin`:** the child
  state must beat its parent state on the child's own task.
- **`edge_distill` is a soft version of MAS retention:** the parent head on the
  child state follows the parent's own prediction instead of only hard labels.
  It is added on top of MAS, not a replacement, to keep one change per arm.

Deliberately excluded: funnel constraints such as `p_buy <= p_cart`. The
repository's labels are not nested (METHOD.md: buy-positive does not imply
click/cart-positive), so such a constraint would be wrong here.

Expected failure cases to watch:

- cart/fav gain but **buy drops**: buy consumes the fusion of both siblings,
  and MAS asks `f_cart` to stay good on `H_buy` while `sib_margin` asks it to be
  worse on `H_fav`.
- Margins met by logit inflation: check logloss, not only AUC.
- `edge_gain` on click -> cart may hurt click through shared parameters.
- `sib_decor` over-separating positively correlated behaviors.

## Files

| File | Role |
|---|---|
| `structure.py` | edges and sibling pairs of a behavior graph (no torch) |
| `losses.py` | `RelSupConfig`, `collect`, `relational_loss` shared by both stacks |
| `arms.py` | arm definitions and weight/margin grids |
| `portable_model.py` | `RelationalLOOM(loom.LOOM)` subclass; core untouched |
| `run_portable.py` | bounded NPZ study, per-task CSVs, cross-state AUC matrix |
| `research_model.py` | `RankMixerRelSup(RankMixer)` subclass; historical file untouched |
| `run_research.py` | runs `research/run_expid.py` with the subclass injected |
| `make_research_configs.py` | derives arm configs from a prepared Full config |
| `run_queue.sh` | sequential resumable queue on one GPU |
| `collect_research.py` | final-Test-block AUC per seed/task, paired deltas vs `full` |
| `tests/test_relsup.py` | parity, gradients, detach, N/A graphs, key encoding |

Defaults: every weight 0 (loss equals the base model), margin 0.2, arm weight
0.1, `detach_reference=true`, `diagnostics=true`. Sibling arms are rejected on
graphs without siblings (linear); edge arms need at least one edge.
`RankMixerRelSup` requires `model_type=looped` and a per-behavior supervision
mode. Checkpoints are interchangeable with the base model.

## Step 0: Verify On The Server (CPU, minutes)

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q                                     # existing suite
python -m pytest explorations/relational_supervision/tests -q # RelSup contracts
loom demo-data --output runs_local/demo-data
python -m explorations.relational_supervision.run_portable \
  --config configs/portable/default.yaml --data runs_local/demo-data \
  --output runs_local/relsup-smoke --seeds 20262027 20262028 --max-runs 10
```

The smoke study only proves the pipeline runs. The demo labels are four
independent linear tasks with no behavior structure.

## Step 1: Baseline Diagnosis (Is There A Problem To Fix?)

`full` runs already log the diagnostics, without gradient:

- `sibling_gap`: mean label-signed logit advantage of a head on its own state
  over the sibling state. Near 0 means heads cannot tell sibling states apart.
- `sibling_decor`: 1/D-scale values mean decorrelated increments; values near
  the within-feature correlation mean the increments are nearly the same.
- `edge_gap`: same advantage of the child state over the parent state.
- Portable runs also store `cross_state_auc` (head t read from state u).

If `sibling_gap` is already clearly positive on the real dataset, the sibling
arms have little room and the edge arms are the better bet. Decide after
looking, and write the numbers down before running phase 2.

## Step 2: Phase 1 On Real Data, One Term Per Arm

```bash
python research/prepare_native_config.py --dataset taobao \
  --data-root /absolute/preprocessed --output configs/local/taobao-full \
  --model-root /absolute/outputs/taobao-full
python -m explorations.relational_supervision.make_research_configs \
  --source configs/local/taobao-full --output configs/local/taobao-relsup \
  --model-root /absolute/outputs/taobao-relsup            # 5 arms x 4 seeds = 20
# one run first, then the queue (resumable; one GPU per queue)
bash explorations/relational_supervision/run_queue.sh configs/local/taobao-relsup 0 '^full_S20262027$'
bash explorations/relational_supervision/run_queue.sh configs/local/taobao-relsup 0 '^(full|sib_margin)_'
bash explorations/relational_supervision/run_queue.sh configs/local/taobao-relsup 1 '^(sib_decor|edge_gain|edge_distill)_'
python -m explorations.relational_supervision.collect_research \
  --logs /absolute/outputs/taobao-relsup/Taobao_Action \
  --output runs_local/relsup-taobao-phase1
```

Two queues may race on feature metadata at start-up (see
HISTORICAL_REPRODUCTION.md); start the second after the first has begun training.

Controls: `full` is rerun through the same launcher so both sides share code
path, hardware and seeds. It should track the historical `DAGTrue` table; with
`net_dropout: 0` the extra no-grad head evaluations do not consume random
numbers, but exact equality with the old logs is not guaranteed and was not
checked. Budget: 20 runs, one epoch each as in the Full config.

## Step 3: Phase 2, Tune Only What Survived

Keep an arm if its paired per-task delta vs `full` is positive on its target
tasks in at least 3 of 4 seeds and no task regresses beyond the `full` seed
standard deviation. For survivors only:

```bash
python -m explorations.relational_supervision.make_research_configs \
  --source configs/local/taobao-full --output configs/local/taobao-relsup-p2 \
  --model-root /absolute/outputs/taobao-relsup \
  --arms full sib_margin --scales 0.3 3 --max-runs 12   # weights 0.03 and 0.3
```

`--margin 0 | 0.5` varies the margin the same way. Reuse the phase-1 `full`
logs (same model root) instead of retraining. Choose weights on validation AUC
from the logs; `collect_research.py` reports test only and must not be used
for selection.

## Step 4: Phase 3, Combination And Transfer

1. `combo_margin_gain` (and `combo_all` only if all four survived) at the
   chosen weights, four seeds.
2. Repeat the winning arm on one more dataset whose graph has siblings
   (`--dataset kuairand` or `merrec`).
3. Mechanism check: `sib_margin` on `no_bcu`-style configs
   (`use_behavior_embedding: false`). If the gain disappears, the loss works by
   making the behavior embedding matter, which is the intended mechanism.

## Reporting

Fill `docs/AGENT_TASK_TEMPLATE.md` from the CSVs: every task, mean and sample
standard deviation, paired delta, common seeds, plus logloss and the three
diagnostics. Parameter count is unchanged; training cost rises by a few head
evaluations per step and is **unmeasured**. No result exists yet.
