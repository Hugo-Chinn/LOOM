# RelSup: Relational Supervision For Sibling And Parent/Child States

Status: **code written, nothing executed with PyTorch yet.** The authoring
environment had no PyTorch, so `tests/test_relsup.py` and every training path
below are unrun. The torch-free protocol tooling (arms, grids, config
generation, validation decisions, gated test reporting) is covered by
`tests/test_protocol.py`, which was executed and passes. Run step 0 first.

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
| `arms.py` | pre-registered arms, seeds, grid, target tasks and decision constants |
| `environment.py` | commit, versions, GPU and determinism flags per run |
| `portable_model.py` | `RelationalLOOM(loom.LOOM)` subclass; core untouched |
| `run_portable.py` | bounded NPZ study, per-task CSVs, cross-state AUC matrix |
| `research_model.py` | `RankMixerRelSup(RankMixer)` subclass; historical file untouched |
| `run_research.py` | runs `research/run_expid.py` with the subclass injected |
| `make_research_configs.py` | phase 1 / 2 / 3 / confirm configs plus `protocol.json` |
| `run_queue.sh` | sequential resumable queue on one GPU |
| `collect_research.py` | validation-only decisions; test tables only after freezing |
| `tests/test_relsup.py` | parity, gradients, detach, N/A graphs, key encoding (needs torch) |
| `tests/test_protocol.py` | grid, targets, phases, survival rule, test gating (no torch) |

Defaults: every weight 0 (loss equals the base model), margin 0.2, arm weight
0.1, `detach_reference=true`, `diagnostics=true`. Sibling arms are rejected on
graphs without siblings (linear); edge arms need at least one edge.
`RankMixerRelSup` requires `model_type=looped` and a per-behavior supervision
mode. Checkpoints are interchangeable with the base model.

## Step 0: Verify On The Server (CPU, minutes)

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q                                     # existing suite
python -m pytest explorations/relational_supervision/tests -q # RelSup + protocol
loom demo-data --output runs_local/demo-data
python -m explorations.relational_supervision.run_portable \
  --config configs/portable/default.yaml --data runs_local/demo-data \
  --output runs_local/relsup-smoke --seeds 20262027 20262028 --max-runs 10
```

The smoke study only proves the pipeline runs. The demo labels are four
independent linear tasks with no behavior structure. The portable runner
writes test CSVs immediately, so it is outside the selection protocol.

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

## Fairness And Selection Protocol

The protocol below is fixed before any real-data result is inspected. It is
part of the experiment definition, not a post-hoc reporting choice.

### Frozen controls

Every arm uses the same dataset version, chronological splits, labels,
feature preprocessing, graph, backbone, optimizer, batch size, learning rate,
dropout, epoch budget, supervision mode and parameter initialization policy.
The only intended change in Phase 1 is one RelSup loss term. The resolved
configuration, parameter count, repository commit, Python/PyTorch/CUDA
versions, GPU identity and seed are saved for every run.

The development seed set is exactly
`{20262027, 20262028, 20262029, 20262030}`. Every arm and its `full` control
must run on the same seed set, and comparisons are paired by the common seed.
No seed is added, removed or replaced after looking at results. If the final
study reuses these seeds after selection, its conclusion is labelled
exploratory rather than independent confirmation.

### Selection data and metrics

Validation is the only data split permitted for choosing an arm, weight,
margin, combination or dataset-specific continuation. The test split is read
only after the arm and all hyperparameters are frozen; test AUC or logloss
must never determine continuation or selection.

The primary selection metric is validation AUC for every task, reported per
task rather than only as a macro average. Validation logloss is a safety
metric and must also be reported. The target tasks for each arm are defined
by its graph relation before running it. An arm survives Phase 1 only if its
paired validation AUC delta versus `full` is positive on the predefined
target tasks in at least 3 of 4 seeds, while no task violates the predefined
validation logloss regression tolerance. The target-task list and tolerance
must not be changed after observing results.

### Pre-registered tuning grid

For a surviving single-term arm, Phase 2 tests only the following candidates:

```
weight  in {0.03, 0.10, 0.30}
margin  in {0.0, 0.2, 0.5}
```

Weight and margin are selected using validation only. This is a finite grid,
not an invitation to expand the search after seeing test results. If a new
grid or training setting is needed, it is a new exploratory phase and all
attempted candidates, including unsuccessful ones, remain in the report.
The selected setting is frozen before any Phase 3 combination or transfer
run. Phase 3 does not retune on the transfer dataset.

### Confirmation and reporting

The preferred confirmation design reserves a fresh, predeclared seed set for
the final frozen configuration. If no fresh seeds are affordable, the four
development seeds may be reused only with an explicit exploratory-result
label; this must not be described as independent confirmation.

All attempted arms and hyperparameters, including failed or incomplete runs,
are recorded. Final reports contain every task and every common seed's AUC
and logloss, paired deltas versus `full`, sample standard deviations, seed
counts, diagnostics, parameter counts and training cost. The report must not
selectively show only improved tasks or the best test result.

GPU execution must use a fixed environment and record GPU model, CUDA,
PyTorch, driver and worker settings. If the stack is not bitwise deterministic
on GPU, results are described as seed-controlled rather than exactly
reproducible.

### How the protocol is enforced

Values the protocol requires to be fixed in advance live in `arms.py` and are
copied into every config directory's `protocol.json`. Tools refuse to mix
directories created under different constants.

| Rule | Enforcement |
|---|---|
| Frozen controls | every arm is a copy of the same-seed Full block; only `model`, `relsup_*`, checkpoint root (and `seed` for confirmation) differ; the generator rejects sources whose seeds differ beyond `seed` |
| Development seeds | generator rejects any source seed set other than 20262027..20262030 |
| One term per arm | phase 1 arms are generated, not hand-written |
| Validation-only selection | `collect_research --stage phase1/phase2` never parses a Test block |
| Test after freezing | `--stage test` needs the frozen phase-2 `selection.json`; outputs are never overwritten |
| Finite grid | phase 2 emits exactly weight {0.03, 0.10, 0.30} x margin {0.0, 0.2, 0.5}; other values cannot be generated |
| No partial-grid choice | phase-2 freezing is refused until every grid point has all four seeds |
| All attempts reported | `attempts.json` lists every arm x seed as complete / incomplete / missing |
| Environment record | `<expid>.env.json`: commit, dirty files, Python, PyTorch, CUDA, cuDNN, GPU, driver, determinism flags, workers, resolved config |
| Seed role labels | test tables carry "exploratory" for development seeds and "confirmation" for 20262031..20262034 |

**Values filled in here because the protocol leaves them open. Confirm or
change them before the first real-data run; afterwards they are frozen.**

- Target tasks, derived from the graph: sibling arms target the sibling tasks
  (Taobao: cart, fav); edge arms target every task with a parent (cart, fav,
  buy); a combination targets the union.
- "Positive in at least 3 of 4 seeds" is applied to **each** target task
  separately, not to their average.
- Validation logloss tolerance: the mean paired relative increase versus
  `full` must be at most **0.5 %** on every task (`LOGLOSS_TOLERANCE_REL`).
- Phase-2 choice among grid points that pass the survival rule: highest mean
  paired validation AUC delta averaged over the target tasks; ties go to the
  smaller weight, then the smaller margin. `sib_decor` and `edge_distill`
  have no margin, so their grid is the three weights.
- Confirmation seeds: 20262031, 20262032, 20262033, 20262034.
- Validation metrics are those of the evaluation that saved the best
  checkpoint. The Full configs already monitor `{AUC: 1, logloss: 0}`, so
  validation logloss is logged without affecting checkpoint choice; the
  generator rejects sources where this is not the case.

Known limits of this protocol, stated so results are read correctly:

- The 3-of-4 sign rule is a screen. Under no effect a single target task
  passes with probability 5/16; requiring every target task lowers this but
  phase-1 survival is not evidence of an effect.
- Phase 2 picks the best of up to nine settings on the same four seeds, so
  its validation gain is optimistically biased. Only confirmation-seed
  results on the frozen setting support a claim.
- All phases must run from one commit. `collect_research` warns when
  completed runs record different commits.

## Step 2: Phase 1 On Real Data, One Term Per Arm

```bash
python research/prepare_native_config.py --dataset taobao \
  --data-root /absolute/preprocessed --output configs/local/taobao-full \
  --model-root /absolute/outputs/taobao-full
python -m explorations.relational_supervision.make_research_configs --phase 1 \
  --source configs/local/taobao-full --output configs/local/taobao-relsup-p1 \
  --model-root /absolute/outputs/taobao-relsup            # 5 arms x 4 seeds = 20
# one run first, then the queues (resumable; one GPU per queue)
bash explorations/relational_supervision/run_queue.sh configs/local/taobao-relsup-p1 0 '^full_S20262027$'
bash explorations/relational_supervision/run_queue.sh configs/local/taobao-relsup-p1 0 '^(full|sib_margin)_'
bash explorations/relational_supervision/run_queue.sh configs/local/taobao-relsup-p1 1 '^(sib_decor|edge_gain|edge_distill)_'
python -m explorations.relational_supervision.collect_research --stage phase1 \
  --config configs/local/taobao-relsup-p1 --output runs_local/relsup-taobao/phase1
```

Two queues may race on feature metadata at start-up (see
HISTORICAL_REPRODUCTION.md); start the second after the first has begun training.

`full` is rerun through the same launcher so both sides share code path,
hardware and seeds. It should track the historical `DAGTrue` table; with
`net_dropout: 0` the extra no-grad head evaluations do not consume random
numbers, but exact equality with the old logs is not guaranteed and was not
checked. `phase1/selection.json` lists the survivors and the reason for every
other arm. Use the same `--model-root` in every phase.

## Step 3: Phase 2, The Pre-Registered Grid For Survivors

```bash
python -m explorations.relational_supervision.make_research_configs --phase 2 \
  --source configs/local/taobao-full --output configs/local/taobao-relsup-p2 \
  --model-root /absolute/outputs/taobao-relsup \
  --selection runs_local/relsup-taobao/phase1/selection.json --max-runs 80
bash explorations/relational_supervision/run_queue.sh configs/local/taobao-relsup-p2 0
python -m explorations.relational_supervision.collect_research --stage phase2 \
  --config configs/local/taobao-relsup-p1 configs/local/taobao-relsup-p2 \
  --output runs_local/relsup-taobao/phase2
```

A margin term adds 8 grid points (32 runs); a weight-only term adds 2 (8 runs).
The phase-1 point is reused, as are the phase-1 `full` runs. The budget is
explicit: `--max-runs` must be raised to the number printed in the refusal.
`phase2/selection.json` freezes one setting per surviving term.

## Step 4: Phase 3, Combination, Confirmation And Transfer

```bash
SEL=runs_local/relsup-taobao/phase2/selection.json
# combination of the frozen settings (only if at least two terms survived)
python -m explorations.relational_supervision.make_research_configs --phase 3 \
  --source configs/local/taobao-full --output configs/local/taobao-relsup-p3 \
  --model-root /absolute/outputs/taobao-relsup --selection $SEL
# fresh seeds: full, each frozen arm, and the combination
python -m explorations.relational_supervision.make_research_configs --phase confirm \
  --source configs/local/taobao-full --output configs/local/taobao-relsup-confirm \
  --model-root /absolute/outputs/taobao-relsup --selection $SEL
# read test only now, for every attempted arm
python -m explorations.relational_supervision.collect_research --stage test \
  --config configs/local/taobao-relsup-p1 configs/local/taobao-relsup-p2 \
           configs/local/taobao-relsup-p3 configs/local/taobao-relsup-confirm \
  --selection $SEL --output runs_local/relsup-taobao/test
```

The combination is judged by the same validation rule (`decisions.json` of a
`--stage phase2` report that includes the phase-3 directory) before any test
read. Transfer: rerun `--phase confirm` with a second dataset's Full source
(`--dataset kuairand` or `merrec` in `prepare_native_config.py`) and the same
`$SEL`; no retuning there. Mechanism check, exploratory and reported as such:
the frozen `sib_margin` on a `use_behavior_embedding: false` source. If the
gain disappears, the loss works by making the behavior embedding matter.

## Reporting

Fill `docs/AGENT_TASK_TEMPLATE.md` from `validation_summary.csv`,
`test_summary.csv`, `attempts.json` and `decisions.json`: every task, every
common seed, AUC and logloss, paired deltas, sample standard deviations, seed
counts, the three diagnostics and parameter counts, for every attempted arm.
Parameter count is unchanged by construction; training cost rises by a few
head evaluations per step and is **unmeasured** until timed on the server.
No result exists yet.
