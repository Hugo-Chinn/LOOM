# Agent Architecture Exploration Task

Copy this into an issue/PR. Fill all fields; use "unmeasured" where appropriate.

## Hypothesis

- Candidate name and motivation:
- Target component (update / fusion / conditioning / supervision / backbone):
- Expected mechanism and possible failure/negative-transfer cases:
- What remains unchanged:
- Parameter-budget change:

## Implementation

- Registered update name and new options, defaults and constraints:
- Exact algorithm change, with equations/pseudocode:
- File list:
- Checkpoint compatibility and registration requirements:
- CPU / CUDA / BF16 support actually tested:

## Budget And Controls

- Dataset and provenance; synthetic or real:
- Split protocol and feature/label preprocessing:
- Baseline config and candidate config:
- Seeds (formal comparison: 20262027..20262030):
- Max runs, epochs, device allocation and stop criteria:
- Checkpoint/architecture selection metric on validation:
- Single-component ablation and capacity accounting:

## Verification

- Contract tests and end-to-end workflow:
- No regression in default Mixer or historical operator parity:
- Configs, checkpoints and raw per-seed/task output paths:
- Failed/missing runs and undefined AUCs:

## Results

| Task | Full AUC Mean ± Sample Std | Candidate AUC Mean ± Sample Std | Paired Delta | Common Seeds |
|---|---|---|---|---|
| every task | | | | |

- Parameters, trainable/frozen scope:
- Inference latency / training time / peak memory / GFLOPs:
- Hardware, dtype, batch size and measurement protocol, or "unmeasured":
- Validation-only selection rationale:
- Gains, regressions and limitations:
- Synthetic smoke tests are **not** method-superiority evidence.

## PR Acceptance

- Tests pass and defaults are unchanged unless explicitly approved.
- Candidate is reproducible from a committed config and named implementation.
- No raw/private datasets, keys, generated weights or fake results committed.
- Claims do not exceed actual validation.
