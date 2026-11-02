# Release Validation

Local release validation on 2026-11-02:

- Python 3.11, PyTorch 2.6.0, CPU execution.
- Portable tests: core forward/backward, disabled BCU, no-aux main loss,
  untied/preloop gradients, mean/softmax/first/branch fusion, residual bypass,
  graph validation, task ordering, label validation and paired reporting.
- Full RankMixer update block matches the historical operator under copied
  weights. This is operator parity, not full checkpoint/training parity.
- Synthetic end-to-end training, per-task evaluation and checkpoint inference.
- Editable package build and installed console entry point verified.

The isolated editable-install check reused existing dependencies. A separate
attempt to download CPU PyTorch did not return promptly and was stopped.
Installation from a completely clean dependency environment is therefore not
locally verified. GitHub Actions defines fresh CPU Python 3.10/3.12 jobs; remote
workflow completion must be checked independently.

No industrial-scale AUC reproduction, GPU throughput benchmark, or superiority
claim is established by these smoke tests. Historical research scripts remain
outside the supported portable test suite.
