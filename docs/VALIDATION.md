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

## Reproduction Audit After Repository Reorganization

- Fresh clone of GitHub commit `18d98dc`: all 25 tests passed using existing local
  Python 3.11 / PyTorch 2.6.0 dependencies, without untracked research assets.
- An audit found a legacy branch supervision mismatch: the portable core used
  graph descendants while historical branch uses all later stages in task order.
  Fixed and covered by a regression test; DAG retains descendant supervision.
- After the fix: all 27 tests passed, including same-seed CPU optimizer updates.
- Two complete CLI training runs on the same generated data with seed 20262027
  produced exactly identical metrics, training histories and selected epoch.
- Demo test AUCs (synthetic pipeline check only):

| Task | Test AUC | Test Logloss |
|---|---:|---:|
| click | 0.7749873482 | 0.6165481806 |
| cart | 0.7568821339 | 0.6184345484 |
| fav | 0.7673451922 | 0.6125177145 |
| buy | 0.6998168498 | 0.6326332092 |

These values are not historical dataset results and cannot be compared with
Taobao/KuaiRand benchmarks. Fresh cloning validates source completeness, not
installation into a clean dependency environment. GitHub Actions API inspection
timed out locally, so remote workflow status is not confirmed.

**Historical AUC reproduction remains unverified.** It requires original
preprocessing, splits, labels, backbone, initializations, optimizer, training
budget and per-seed/task test comparison using the archived research runner.
The portable numeric encoder is intentionally different from that historical
feature/sequence backbone.
