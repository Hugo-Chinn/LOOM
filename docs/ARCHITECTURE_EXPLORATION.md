# Architecture Extension Contract

Read METHOD.md first. Changing an update block is a multi-task-module study;
changing the encoder is a separate backbone study. Do not combine them and
attribute the result to one component.

## Built-In Registry

`loom/updates.py` registers two factories:

- `mixer`: default historical-style token Mixer; no `update_options`.
- `attention`: exploratory pre-norm token self-attention and FFN;
  `update_options.num_heads` defaults to 4 and must divide token_dim.

Attention is an example candidate, not a validated improved LOOM recipe.
Default Mixer parameter names and math remain unchanged. Its added configuration
keys default to Mixer, so prior portable configs/checkpoints remain readable.

```bash
loom train --config configs/portable/attention.yaml --data runs_local/demo-data \
  --output runs_local/attention/S20262027 --seed 20262027
python -m loom.sweep --config configs/portable/default.yaml \
  --data runs_local/demo-data --output runs_local/architecture-study \
  --variants full attention --max-runs 8
```

The sweep's `attention` preset uses four heads. Adapt it deliberately if the
base token dimension is not divisible by four.

## Register Your Update

```python
from torch import nn
from loom import LOOM, LOOMConfig, register_update

class MyUpdate(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.update_options:
            raise ValueError("my_update accepts no options")
        self.projection = nn.Linear(config.token_dim, config.token_dim)

    def forward(self, tokens):
        return tokens + self.projection(tokens).tanh()

register_update("my_update", MyUpdate)
config = LOOMConfig(
    tasks=["click", "buy"], topology="linear",
    update_type="my_update", token_dim=64, num_tokens=4,
)
core = LOOM(config)
```

Each factory must return a **fresh** nn.Module. Input/output contract:

- Float tensor `[B,T,D]`, same shape, device and dtype.
- Autograd must reach inputs and trainable update parameters.
- No batch-dependent parameter registration or hidden persistent batch state.
- Dropout follows train/eval mode; inference does not mutate model state.
- Shared mode invokes the same block for all behaviors/preloops.
- Untied mode calls the factory separately and must have disjoint parameters.
- State must be serializable with `state_dict`.

The registry is intentionally not an arbitrary module-import mechanism in YAML.
For a custom checkpoint, register the implementation before constructing/loading
the model in every process. Stock `loom train/predict` only knows built-ins.
To make a candidate available through stock CLI, add its implementation and
registration to `loom/updates.py` and submit a PR.

The core owns conditioning, graph execution, merge, gating, readouts and loss.
Do not duplicate those mechanisms inside a factory unless that is the explicitly
declared experimental change.

## Required Candidate Tests

Extend `tests/test_update_contracts.py`:

1. Forward on fixed tokens, all ordered task outputs finite.
2. Backward reaches update blocks and token inputs.
3. Shared/untied and preloops preserve intended parameter ownership.
4. Save/load yields equal eval predictions.
5. Invalid options fail early; no ignored configuration keys.
6. Default Mixer and historical parity tests still pass.
7. End-to-end train and predict on the small demo.

CPU tests do not establish CUDA/BF16 compatibility. Test those separately before
claiming support; keep FLOPs/latency claims tied to measured hardware/protocol.

## Exploration Protocol

Use the [task template](AGENT_TASK_TEMPLATE.md) in each issue or PR. Define
hypothesis, changes, budget and validation criteria before running.
Fix encoder, data/splits, primary label semantics, optimizer, epoch budget,
batch size, seeds and checkpoint selection. If capacity changes, report it.
Keep per-task tradeoffs; no macro-only conclusions or test-set selection.

Run all tests, then a bounded synthetic smoke study. For real evidence use a
declared real dataset and four shared seeds. Persist resolved config and
per-seed/task metrics; `loom.report` computes paired differences. Report undefined
AUC, failed runs and common-seed counts instead of silently dropping them.

For efficiency comparisons declare device, dtype, T/D, batch size, warmup,
timing synchronization, measured iteration count, auxiliary supervision state
and whether the encoder is included. Separate inference from forward+backward
training. Parameter count alone is not latency or GFLOPs. The current CLI does
not provide a validated FLOPs benchmark; unsupported numbers must be marked
unmeasured.

CI tests contracts and provides a manual bounded synthetic study. It does not
generate architectures or supply an autonomous external Agent.
