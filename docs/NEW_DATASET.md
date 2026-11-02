# New Dataset And Custom Backbone

## NPZ Reference Workflow

Export `train.npz`, `valid.npz`, `test.npz` with:

- `x`: finite float32 `[N,F]`, identical feature columns in all splits.
- `y`: float32 `[N,K]`, binary 0/1; NaN means missing label.
- `tasks`: Unicode string `[K]`, exactly matching the configured order.

```python
import numpy as np
np.savez_compressed("train.npz", x=train_features.astype("float32"),
                    y=train_labels.astype("float32"),
                    tasks=np.asarray(["click", "like", "purchase"]))
```

Repeat for validation/test. Split before fitting transforms, following deployment
semantics (often chronological). Prevent future-history leakage and duplicate
events across splits. Do not infer ancestor labels from purchase unless the
source data explicitly supports it. Encode categories appropriately; raw integer
IDs are not meaningful numeric distances.

Copy `examples/portable.yaml`, change tasks and graph. Every DAG task must be
declared, including roots.

```bash
loom train --config my_dataset.yaml --data /path/to/splits \
  --output runs_local/my_dataset/S20262027 --seed 20262027 --device cpu
loom predict --checkpoint runs_local/my_dataset/S20262027/best.pt \
  --input /path/to/splits/test.npz --output predictions.npz
```

GPU: pass `--device cuda:0` with a suitable PyTorch build.
The runner loads all arrays into RAM; it is not an industrial streaming loader.
Normalization is fitted on train only and persisted in `best.pt`. Validation
selects the checkpoint, test is evaluated afterward.

Run four seeds into separate output directories. `metrics.json` includes seed,
selected epoch, total parameters and **AUC/logloss/observed count per task**.
AUC is JSON null if a task lacks both classes, never fabricated as zero.
Inference needs only `x` and uses checkpoint-owned normalization; output NPZ has
`probabilities` and ordered `tasks`.

## Attach A Backbone

```python
import torch
from loom import LOOM, LOOMConfig

core = LOOM(LOOMConfig(
    tasks=["click", "like", "purchase"], token_dim=128, num_tokens=4,
    graph={"click": [], "like": ["click"], "purchase": ["like"]},
))
# encoder is your nn.Module returning [B,4,128].
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(core.parameters()), lr=1e-3
)
encoder.train()
core.train()
optimizer.zero_grad()
output = core(encoder(batch_features))
loss = core.loss(output, batch_labels)  # [B,3]
loss.backward()
optimizer.step()

encoder.eval()
core.eval()
with torch.inference_mode():
    probabilities = core(encoder(batch_features))["logits"].sigmoid()
```

Persist both encoder/core states, config, feature/task ordering, vocabulary and
preprocessing. Move modules and labels to the same device. Tokens are assumed
valid; sequence padding masks belong to the encoder.

Replacing RankMixer's interaction stack with LOOM is Native-style integration.
Attaching LOOM after an entire encoder is post-backbone refinement, a different
architecture that must be reported separately. The numeric reference encoder is
not the historical RankMixer tokenizer.

For original industrial experiments use retained `run_expid.py`, UniRank data
preprocessing and dataset-specific configurations, installing `requirements.txt`
as needed. See `README_UNIRANK.md`; machine-specific YAML paths need adjustment.
