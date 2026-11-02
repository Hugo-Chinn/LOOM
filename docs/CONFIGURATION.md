# Configuration Schema And Search Space

`loom/config.py` provides the executable `LOOMConfig` schema. YAML sections are
`model` and `training`; unknown keys fail rather than being silently ignored.

| Model Key | Default | Meaning |
|---|---|---|
| `tasks` | required | Unique ordered names, matching label and output columns |
| `token_dim` | 64 | Positive integer D supplied by encoder |
| `num_tokens` | 4 | Positive integer T, fixed encoder token count |
| `topology` | dag | `dag`, `linear`, legacy four-task `branch` |
| `graph` | {} | Complete child-to-parent mapping required for DAG, including roots |
| `expansion_factor` | 2 | FFN hidden width = factor times D |
| `update_layers` | 1 | Mixer layers inside each recurrent update |
| `num_preloops` | 0 | Behavior-agnostic refinement before graph execution |
| `behavior_conditioning` | true | Behavior embedding; false genuinely removes it |
| `depth_conditioning` | false | Separate depth embedding, including preloop offset |
| `shared_update` | true | Reuse one Mixer; false allocates independent blocks |
| `use_loop_gate` | true | Per-node/preloop residual interpolation gates |
| `loop_gate_init` | 0.5 | Initial sigmoid gate, strictly between 0 and 1 |
| `supervision` | deep_progressive | `progressive` keeps main losses only |
| `auxiliary_loss_weight` | 0.5 | Nonnegative weight of mean auxiliary losses |
| `parent_fusion` | mean | DAG mean, learned softmax, or first-parent no-merge control |
| `task_loss_weights` | null | Positive primary weights in task order |
| `base_bypass_tasks` | [] | Tasks receiving zero-initialized tanh base-state residual |
| `dropout` | 0 | FFN dropout in [0,1) |

Training keys: `epochs=3`, `batch_size=128`, `learning_rate=0.001`. The CLI uses
AdamW with zero weight decay, validation macro AUC for checkpoint selection,
and **test AUC/logloss for every task**. `--seed` defaults to 20262027.

## Graph And Historical Mapping

```yaml
tasks: [click, cart, fav, buy]
topology: dag
graph:
  click: []
  cart: [click]
  fav: [click]
  buy: [cart, fav]
```

Values are parents, not children. Task order need not be topological.
`linear` derives a chain from task order; `branch` derives the four-task fork.
An explicit graph for a non-DAG mode is rejected, preventing ignored graph edits.

| Historical RankMixer | Portable |
|---|---|
| `use_behavior_embedding` | `behavior_conditioning` |
| `use_iteration_depth_embedding` | `depth_conditioning` |
| `share_loop_parameters` | `shared_update` |
| `loop_topology` | `topology` |
| `behavior_graph` | `graph` |
| `dag_parent_fusion` | `parent_fusion` for DAG |
| `supervision_mode` | `supervision`, progressive/deep_progressive only |
| `loop_block_layers` | `update_layers` |
| `base_bypass_tasks` | `base_bypass_tasks`, residual mode only |

Portable loop count equals task count. Historical `num_layers`, `loop_scope`,
final-state readout and task tokens are not implicitly mapped.

## Bounded Exploration

Start with width 2, one update layer, no preloops. Test components before combinations:

| Arm | Change |
|---|---|
| w/o BCU | `behavior_conditioning: false` |
| w/o MAS | `supervision: progressive`, `auxiliary_loss_weight: 0` |
| w/o CBM | `parent_fusion: first`, only when a merge node exists |
| Untied SRU | `shared_update: false`, report parameter increase |
| Behavior vs depth | Compare behavior-only, depth-only, both and neither |
| Capacity | Expansion 2 vs 4; do not presume width 8 helps |
| Preloops | 0 vs 2 |
| Gate | Off vs on with initial 0.5 |
| Topology | Predeclared meaningful graph vs representative perturbation |

Rare-task primary weights and bypass are dataset-specific tuning, not mandatory
default ingredients. Derive weights and tune on training/validation, not test.
Historical dataset gains do not establish gains on unseen datasets.

Keep splits, backbone, optimizer and seeds fixed across arms. Use four seeds
20262027..20262030. Report per-task per-seed AUC, mean, sample standard deviation,
paired delta against Full, and common seed count. Never replace this with
macro-only reporting or select topology using test AUC.

`loom.sweep` automatically writes per-seed/task metrics and paired summary CSVs.
For existing run directories, rerun aggregation with:

```bash
python -m loom.report --input runs_local/demo-ablation
```
