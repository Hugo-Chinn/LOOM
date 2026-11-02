# LOOM Algorithm And Scope

The portable `loom.LOOM` module implements Native-style recurrent updates over
token states, not historical Head-LoopMT (`research/model_zoo/LoopMT.py`). The original
UniRank research implementations remain available separately.

## Feedback State And Recurrent Update

An encoder produces `H = E(x)` of shape `[B,T,D]` once. Behavior-agnostic
preloops produce the base state `H0`. With parents `P(t)`:

```math
Z_t =
\begin{cases}
H_0 & P(t)=\varnothing,\\
H_p & P(t)=\{p\},\\
\frac{1}{|P(t)|}\sum_{p\in P(t)}H_p & \text{mean fusion}.
\end{cases}
```

```math
\widetilde H_t=U_\theta(Z_t+e_t+q_{\operatorname{depth}(t)}),
\qquad H_t=Z_t+\sigma(g_t)(\widetilde H_t-Z_t).
```

`e_t` is the behavior embedding; optional `q` is a separate depth embedding,
including preloop offset. Without gates the candidate directly becomes `H_t`.
`U` is RankMixer's token permutation and per-token GELU FFN, with pre-norm
residuals. Shared mode reuses one block across all behaviors and preloops;
untied mode allocates a block per behavior and per preloop.

DAG fusion supports mean and learned softmax. `parent_fusion=first` keeps every
branch computation but selects the first declared parent at merges. It is the
no-merge control; parent order matters. It is N/A on graphs with no merge nodes.

Legacy four-task `branch` mode uses ordered tasks `[Click, Cart, Fav, Buy]`:

```math
Z_{\mathrm{buy}}=H_{\mathrm{click}}
+\sigma(a)(H_{\mathrm{cart}}-H_{\mathrm{click}})
+\sigma(b)(H_{\mathrm{fav}}-H_{\mathrm{click}}).
```

This is not identical to arbitrary DAG mean fusion.

## Readouts, Labels And Optimization

```math
\ell_t=f_t(\operatorname{mean}_{tokens}H_t),\qquad \hat y_t=\sigma(\ell_t).
```

Deep supervision applies the same task head `f_t` at every descendant state.
Those auxiliary logits are never substituted for the main evaluated prediction.
For legacy `linear` and `branch` modes, historical supervision uses every later
stage in task order, rather than graph descendants. In branch mode this includes
Cart's auxiliary prediction on the Fav state. DAG mode uses descendants only.

```math
\mathcal L=
\sum_t w_t\,\operatorname{mean}_{i\in O_t}
\operatorname{BCEWithLogits}(\ell_{i,t},y_{i,t})
+\lambda\,\operatorname{mean}_{(t,u):u\in Desc(t)}
\operatorname{mean}_{i\in O_t}
\operatorname{BCEWithLogits}(f_t(H_{i,u}),y_{i,t}).
```

`O_t` is the set of samples with an observed task label. NaN means missing,
not negative. Primary weights do not reweight auxiliary losses, matching
historical RankMixer's convention. Empty task/stage observed sets are omitted.

**Buy-positive does not create Click/Cart-positive labels.** Preprocessing owns
label semantics. Main loss uses each observed task label; deep supervision
reuses that same label at descendant states. Missing labels are never silently
converted to negatives. The CLI rejects rows with no observed labels.

Selected tasks can receive a zero-initialized base-state residual bypass:

```math
r_t=\operatorname{mean}(H_t)+\tanh(b_t)\operatorname{mean}(H_0),\quad b_t=0.
```

## Compatibility Boundary

Supported: behavior/depth embeddings, tied/untied update, preloops,
mean/softmax/first DAG fusion, legacy branch fusion, gates, progressive/deep
supervision, primary weights, residual bypass, independent towers and mean
token readout.

The numeric example encoder is **not the historical feature/sequence
tokenizer**. Exact historical reproduction uses original preprocessing,
features, splits, optimizer and configurations through `research/run_expid.py`.
Portable initialization/checkpoints are not claimed identical to all historical
models. CI checks operator parity and API contracts, not industrial benchmark AUC.

Historical dynamic routers, private adapters, sample-trust gates, preference
losses, grouped heads, task tokens, custom depth gates and distillation extras
are not included in the portable API.
