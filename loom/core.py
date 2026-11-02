"""Native-style LOOM over token states; no dataset or UniRank dependency.

Mixer operators follow UniRank's MultiHeadTokenMixing and PerTokenFeedForward
(Apache-2.0). This portable API deliberately excludes historical routing extras.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import LOOMConfig


class TokenMixerBlock(nn.Module):
    """RankMixer permutation + per-token FFN, with pre-norm residuals."""

    def __init__(self, config):
        super().__init__()
        t, d = config.num_tokens, config.token_dim
        self.num_tokens, self.token_dim = t, d
        self.pad_dim = (-d) % t
        mask = (torch.arange(d + self.pad_dim) < d).expand(t, -1)
        valid = mask.reshape(t, t, -1).transpose(0, 1).reshape(-1)
        self.register_buffer("valid_index", valid.nonzero().flatten(), persistent=False)
        self.mixer_norms = nn.ModuleList([nn.LayerNorm(d)
                                        for _ in range(config.update_layers)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d)
                                      for _ in range(config.update_layers)])
        h = d * config.expansion_factor
        self.w1 = nn.ParameterList([nn.Parameter(torch.empty(t, d, h))
                                   for _ in range(config.update_layers)])
        self.w2 = nn.ParameterList([nn.Parameter(torch.empty(t, h, d))
                                   for _ in range(config.update_layers)])
        self.b1 = nn.ParameterList([nn.Parameter(torch.zeros(t, h))
                                   for _ in range(config.update_layers)])
        self.b2 = nn.ParameterList([nn.Parameter(torch.zeros(t, d))
                                   for _ in range(config.update_layers)])
        self.dropout = nn.Dropout(config.dropout)
        for parameter in list(self.w1) + list(self.w2):
            nn.init.xavier_uniform_(parameter)

    def mix(self, x):
        padded = F.pad(x, (0, self.pad_dim)) if self.pad_dim else x
        heads = torch.tensor_split(padded, self.num_tokens, dim=-1)
        mixed = torch.stack(heads, dim=1).flatten(start_dim=2).flatten(start_dim=1)
        return mixed.index_select(1, self.valid_index).reshape(
            x.shape[0], self.num_tokens, self.token_dim)

    def forward(self, x):
        for i, norm in enumerate(self.mixer_norms):
            x = x + self.mix(norm(x))
            z = self.ffn_norms[i](x)
            h = F.gelu(torch.einsum("btd,tdk->btk", z, self.w1[i]) + self.b1[i])
            x = x + torch.einsum("btk,tkd->btd", self.dropout(h), self.w2[i]) + self.b2[i]
        return x


class LOOM(nn.Module):
    """Input [B,T,D], output one logit per task plus optional auxiliary logits.

    Attach your encoder outside this module. It is run once, never per behavior.
    """

    def __init__(self, config: LOOMConfig):
        super().__init__()
        self.config = config
        self.tasks = tuple(config.tasks)
        n, d = len(self.tasks), config.token_dim
        if config.topology == "linear":
            self.parents = [[]] + [[i - 1] for i in range(1, n)]
        elif config.topology == "branch":
            self.parents = [[], [0], [0], [1, 2]]
        else:
            self.parents = [[self.tasks.index(p) for p in config.graph[t]]
                            for t in self.tasks]
        self.order = []
        while len(self.order) < n:
            ready = [i for i in range(n) if i not in self.order
                     and all(p in self.order for p in self.parents[i])]
            if not ready:
                raise ValueError("behavior graph contains a cycle")
            self.order.extend(ready)
        self.depths = [0] * n
        for i in self.order:
            self.depths[i] = max((self.depths[p] + 1 for p in self.parents[i]), default=0)
        self.descendants = []
        for i in range(n):
            found = set()
            for j in self.order:
                if i in self.parents[j] or any(p in found for p in self.parents[j]):
                    found.add(j)
            self.descendants.append(sorted(found))
        self.shared_block = TokenMixerBlock(config) if config.shared_update else None
        self.blocks = (None if config.shared_update else nn.ModuleList(
            [TokenMixerBlock(config) for _ in range(n + config.num_preloops)]))
        self.behavior_embedding = nn.Embedding(n, d) if config.behavior_conditioning else None
        self.depth_embedding = (nn.Embedding(config.num_preloops + max(self.depths) + 1, d)
                                if config.depth_conditioning else None)
        init = math.log(config.loop_gate_init / (1 - config.loop_gate_init))
        self.gate_logits = nn.Parameter(torch.full((n,), init)) if config.use_loop_gate else None
        self.preloop_gate_logits = (nn.Parameter(torch.full((config.num_preloops,), init))
                                   if config.use_loop_gate else None)
        self.fusion_logits = nn.Parameter(torch.zeros(n, n)) if config.parent_fusion == "softmax" else None
        self.branch_logits = nn.Parameter(torch.zeros(2)) if config.topology == "branch" else None
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64),
                          nn.ReLU(), nn.Linear(64, 1)) for _ in self.tasks])
        self.bypass_logits = nn.Parameter(torch.zeros(n)) if config.base_bypass_tasks else None
        weights = config.task_loss_weights or [1.0] * n
        self.register_buffer("loss_weights", torch.tensor(weights))

    def _update(self, previous, stage, depth, behavior=True):
        x = previous
        if behavior and self.behavior_embedding is not None:
            x = x + self.behavior_embedding.weight[stage][None, None, :]
        if self.depth_embedding is not None:
            x = x + self.depth_embedding.weight[depth][None, None, :]
        block = self.shared_block if self.shared_block is not None else self.blocks[stage]
        candidate = block(x)
        gates = self.gate_logits if behavior else self.preloop_gate_logits
        if gates is None:
            return candidate
        gate_index = stage if behavior else stage - len(self.tasks)
        return previous + torch.sigmoid(gates[gate_index]) * (candidate - previous)

    def _merge(self, node, states, base):
        parents = self.parents[node]
        if not parents:
            return base
        if len(parents) == 1 or self.config.parent_fusion == "first":
            return states[parents[0]]
        if self.config.topology == "branch":
            root = states[0]
            gates = torch.sigmoid(self.branch_logits)
            return root + sum(gates[k] * (states[p] - root) for k, p in enumerate(parents))
        stacked = torch.stack([states[p] for p in parents])
        if self.config.parent_fusion == "softmax":
            weights = self.fusion_logits[node, parents].softmax(0)
            return (stacked * weights[:, None, None, None]).sum(0)
        return stacked.mean(0)

    def _readout(self, state, task, base):
        rep = state.mean(1)
        if self.tasks[task] in self.config.base_bypass_tasks:
            rep = rep + self.bypass_logits[task].tanh() * base.mean(1)
        return self.heads[task](rep).squeeze(-1)

    def forward(self, tokens, return_auxiliary=None):
        if tokens.ndim != 3 or tokens.shape[1:] != (
                self.config.num_tokens, self.config.token_dim):
            raise ValueError("tokens must have shape [batch, num_tokens, token_dim]")
        n = len(self.tasks)
        base = tokens
        for i in range(self.config.num_preloops):
            base = self._update(base, n + i, i, behavior=False)
        states = [None] * n
        for node in self.order:
            states[node] = self._update(self._merge(node, states, base), node,
                                        self.config.num_preloops + self.depths[node])
        logits = torch.stack([self._readout(states[i], i, base) for i in range(n)], 1)
        if return_auxiliary is None:
            return_auxiliary = self.training
        auxiliary = {}
        if return_auxiliary and self.config.supervision == "deep_progressive":
            for task in range(n):
                for descendant in self.descendants[task]:
                    auxiliary[task, descendant] = self._readout(states[descendant], task, base)
        return {"logits": logits, "auxiliary": auxiliary}

    def loss(self, output, targets):
        """Sum of per-task means + weight * mean of auxiliary-task/stage means.

        NaN is a missing label, not a negative label. No ancestor labels are
        inferred from a positive descendant event.
        """
        if targets.shape != output["logits"].shape:
            raise ValueError("targets must have shape [batch, num_tasks]")
        observed = torch.isfinite(targets)
        if torch.isinf(targets).any() or (
                observed & ((targets < 0) | (targets > 1))).any():
            raise ValueError("labels must be in [0,1] or NaN for missing")
        if not observed.any():
            raise ValueError("batch contains no observed labels")
        primary, auxiliary = [], []
        for task in range(len(self.tasks)):
            mask = observed[:, task]
            if mask.any():
                primary.append(self.loss_weights[task] * F.binary_cross_entropy_with_logits(
                    output["logits"][mask, task], targets[mask, task]))
        for (task, _), logits in output["auxiliary"].items():
            mask = observed[:, task]
            if mask.any():
                auxiliary.append(F.binary_cross_entropy_with_logits(
                    logits[mask], targets[mask, task]))
        total = torch.stack(primary).sum()
        if auxiliary:
            total = total + self.config.auxiliary_loss_weight * torch.stack(auxiliary).mean()
        return total
