# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""OneRank-style multi-task head on top of the standard RankMixer backbone.

The backbone is *not* changed: this class uses the same RankMixer bottom as the
parallel control and only replaces the multi-task module. The head has three
parts, all from OneRank (KDD'26):

1. **Task queries with mutual invisibility.** One learnable query per task
   reads the backbone token states. Task queries are masked from each other
   during that read, so no cross-task leakage happens inside the backbone
   read; they only share the backbone tokens.
2. **Candidate-aware contextualisation.** The candidate vector (pooled backbone
   state) is contextualised against the other candidates of the same request
   before scoring. The legacy in-batch pool is not request grouped and is
   disabled by default. This representation-matching implementation is not a
   verified reproduction of OneRank's generative objective.
3. **Stop-gradient cross-task reads + dynamic matching.** Tasks may read each
   other's states, but the read is gradient-detached, so relational
   information flows without negative-transfer gradients. The score is the
   inner product of the task state and a task-conditioned candidate vector,
   replacing the static MLP tower.
"""

import math
import torch
from torch import nn

from model_zoo.MTLBaseline import MTLBaseline


class OneRankHeadLayer(nn.Module):
    """Masked self-attention over task queries + cross-attention to the backbone."""

    def __init__(self, dim, num_heads=4, dropout=0.0, ffn_ratio=2):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                                    batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                                     batch_first=True)
        self.norm_self = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ffn_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ffn_ratio), dim),
        )

    def forward(self, task_states, memory, task_mask):
        hidden = self.norm_self(task_states)
        attended, _ = self.self_attention(hidden, hidden, hidden,
                                          attn_mask=task_mask, need_weights=False)
        task_states = task_states + attended
        query = self.norm_cross(task_states)
        read, _ = self.cross_attention(query, memory, memory, need_weights=False)
        task_states = task_states + read
        return task_states + self.ffn(self.norm_ffn(task_states))


class OneRank(MTLBaseline):
    """RankMixer backbone + task-query head with dynamic matching scoring."""

    def __init__(self, feature_map, model_id="OneRank", bottom_type="mixer",
                 head_num_layers=2, num_heads=2, head_num_heads=None, ffn_ratio=2,
                 candidate_pool_mode="none", exclude_self=True,
                 cross_task_detach=True, matching_scale=1.0,
                 tower_hidden_units=[128, 64], tower_activations="ReLU",
                 **kwargs):
        super(OneRank, self).__init__(feature_map, model_id=model_id,
                                      bottom_type=bottom_type,
                                      tower_hidden_units=tower_hidden_units,
                                      tower_activations=tower_activations,
                                      num_heads=num_heads,
                                      **kwargs)
        if self.bottom_type not in {"mixer", "onetrans"}:
            raise ValueError("OneRank needs a token backbone")
        self.tower = nn.ModuleList()  # scoring is dynamic matching, not an MLP
        dim = self.token_dim
        self.head_num_layers = int(head_num_layers)
        self.candidate_pool_mode = str(candidate_pool_mode).lower()
        if self.candidate_pool_mode not in {"batch", "none"}:
            raise ValueError(f"Unsupported candidate_pool_mode={candidate_pool_mode}")
        self.exclude_self = bool(exclude_self)
        self.cross_task_detach = bool(cross_task_detach)
        self.matching_scale = float(matching_scale)

        self.task_queries = nn.Parameter(torch.zeros(self.num_tasks, dim))
        nn.init.normal_(self.task_queries, std=0.02)
        self.head_layers = nn.ModuleList([
            OneRankHeadLayer(dim, int(head_num_heads or num_heads), self.net_dropout, ffn_ratio)
            for _ in range(self.head_num_layers)
        ])
        if self.candidate_pool_mode == "batch":
            self.pool_query = nn.Linear(dim, dim)
            self.pool_proj = nn.Linear(2 * dim, dim)
        else:
            self.register_parameter("pool_query", None)
            self.register_parameter("pool_proj", None)

        self.candidate_value = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(self.num_tasks)
        ])
        if self.cross_task_detach:
            self.cross_query = nn.ModuleList([
                nn.Linear(dim, dim) for _ in range(self.num_tasks)
            ])
            self.cross_proj = nn.ModuleList([
                nn.Linear(dim, dim) for _ in range(self.num_tasks)
            ])
            for module in self.cross_proj:
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        else:
            self.cross_query = None
            self.cross_proj = None
        self.matching_bias = nn.Parameter(torch.zeros(self.num_tasks))
        self.matching_temperature = nn.Parameter(torch.tensor(1.0))

        # Mutual invisibility among task queries, over the task-query block only.
        mask = torch.zeros(self.num_tasks, self.num_tasks, dtype=torch.bool)
        for row in range(self.num_tasks):
            for column in range(self.num_tasks):
                if column != row:
                    mask[row, column] = True
        self.register_buffer("task_mask", mask)
        self._finalize(**kwargs)

    def _init_zero_parameters(self):
        # Relational reads start as a no-op and are learned only if useful.
        # Preserve the no-op relational initialization after reset_parameters.
        cross_proj = getattr(self, "cross_proj", None)
        if cross_proj is None:
            return
        for module in cross_proj:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def _candidate_context(self, candidate):
        if self.candidate_pool_mode == "none":
            return candidate
        if candidate.size(0) < 2:
            return candidate  # a one-row batch has no pool to compare against
        dim = candidate.size(-1)
        query = self.pool_query(candidate)
        scores = torch.matmul(query, candidate.transpose(0, 1)) / math.sqrt(dim)
        if self.exclude_self:
            scores = scores.masked_fill(
                torch.eye(scores.size(0), dtype=torch.bool, device=scores.device),
                float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, candidate)
        return candidate + self.pool_proj(torch.cat([candidate, context], dim=-1))

    def _task_representations(self, shared):
        raise NotImplementedError("OneRank needs the backbone tokens, not only "
                                  "the pooled state")

    def forward(self, inputs):
        tokens = self._token_representations(inputs)   # RankMixer backbone output
        candidate = tokens.mean(dim=1)
        candidate = self._candidate_context(candidate)

        task_states = self.task_queries.unsqueeze(0).expand(tokens.size(0), -1, -1)
        for layer in self.head_layers:
            task_states = self.activation_checkpoint(layer, task_states, tokens,
                                                     self.task_mask)

        task_reprs = []
        for task_idx in range(self.num_tasks):
            state = task_states[:, task_idx, :]
            if self.cross_task_detach:
                others = [task_states[:, other, :].detach()
                          for other in range(self.num_tasks) if other != task_idx]
                if others:
                    others = torch.stack(others, dim=1)   # B x (K-1) x D
                    query = self.cross_query[task_idx](state).unsqueeze(1)
                    scores = torch.matmul(query, others.transpose(1, 2))
                    scores = scores / math.sqrt(state.size(-1))
                    weights = torch.softmax(scores, dim=-1)
                    relational = torch.matmul(weights, others).squeeze(1)
                    state = state + self.cross_proj[task_idx](relational)
            task_reprs.append(state)

        return_dict = {}
        labels = self.feature_map.labels
        scale = self.matching_scale * self.matching_temperature
        for i in range(self.num_tasks):
            candidate_value = self.candidate_value[i](candidate)
            logit = (task_reprs[i] * candidate_value).sum(dim=-1, keepdim=True)
            logit = logit / math.sqrt(candidate_value.size(-1)) * scale
            logit = logit + self.matching_bias[i]
            return_dict["{}_pred".format(labels[i])] = self.output_activation[i](logit)
        return self._extra_outputs(return_dict, candidate)
