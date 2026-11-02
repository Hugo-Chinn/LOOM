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
"""STEM: shared and task-specific embeddings with all-forward-task routing.

STEM (AAAI 2024) separates *what is shared* into two axes:

1. embeddings: a shared embedding table plus task-specific embedding tables;
2. experts: a shared expert group plus one expert group per task.

The routing rule is "all forward, task-specific backward" (AFTB): every task
reads all expert groups, but reads other tasks' private experts detached.

Embedding axis: duplicating tables is not viable at the vocab sizes used here
(millions of ids per field, four tasks), so ``task_embedding=residual`` uses a
zero-initialized gated low-rank residual on the pooled backbone representation
instead. This is an explicit head-only adaptation, not task-specific embedding
tables and not an exact reproduction of the original embedding mechanism.
"""

import math
import torch
from torch import nn

from unirank.pytorch.layers import MLP_Block

from model_zoo.MTLBaseline import MTLBaseline


class GatedPrivateResidual(nn.Module):
    """Zero-initialized low-rank task-private residual with a per-dim gate."""

    def __init__(self, dim, rank, gate_init=-2.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.gate_logit = nn.Parameter(torch.full((dim,), float(gate_init)))
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, shared):
        delta = self.up(nn.functional.silu(self.down(shared)))
        return shared + torch.sigmoid(self.gate_logit).to(shared.dtype) * delta


class STEM(MTLBaseline):
    """Shared + task-specific expert groups with AFT routing."""

    def __init__(self, feature_map, model_id="STEM",
                 num_shared_experts=2, num_task_experts=2,
                 expert_hidden_units=None, expert_activation="ReLU",
                 task_embedding="residual", private_rank=16,
                 private_gate_init=-2.0, stem_fusion="aftb",
                 tower_hidden_units=[128, 64], tower_activations="ReLU",
                 **kwargs):
        super(STEM, self).__init__(feature_map, model_id=model_id,
                                   tower_hidden_units=tower_hidden_units,
                                   tower_activations=tower_activations,
                                   **kwargs)
        if num_shared_experts < 1 or num_task_experts < 1:
            raise ValueError("STEM needs at least one shared and one task expert")
        self.num_shared_experts = int(num_shared_experts)
        self.num_task_experts = int(num_task_experts)
        self.stem_fusion = str(stem_fusion).lower()
        if self.stem_fusion != "aftb":
            raise ValueError(f"Unsupported stem_fusion={stem_fusion}")
        dim = self.repr_dim
        if expert_hidden_units is None:
            expert_hidden_units = [dim]
        self.expert_hidden_units = list(expert_hidden_units)
        expert_dim = int(self.expert_hidden_units[-1])

        def expert():
            return MLP_Block(input_dim=dim,
                             hidden_units=self.expert_hidden_units,
                             hidden_activations=expert_activation,
                             output_dim=None,
                             output_activation=None,
                             dropout_rates=self.net_dropout)

        self.shared_experts = nn.ModuleList([expert()
                                             for _ in range(self.num_shared_experts)])
        self.task_experts = nn.ModuleList([
            nn.ModuleList([expert() for _ in range(self.num_task_experts)])
            for _ in range(self.num_tasks)
        ])
        self.task_gates = nn.ModuleList([
            nn.Linear(dim, self.num_shared_experts + self.num_tasks * self.num_task_experts)
            for _ in range(self.num_tasks)
        ])

        self.task_embedding = str(task_embedding).lower()
        if self.task_embedding not in {"none", "residual"}:
            raise ValueError(f"Unsupported task_embedding={task_embedding}")
        if self.task_embedding == "residual":
            if private_rank <= 0:
                raise ValueError("task_embedding=residual requires private_rank > 0")
            self.private_residuals = nn.ModuleList([
                GatedPrivateResidual(dim, int(private_rank), private_gate_init)
                for _ in range(self.num_tasks)
            ])
        else:
            self.private_residuals = None

        fused_dim = expert_dim
        self.tower = nn.ModuleList([
            MLP_Block(input_dim=fused_dim,
                      output_dim=1,
                      hidden_units=list(tower_hidden_units),
                      hidden_activations=tower_activations,
                      output_activation=None,
                      dropout_rates=self.net_dropout)
            for _ in range(self.num_tasks)
        ])
        self._finalize(**kwargs)

    def _init_zero_parameters(self):
        # Task-private residuals must start at the shared-embedding function.
        # Preserve the identity adapter after full-model initialization.
        residuals = getattr(self, "private_residuals", None)
        if residuals is None:
            return
        for residual in residuals:
            nn.init.zeros_(residual.up.weight)

    def _task_representations(self, shared):
        shared_outputs = torch.stack([expert(shared)
                                      for expert in self.shared_experts], dim=1)
        task_inputs = [
            self.private_residuals[i](shared) if self.private_residuals is not None
            else shared for i in range(self.num_tasks)
        ]
        task_outputs = [
            torch.stack([expert(task_inputs[i]) for expert in self.task_experts[i]], dim=1)
            for i in range(self.num_tasks)
        ]
        reprs = []
        for task_idx in range(self.num_tasks):
            candidates = torch.cat([shared_outputs] + [
                output if i == task_idx else output.detach()
                for i, output in enumerate(task_outputs)
            ], dim=1)
            weights = torch.softmax(self.task_gates[task_idx](task_inputs[task_idx]), dim=-1)
            reprs.append(torch.einsum("be,bed->bd", weights, candidates))
        return reprs
