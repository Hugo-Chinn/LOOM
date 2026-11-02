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
"""Mixture-of-Experts multi-task baselines.

``MMoE`` (Ma et al., KDD 2018)
    One pool of shared experts plus one softmax gate per task. All tasks see
    the same experts; only the mixing weights are task-specific.

``PLE`` (Tang et al., RecSys 2020)
    Multi-level Customized Gate Control: every level owns task-specific expert
    groups *and* a shared expert group. Task k's gate mixes its own experts
    (fed by task k's level input) with the shared experts (fed by the shared
    level input); a separate shared gate mixes every expert to produce the next
    shared input. The tower of task k reads the last level of its own branch.
"""

import torch
from torch import nn

from unirank.pytorch.layers import MLP_Block

from model_zoo.MTLBaseline import MTLBaseline


def _expert(input_dim, hidden_units, activation, dropout):
    return MLP_Block(input_dim=input_dim,
                     hidden_units=list(hidden_units),
                     hidden_activations=activation,
                     output_dim=None,
                     output_activation=None,
                     dropout_rates=dropout)


class MMoE(MTLBaseline):
    """Shared expert pool + per-task gates."""

    def __init__(self, feature_map, model_id="MMoE", num_experts=4,
                 expert_hidden_units=None, expert_activation="ReLU",
                 gate_hidden_units=None, gate_temperature=1.0,
                 tower_hidden_units=[128, 64], tower_activations="ReLU",
                 **kwargs):
        super(MMoE, self).__init__(feature_map, model_id=model_id, **kwargs)
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        self.num_experts = int(num_experts)
        dim = self.repr_dim
        if expert_hidden_units is None:
            expert_hidden_units = [dim]
        self.expert_hidden_units = list(expert_hidden_units)
        self.expert_output_dim = int(self.expert_hidden_units[-1])
        self.gate_temperature = float(gate_temperature)
        if self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")

        self.experts = nn.ModuleList([
            _expert(dim, self.expert_hidden_units, expert_activation, self.net_dropout)
            for _ in range(self.num_experts)
        ])
        gate_layers = list(gate_hidden_units or [])
        if gate_layers:
            self.gates = nn.ModuleList([
                MLP_Block(input_dim=dim,
                          hidden_units=gate_layers,
                          hidden_activations=expert_activation,
                          output_dim=self.num_experts,
                          output_activation=None,
                          dropout_rates=self.net_dropout)
                for _ in range(self.num_tasks)
            ])
        else:
            self.gates = nn.ModuleList([
                nn.Linear(dim, self.num_experts) for _ in range(self.num_tasks)
            ])
        # Expert towers read a mixed representation of expert-output width.
        self.tower = nn.ModuleList([
            MLP_Block(input_dim=self.expert_output_dim,
                      output_dim=1,
                      hidden_units=list(tower_hidden_units),
                      hidden_activations=tower_activations,
                      output_activation=None,
                      dropout_rates=self.net_dropout)
            for _ in range(self.num_tasks)
        ])
        self._finalize(**kwargs)

    def _task_representations(self, shared):
        expert_outputs = torch.stack([expert(shared) for expert in self.experts],
                                     dim=1)  # B x E x D
        reprs = []
        for task_idx in range(self.num_tasks):
            weights = torch.softmax(
                self.gates[task_idx](shared) / self.gate_temperature, dim=-1)
            reprs.append(torch.einsum("be,bed->bd", weights, expert_outputs))
        return reprs


class PLE(MTLBaseline):
    """Progressive Layered Extraction with customized gate control."""

    def __init__(self, feature_map, model_id="PLE", num_levels=2,
                 num_shared_experts=2, num_task_experts=2,
                 expert_hidden_units=None, expert_activation="ReLU",
                 tower_hidden_units=[128, 64], tower_activations="ReLU",
                 **kwargs):
        super(PLE, self).__init__(feature_map, model_id=model_id, **kwargs)
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        if num_shared_experts < 1 or num_task_experts < 1:
            raise ValueError("PLE needs at least one shared and one task expert")
        self.num_levels = int(num_levels)
        self.num_shared_experts = int(num_shared_experts)
        self.num_task_experts = int(num_task_experts)
        dim = self.repr_dim
        if expert_hidden_units is None:
            expert_hidden_units = [dim]
        self.expert_hidden_units = list(expert_hidden_units)
        expert_dim = int(self.expert_hidden_units[-1])
        if expert_dim != dim:
            raise ValueError(
                "PLE stacks levels, so experts must preserve the representation "
                f"dimension: got {expert_dim} for input dim {dim}")

        def expert():
            return _expert(dim, self.expert_hidden_units, expert_activation,
                           self.net_dropout)

        self.task_experts = nn.ModuleList([
            nn.ModuleList([nn.ModuleList([expert() for _ in range(self.num_task_experts)])
                           for _ in range(self.num_tasks)])
            for _ in range(self.num_levels)
        ])
        self.shared_experts = nn.ModuleList([
            nn.ModuleList([expert() for _ in range(self.num_shared_experts)])
            for _ in range(self.num_levels)
        ])
        # Task gate: own experts + shared experts. Shared gate: every expert.
        self.task_gates = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim, self.num_task_experts + self.num_shared_experts)
                           for _ in range(self.num_tasks)])
            for _ in range(self.num_levels)
        ])
        self.shared_gates = nn.ModuleList([
            nn.Linear(dim, self.num_tasks * self.num_task_experts + self.num_shared_experts)
            for _ in range(self.num_levels)
        ])
        self.tower = nn.ModuleList([
            MLP_Block(input_dim=expert_dim,
                      output_dim=1,
                      hidden_units=list(tower_hidden_units),
                      hidden_activations=tower_activations,
                      output_activation=None,
                      dropout_rates=self.net_dropout)
            for _ in range(self.num_tasks)
        ])
        self._finalize(**kwargs)

    def _task_representations(self, shared):
        task_inputs = [shared for _ in range(self.num_tasks)]
        shared_input = shared
        for level in range(self.num_levels):
            shared_outputs = [expert(shared_input)
                              for expert in self.shared_experts[level]]
            task_outputs = [[expert(task_inputs[task_idx])
                             for expert in self.task_experts[level][task_idx]]
                            for task_idx in range(self.num_tasks)]
            next_task_inputs = []
            for task_idx in range(self.num_tasks):
                candidate = torch.stack(
                    task_outputs[task_idx] + shared_outputs, dim=1)  # B x (T+S) x D
                weights = torch.softmax(
                    self.task_gates[level][task_idx](task_inputs[task_idx]), dim=-1)
                next_task_inputs.append(
                    torch.einsum("be,bed->bd", weights, candidate))
            all_outputs = []
            for task_idx in range(self.num_tasks):
                all_outputs.extend(task_outputs[task_idx])
            all_outputs.extend(shared_outputs)
            all_stack = torch.stack(all_outputs, dim=1)
            shared_weights = torch.softmax(self.shared_gates[level](shared_input),
                                           dim=-1)
            shared_input = torch.einsum("be,bed->bd", shared_weights, all_stack)
            task_inputs = next_task_inputs
        return task_inputs
