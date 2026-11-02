# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""MoME: Mixture-of-Masked-Experts (SIGIR 2024).

Instead of instantiating E independent expert networks, MoME keeps one
over-parameterized shared network and represents every expert as a *mask* over
its neurons:

* a neuron-level model mask m_k selects the sub-network used by task k;
* a weight-level expert mask w_e selects which of those neurons expert e uses.

Because a masked expert output is linear in its mask, the per-task mixture can
be computed without materialising ``B x E x H`` tensors: the task gate simply
mixes the expert masks first and the result is applied once to the shared
hidden activation.

Masks are binary at inference and trained with the Gumbel-sigmoid
straight-through estimator; ``mask_sparsity_weight`` applies the L1 pressure on
mask density used by the paper.
"""

import torch
from torch import nn

from unirank.pytorch.layers import MLP_Block

from model_zoo.MTLBaseline import MTLBaseline


def gumbel_sigmoid(logits, training, tau=1.0, hard=True):
    """Gumbel-sigmoid sample with an optional straight-through hard mask."""
    if training:
        uniform = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
        gumbel = torch.log(uniform) - torch.log(1.0 - uniform)
        probs = torch.sigmoid((logits + gumbel) / tau)
    else:
        probs = torch.sigmoid(logits / tau)
    if not hard:
        return probs
    hard_mask = (probs > 0.5).to(probs.dtype)
    return (hard_mask - probs).detach() + probs


class MoME(MTLBaseline):
    """Shared over-parameterized network with task and expert masks."""

    def __init__(self, feature_map, model_id="MoME", num_experts=4,
                 base_hidden_units=None, base_activation="ReLU",
                 mask_conditional=False, mask_temperature=1.0,
                 mask_sparsity_weight=1e-4, mask_init=1.0,
                 tower_hidden_units=[128, 64], tower_activations="ReLU",
                 **kwargs):
        super(MoME, self).__init__(feature_map, model_id=model_id,
                                   tower_hidden_units=tower_hidden_units,
                                   tower_activations=tower_activations,
                                   **kwargs)
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        self.num_experts = int(num_experts)
        self.mask_conditional = bool(mask_conditional)
        self.mask_temperature = float(mask_temperature)
        self.mask_sparsity_weight = float(mask_sparsity_weight)
        dim = self.repr_dim
        if base_hidden_units is None:
            base_hidden_units = [dim]
        self.base_hidden_units = list(base_hidden_units)
        hidden_dim = int(self.base_hidden_units[-1])
        self.hidden_dim = hidden_dim

        self.base_network = MLP_Block(input_dim=dim,
                                      hidden_units=self.base_hidden_units,
                                      hidden_activations=base_activation,
                                      output_dim=None,
                                      output_activation=None,
                                      dropout_rates=self.net_dropout)

        if self.mask_conditional:
            self.model_mask_layers = nn.ModuleList([
                nn.Linear(dim, hidden_dim) for _ in range(self.num_tasks)
            ])
            self.model_mask_logits = None
        else:
            self.model_mask_layers = None
            self.model_mask_logits = nn.Parameter(
                torch.full((self.num_tasks, hidden_dim), float(mask_init)))
        self.expert_mask_logits = nn.Parameter(
            torch.full((self.num_experts, hidden_dim), float(mask_init)))
        self.expert_gates = nn.ModuleList([
            nn.Linear(dim, self.num_experts) for _ in range(self.num_tasks)
        ])

        self.tower = nn.ModuleList([
            MLP_Block(input_dim=hidden_dim,
                      output_dim=1,
                      hidden_units=list(tower_hidden_units),
                      hidden_activations=tower_activations,
                      output_activation=None,
                      dropout_rates=self.net_dropout)
            for _ in range(self.num_tasks)
        ])
        self._finalize(**kwargs)

    def _model_mask(self, shared, task_idx):
        if self.model_mask_layers is not None:
            logits = self.model_mask_layers[task_idx](shared)
        else:
            logits = self.model_mask_logits[task_idx].expand(shared.size(0), -1)
        return gumbel_sigmoid(logits, self.training, self.mask_temperature)

    def _expert_masks(self):
        return gumbel_sigmoid(self.expert_mask_logits, self.training,
                              self.mask_temperature)  # E x H

    def _task_representations(self, shared):
        hidden = self.base_network(shared)              # B x H
        expert_masks = self._expert_masks()             # E x H
        reprs = []
        densities = []
        for task_idx in range(self.num_tasks):
            weights = torch.softmax(self.expert_gates[task_idx](shared), dim=-1)
            mixed_expert_mask = torch.einsum("be,eh->bh", weights, expert_masks)
            model_mask = self._model_mask(shared, task_idx)
            combined = mixed_expert_mask * model_mask
            reprs.append(combined * hidden)
            densities.append(combined.mean())
        self._mask_density = torch.stack(densities).mean()
        return reprs

    def _extra_outputs(self, return_dict, shared):
        density = getattr(self, "_mask_density", None)
        if self.mask_sparsity_weight > 0 and self.training and density is not None:
            return_dict["__reg"] = self.mask_sparsity_weight * density
        return return_dict
