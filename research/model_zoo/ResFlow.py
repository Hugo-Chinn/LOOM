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
"""ResFlow: residual information flow between task towers (KDD 2024).

Two mechanisms, both from the paper:

1. **Layer-wise residual flow.** Towers are ordered from the densest task to
   the sparsest one (default: the dataset label order, e.g. click -> cart ->
   fav -> buy) and layer ``l`` of tower ``k`` receives the activation of layer
   ``l`` of the preceding tower as an additive residual. The dense task
   therefore teaches the sparse task without sharing any tower parameter.
   ``stop_gradient=True`` cuts the gradient on that path, which is the
   privacy-preserving variant of the paper.

2. **Additive score fusion.** The sparse task's logit is the sum of its own
   tower logit and the logits of all preceding tasks. This multiplies odds,
   not probabilities, and does not guarantee monotonic probabilities.
   This scoring variant requires reference verification. ``fusion_mode=gated`` learns one
   coefficient per fusion step instead of hard-wiring the sum.
"""

import torch
from torch import nn

from unirank.pytorch.torch_utils import get_activation

from model_zoo.MTLBaseline import MTLBaseline


class ResFlow(MTLBaseline):
    """Per-task towers wired with cross-tower residual flow."""

    def __init__(self, feature_map, model_id="ResFlow",
                 tower_hidden_units=[128, 64], tower_activations="ReLU",
                 task_order=None, use_residual_flow=True, stop_gradient=False,
                 fusion_mode="additive", **kwargs):
        super(ResFlow, self).__init__(feature_map, model_id=model_id,
                                      tower_hidden_units=tower_hidden_units,
                                      tower_activations=tower_activations,
                                      **kwargs)
        self.tower_hidden_units = list(tower_hidden_units)
        if not self.tower_hidden_units:
            raise ValueError("ResFlow requires tower hidden layers")
        self.use_residual_flow = bool(use_residual_flow)
        self.stop_gradient = bool(stop_gradient)
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in {"none", "additive", "gated"}:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode}")
        if task_order is None:
            task_order = list(range(self.num_tasks))
        task_order = [int(i) for i in task_order]
        if sorted(task_order) != list(range(self.num_tasks)):
            raise ValueError("task_order must be a permutation of the task indices")
        self.task_order = task_order
        # Position of each task in the residual chain.
        self.task_position = [0] * self.num_tasks
        for position, task_idx in enumerate(self.task_order):
            self.task_position[task_idx] = position

        dim = self.repr_dim
        widths = [dim] + self.tower_hidden_units
        activations = get_activation(tower_activations, self.tower_hidden_units)
        if not isinstance(activations, list):
            activations = [activations] * len(self.tower_hidden_units)
        self.activations = nn.ModuleList(activations)
        self.tower_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(widths[layer_idx], widths[layer_idx + 1])
                for layer_idx in range(len(widths) - 1)
            ])
            for _ in range(self.num_tasks)
        ])
        self.tower_output = nn.ModuleList([
            nn.Linear(widths[-1], 1) for _ in range(self.num_tasks)
        ])
        self.tower_dropout = nn.Dropout(self.net_dropout)
        # Residual flow needs equal widths across towers, which holds by
        # construction; a scale keeps the added activation comparable to the
        # tower's own pre-activation at initialization.
        self.residual_scale = nn.ParameterList([
            nn.Parameter(torch.zeros(len(self.tower_hidden_units)))
            for _ in range(self.num_tasks)
        ])
        if self.fusion_mode == "gated":
            self.fusion_logits = nn.Parameter(
                torch.zeros(max(self.num_tasks - 1, 1)))
        else:
            self.register_parameter("fusion_logits", None)
        # ResFlow scores with explicit layer stacks, not the base MLP tower.
        self.tower = nn.ModuleList()
        self._finalize(**kwargs)

    def _init_zero_parameters(self):
        # Residual flow starts disabled: independent towers at step 0.
        # Preserve the disabled-flow initialization after reset_parameters.
        for scale in getattr(self, "residual_scale", []):
            scale.data.fill_(0.0)

    def _task_representations(self, shared):
        """Run the towers in chain order, returning one hidden state per task."""
        hidden_by_task = {}
        layer_outputs = {}
        for position, task_idx in enumerate(self.task_order):
            layers = self.tower_layers[task_idx]
            scales = self.residual_scale[task_idx]
            hidden = shared
            for layer_idx, layer in enumerate(layers):
                pre_activation = layer(hidden)
                if self.use_residual_flow and position > 0:
                    prev_task = self.task_order[position - 1]
                    residual = layer_outputs[prev_task][layer_idx]
                    if self.stop_gradient:
                        residual = residual.detach()
                    pre_activation = pre_activation + scales[layer_idx] * residual
                hidden = self.activations[layer_idx](pre_activation)
                hidden = self.tower_dropout(hidden)
                layer_outputs.setdefault(task_idx, {})[layer_idx] = hidden
            hidden_by_task[task_idx] = hidden
        return [hidden_by_task[task_idx] for task_idx in range(self.num_tasks)]

    def _task_logit(self, representation, task_idx):
        return self.tower_output[task_idx](representation)

    def forward(self, inputs):
        shared = self._shared_representation(inputs)
        task_reprs = self._task_representations(shared)
        own_logits = [self._task_logit(task_reprs[task_idx], task_idx)
                      for task_idx in range(self.num_tasks)]
        logits = list(own_logits)
        if self.fusion_mode != "none":
            running = None
            for position, task_idx in enumerate(self.task_order):
                if position == 0:
                    running = own_logits[task_idx]
                    continue
                coefficient = 1.0
                if self.fusion_logits is not None:
                    coefficient = torch.sigmoid(
                        self.fusion_logits[position - 1]).to(running.dtype)
                logits[task_idx] = own_logits[task_idx] + coefficient * running
                running = logits[task_idx]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = self.output_activation[i](
                logits[i])
        return self._extra_outputs(return_dict, shared)
