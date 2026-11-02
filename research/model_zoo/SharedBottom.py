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
"""Shared-bottom arms, both on the *same* RankMixer backbone.

Under a fixed backbone the textbook Shared-Bottom (Caruana, 1997: one shared
bottom, one tower per task) and the project's parallel control are the same
model. The optional shared-first tower setting is a separate ablation:

``SharedBottom``
    RankMixer backbone + independent towers: the textbook shared-bottom arm.
    ``head_sharing=shared_first`` is an explicitly named optional tower ablation.

``ParallelIndependentTower``
    RankMixer backbone + fully independent towers (the control every routing
    baseline is compared against).
"""

import torch
from torch import nn

from unirank.pytorch.layers import MLP_Block

from model_zoo.MTLBaseline import MTLBaseline


class SharedBottom(MTLBaseline):
    """Textbook shared bottom, with an optional shared-first tower ablation."""

    def __init__(self, feature_map, model_id="SharedBottom",
                 bottom_type="mixer", head_sharing="independent",
                 tower_hidden_units=[128, 64], tower_activations="ReLU",
                 **kwargs):
        super(SharedBottom, self).__init__(
            feature_map, model_id=model_id, bottom_type=bottom_type,
            tower_hidden_units=tower_hidden_units,
            tower_activations=tower_activations, **kwargs)
        self.head_sharing = str(head_sharing).lower()
        if self.head_sharing == "independent":
            self._finalize(**kwargs)
            return
        if self.head_sharing != "shared_first":
            raise ValueError("head_sharing must be independent or shared_first")
        if len(tower_hidden_units) < 2:
            raise ValueError("SharedBottom needs at least two tower layers")
        if isinstance(tower_activations, list):
            first_activations = tower_activations[:1]
            tail_activations = tower_activations[1:]
        else:
            first_activations = tower_activations
            tail_activations = tower_activations
        self.shared_first_layer = MLP_Block(
            input_dim=self.repr_dim,
            output_dim=None,
            hidden_units=list(tower_hidden_units[:1]),
            hidden_activations=first_activations,
            output_activation=None,
            dropout_rates=self.net_dropout,
        )
        self.task_head_tails = nn.ModuleList([
            MLP_Block(input_dim=int(tower_hidden_units[0]),
                      output_dim=1,
                      hidden_units=list(tower_hidden_units[1:]),
                      hidden_activations=tail_activations,
                      output_activation=None,
                      dropout_rates=self.net_dropout)
            for _ in range(self.num_tasks)
        ])
        self.tower = nn.ModuleList()  # replaced by shared-first + tails
        self._finalize(**kwargs)

    def _task_logit(self, representation, task_idx):
        if self.head_sharing == "independent":
            return super()._task_logit(representation, task_idx)
        return self.task_head_tails[task_idx](
            self.shared_first_layer(representation))

    def _task_representations(self, shared):
        return [shared for _ in range(self.num_tasks)]


class ParallelIndependentTower(MTLBaseline):
    """RankMixer backbone + independent per-task towers (the control arm)."""

    def __init__(self, feature_map, model_id="ParallelIndependentTower",
                 bottom_type="mixer", **kwargs):
        super(ParallelIndependentTower, self).__init__(feature_map,
                                                      model_id=model_id,
                                                      bottom_type=bottom_type,
                                                      **kwargs)
        self._finalize(**kwargs)

    def _task_representations(self, shared):
        return [shared for _ in range(self.num_tasks)]
