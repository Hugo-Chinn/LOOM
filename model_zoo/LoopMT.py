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
"""LoopMT: one shared encoder, one refinement block, applied in a loop.

The method being evaluated by this repository. Its three defining choices:

1. **Post-refinement loop scope.** The shared encoder runs once; a single
   refinement block is then applied ``num_loops`` times with *shared*
   parameters. This is a fixed-full-backbone head study, not the historical
   RankMixer loop that replaced its interaction stack. Untied blocks are
   available as a depth-matched control.
2. **Progressive readout.** Task k reads the loop state of stage k (clamped to
   the last stage), so shallow objectives are read before the representation
   has been refined towards the deepest one.
3. **Deep supervision.** Every task is additionally supervised at all deeper
   stages. These auxiliary outputs are consumed only by :meth:`add_loss`; they
   are never part of the evaluated predictions.

Optional extras, both off by default so the default configuration is the plain
loop: a learnable residual gate per stage (``use_loop_gate``) and a
zero-initialized gated private residual per task (``private_rank``), which
starts as the identity on its input representation.
"""

import torch
from torch import nn

from model_zoo.MTLBaseline import MTLBaseline
from model_zoo.RankMixer import RankMixerBlock
from model_zoo.STEM import GatedPrivateResidual


class LoopMT(MTLBaseline):
    """Shared encoder + shared refinement block applied recursively."""

    def __init__(self, feature_map, model_id="LoopMT", bottom_type="mixer",
                 num_loops=None, refinement_layers=1, refinement_expansion=2,
                 num_preloops=0,
                 use_stage_embedding=True, stage_embedding_init="default",
                 share_loop_parameters=True, readout="progressive",
                 deep_supervision=True, auxiliary_loss_weight=0.5,
                 use_loop_gate=False, loop_gate_init=0.5,
                 private_rank=0, private_gate_init=-2.0,
                 expansion_factor=2, **kwargs):
        super(LoopMT, self).__init__(feature_map, model_id=model_id,
                                     bottom_type=bottom_type,
                                     expansion_factor=expansion_factor,
                                     auxiliary_loss_weight=auxiliary_loss_weight,
                                     **kwargs)
        if self.bottom_type not in {"mixer", "onetrans"}:
            raise ValueError("LoopMT needs a token backbone")
        self.num_loops = int(num_loops or self.num_tasks)
        if self.num_loops < 1:
            raise ValueError("num_loops must be >= 1")
        self.readout = str(readout).lower()
        if self.readout not in {"progressive", "final"}:
            raise ValueError(f"Unsupported readout={readout}")
        self.deep_supervision = bool(deep_supervision)
        self.use_loop_gate = bool(use_loop_gate)
        self.num_preloops = int(num_preloops)
        self.refinement_layers = int(refinement_layers)
        self.share_loop_parameters = bool(share_loop_parameters)
        self.stage_embedding_init = str(stage_embedding_init)
        if self.stage_embedding_init not in {"default", "zero", "small_normal"}:
            raise ValueError("Unsupported stage_embedding_init")
        if self.num_preloops < 0 or self.refinement_layers < 1:
            raise ValueError("num_preloops must be >= 0 and refinement_layers >= 1")

        mixer_num_tokens = self.num_ns_token
        if self.task_token_mode == "single":
            mixer_num_tokens += 1
        elif self.task_token_mode == "multi":
            mixer_num_tokens += self.num_tasks
        def make_block():
            return RankMixerBlock(
                input_dim=self.token_dim, num_ns_token=mixer_num_tokens,
                num_layers=self.refinement_layers, expand=refinement_expansion,
                net_dropout=self.net_dropout)
        if self.share_loop_parameters:
            self.refinement_block = make_block()
            self.refinement_blocks = None
        else:
            self.refinement_block = None
            self.refinement_blocks = nn.ModuleList([
                make_block() for _ in range(self.num_preloops + self.num_loops)
            ])

        if use_stage_embedding:
            self.stage_embedding = nn.Embedding(self.num_loops, self.token_dim)
        else:
            self.register_parameter("stage_embedding", None)
        if self.use_loop_gate:
            gate_init = min(max(float(loop_gate_init), 1e-4), 1.0 - 1e-4)
            gate_logit = torch.logit(torch.tensor(gate_init)).item()
            self.loop_gate_logits = nn.Parameter(
                torch.full((self.num_loops,), gate_logit))
            self.preloop_gate_logits = nn.Parameter(
                torch.full((self.num_preloops,), gate_logit))
        else:
            self.register_parameter("loop_gate_logits", None)
            self.register_parameter("preloop_gate_logits", None)

        if private_rank > 0:
            self.private_residuals = nn.ModuleList([
                GatedPrivateResidual(self.repr_dim, int(private_rank),
                                     private_gate_init)
                for _ in range(self.num_tasks)
            ])
        else:
            self.private_residuals = None
        self._finalize(**kwargs)

    def _init_zero_parameters(self):
        # Run after full-model initialization to preserve explicit zero inits.
        stage_embedding = getattr(self, "stage_embedding", None)
        if stage_embedding is not None and self.stage_embedding_init == "zero":
            nn.init.zeros_(stage_embedding.weight)
        elif stage_embedding is not None and self.stage_embedding_init == "small_normal":
            nn.init.normal_(stage_embedding.weight, std=0.02)
        residuals = getattr(self, "private_residuals", None)
        if residuals is None:
            return
        for residual in residuals:
            nn.init.zeros_(residual.up.weight)

    def _state_representation(self, state, task_idx):
        if self.task_token_mode == "single":
            return state[:, -1, :]
        if self.task_token_mode == "multi":
            return state[:, self.num_ns_token + task_idx, :]
        return self._pool_tokens(state)

    def _task_representation(self, state, task_idx):
        representation = self._state_representation(state, task_idx)
        if self.private_residuals is not None:
            representation = self.private_residuals[task_idx](representation)
        return representation

    def _loop_states(self, tokens):
        states = []
        hidden = tokens
        # Behavior-agnostic warm-up passes: same block, no stage conditioning,
        # not read by any task. They let the refinement start from a deeper
        # shared state without giving any task an early readout.
        for index in range(self.num_preloops):
            block = (self.refinement_block if self.share_loop_parameters
                     else self.refinement_blocks[index])
            candidate = self.activation_checkpoint(block, hidden)
            if self.preloop_gate_logits is not None:
                gate = torch.sigmoid(self.preloop_gate_logits[index]).to(candidate.dtype)
                hidden = hidden + gate * (candidate - hidden)
            else:
                hidden = candidate
        for stage in range(self.num_loops):
            conditioned = hidden
            if self.stage_embedding is not None:
                conditioned = conditioned + self.stage_embedding.weight[stage].view(
                    1, 1, -1)
            block = (self.refinement_block if self.share_loop_parameters
                     else self.refinement_blocks[self.num_preloops + stage])
            candidate = self.activation_checkpoint(block, conditioned)
            if self.loop_gate_logits is not None:
                gate = torch.sigmoid(self.loop_gate_logits[stage]).to(candidate.dtype)
                hidden = hidden + gate * (candidate - hidden)
            else:
                hidden = candidate
            states.append(hidden)
        return states

    def forward(self, inputs):
        tokens = self._token_representations(inputs)   # shared encoder, once
        states = self._loop_states(tokens)
        return_dict = {}
        labels = self.feature_map.labels
        for task_idx in range(self.num_tasks):
            stage = task_idx if self.readout == "progressive" else self.num_loops - 1
            stage = min(stage, len(states) - 1)
            logit = self._task_logit(
                self._task_representation(states[stage], task_idx), task_idx)
            return_dict["{}_pred".format(labels[task_idx])] = \
                self.output_activation[task_idx](logit)
            if self.deep_supervision:
                for later_stage in range(stage + 1, len(states)):
                    aux_logit = self._task_logit(
                        self._task_representation(states[later_stage], task_idx),
                        task_idx)
                    return_dict["__aux_{}_{}".format(task_idx, later_stage)] = \
                        self.output_activation[task_idx](aux_logit)
        return self._extra_outputs(return_dict, states[-1])
