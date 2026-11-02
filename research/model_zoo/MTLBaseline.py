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
"""Shared front-end for the multi-task routing baselines.

Every baseline in this family shares the same UniRank input pipeline
(embedding -> target attention -> tokenizer -> shared bottom) and only differs
in how the shared representation is routed into per-task readouts. Keeping the
front-end in one place is what makes the comparison "same backbone, different
routing" rather than "different backbone, different routing".

Subclasses implement :meth:`_task_representations`.
"""

import logging
import torch
from torch import nn

from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import (FeatureEmbedding, MLP_Block,
                                    MultiHeadTargetAttention)
from unirank.pytorch.layers.tokenization import build_unified_tokenizer

from model_zoo.RankMixer import RankMixerBlock
from model_zoo.OneTrans import OneTransBlock

BOTTOM_TYPES = ("mixer", "onetrans", "mlp", "none")


class MTLBaseline(MultiTaskModel):
    """Multi-task model with a pluggable task-routing module.

    Args:
        bottom_type: ``mixer`` runs UniRank RankMixer blocks over the token
            sequence and mean-pools (the backbone used by the parallel arm);
            ``mlp`` mean-pools the tokens and then applies a shared MLP bottom
            (the textbook shared-bottom form); ``none`` exposes the pooled
            tokens directly and leaves all depth to the routing module.
    """

    def __init__(self,
                 feature_map,
                 model_id="MTLBaseline",
                 task=["binary_classification"],
                 gpu=-1,
                 num_tasks=4,
                 accumulation_steps=1,
                 embedding_dim=16,
                 token_dim=64,
                 num_layers=3,
                 expansion_factor=2,
                 num_ns_token=4,
                 tokenizer_type="Chunk",
                 attention_dim=None,
                 attention_dropout=0,
                 attention_activation_type="SoftMax",
                 net_dropout=0,
                 tower_hidden_units=[128, 64],
                 tower_activations="ReLU",
                 bottom_type="mixer",
                 num_heads=2,
                 dnn_activations="ReLU",
                 reduction_ratio=0.5,
                 bottom_hidden_units=[256, 128],
                 bottom_activation="ReLU",
                 task_token_mode="none",
                 task_loss_weights=None,
                 auxiliary_loss_weight=0.5,
                 **kwargs):
        super(MTLBaseline, self).__init__(feature_map,
                                         model_id=model_id,
                                         task=task,
                                         gpu=gpu,
                                         **kwargs)
        self.num_tasks = int(num_tasks)
        self.accumulation_steps = int(accumulation_steps)
        self._task_activation = task[0] if isinstance(task, list) else task
        labels = list(getattr(feature_map, "labels", []) or [])
        if labels and len(labels) != self.num_tasks:
            raise ValueError(
                f"num_tasks={self.num_tasks} does not match the dataset labels "
                f"{labels}")
        self.bottom_type = str(bottom_type).lower()
        if self.bottom_type not in BOTTOM_TYPES:
            raise ValueError(f"Unsupported bottom_type={bottom_type}")
        self.task_token_mode = str(task_token_mode).lower()
        if self.task_token_mode not in {"none", "single", "multi"}:
            raise ValueError(f"Unsupported task_token_mode={task_token_mode}")
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.net_dropout = float(net_dropout)
        self.embedding_dim = int(embedding_dim)
        self.token_dim = int(token_dim)

        # ---- item / non-item feature bookkeeping (same split as RankMixer) ----
        self.item_info_dim = 0
        self.non_item_dim = 0
        self.num_item_fields = 0
        self.num_non_item_fields = 0
        for feat, spec in feature_map.features.items():
            if feat in labels:
                continue
            if spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item", "action"]:
                self.item_info_dim += emb_dim
                self.num_item_fields += 1
            else:
                self.non_item_dim += emb_dim
                self.num_non_item_fields += 1

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        is_onetrans = self.bottom_type == "onetrans"
        input_dim = (self.non_item_dim + self.item_info_dim if is_onetrans
                     else feature_map.sum_emb_out_dim() + self.item_info_dim)
        self.num_tokenizer_fields = (self.num_non_item_fields + self.num_item_fields
                                     if is_onetrans else
                                     self.num_non_item_fields + 2 * self.num_item_fields)
        (self.unified_tokenizer_layer,
         self.num_ns_token,
         self.tokenizer_uses_field_input) = build_unified_tokenizer(
            tokenizer_type=tokenizer_type,
            input_dim=input_dim,
            field_dim=embedding_dim,
            token_dim=token_dim,
            num_tokens=num_ns_token,
            num_fields=self.num_tokenizer_fields,
        )
        self.tokenizer_type = str(tokenizer_type).strip().title()
        self.attention_layers = None if is_onetrans else MultiHeadTargetAttention(
            input_dim=self.item_info_dim,
            attention_dim=token_dim if attention_dim is None else attention_dim,
            dropout_rate=attention_dropout,
            attention_activation_type=attention_activation_type,
        )
        self.item_token_proj = (
            nn.Linear(self.item_info_dim, token_dim)
            if is_onetrans and self.item_info_dim != token_dim
            else nn.Identity() if is_onetrans else None)
        if is_onetrans and self.task_token_mode != "none":
            raise ValueError("OneTrans baseline preserves its native NS token count")

        if self.task_token_mode == "single":
            self.task_tokens = nn.Parameter(torch.zeros(1, 1, token_dim))
            mixer_num_tokens = self.num_ns_token + 1
        elif self.task_token_mode == "multi":
            self.task_tokens = nn.Parameter(torch.zeros(1, self.num_tasks, token_dim))
            mixer_num_tokens = self.num_ns_token + self.num_tasks
        else:
            self.register_parameter("task_tokens", None)
            mixer_num_tokens = self.num_ns_token

        # ---- shared bottom ----
        if self.bottom_type in {"mixer", "onetrans"}:
            self.bottom_blocks = (OneTransBlock(
                input_dim=token_dim, num_heads=num_heads, num_layers=num_layers,
                num_ns_token=self.num_ns_token, dnn_activations=dnn_activations,
                expansion_factor=expansion_factor,
                attention_activation_type=attention_activation_type,
                reduction_ratio=reduction_ratio) if is_onetrans else RankMixerBlock(
                input_dim=token_dim, num_ns_token=mixer_num_tokens,
                num_layers=num_layers, expand=expansion_factor,
                net_dropout=net_dropout))
            self.bottom_mlp = None
            self.repr_dim = int(token_dim * self.num_ns_token if is_onetrans else token_dim)
        else:
            self.bottom_blocks = None
            if self.bottom_type == "mlp":
                if not bottom_hidden_units:
                    raise ValueError("bottom_type=mlp requires bottom_hidden_units")
                self.bottom_mlp = MLP_Block(input_dim=token_dim,
                                            hidden_units=list(bottom_hidden_units),
                                            hidden_activations=bottom_activation,
                                            output_dim=None,
                                            output_activation=None,
                                            dropout_rates=net_dropout)
                self.repr_dim = int(bottom_hidden_units[-1])
            else:
                self.bottom_mlp = None
                self.repr_dim = int(token_dim)

        # ---- per-task readouts ----
        self.tower = nn.ModuleList([
            MLP_Block(input_dim=self.repr_dim,
                      output_dim=1,
                      hidden_units=list(tower_hidden_units),
                      hidden_activations=tower_activations,
                      output_activation=None,
                      dropout_rates=net_dropout)
            for _ in range(self.num_tasks)
        ])
        if task_loss_weights is None:
            self.register_buffer("task_loss_weight_tensor", None)
        else:
            weights = torch.tensor([float(w) for w in task_loss_weights],
                                   dtype=torch.float32)
            if weights.numel() != self.num_tasks:
                raise ValueError(
                    "task_loss_weights must have one weight per task: got "
                    f"{weights.numel()} for {self.num_tasks}")
            if bool((weights <= 0).any()):
                raise ValueError("task_loss_weights must be positive")
            self.register_buffer("task_loss_weight_tensor", weights)

        # Subclasses finalize only after all routing/head parameters exist.
        if type(self) is MTLBaseline:
            self._finalize(**kwargs)

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    def _maybe_enable_torch_compile(self):
        """Compile callable children, never structural expert containers."""
        if self._torch_compile_enabled or not self.enable_torch_compile:
            return

        def compile_child(parent, name, module):
            if isinstance(module, (nn.ParameterList, nn.ParameterDict)):
                return
            if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
                for key, child in list(module.named_children()):
                    compile_child(module, key, child)
                return
            if not any(p.requires_grad for p in module.parameters()):
                return
            if any(isinstance(child, nn.Embedding) for child in module.modules()):
                return
            setattr(parent, name, torch.compile(module, backend="inductor"))
            logging.info("Compiled MTL callable: %s", name)

        for name, module in list(self.named_children()):
            if name != "output_activation":
                compile_child(self, name, module)
        self._torch_compile_enabled = True

    def _finalize(self, **kwargs):
        """Expand the output activations, build the optimizers and reset weights.

        Subclasses call this once at the end of their own ``__init__`` so that
        initialization, optimizers and compilation cover the complete model.
        """
        activation = getattr(self, "_task_activation", "binary_classification")
        self.output_activation = nn.ModuleList(
            [self.get_output_activation(activation) for _ in range(self.num_tasks)])
        self.compile(kwargs.get("dense_optimizer"), kwargs.get("loss"),
                     kwargs.get("dense_learning_rate"))
        self.reset_parameters()
        self._init_zero_parameters()

    def _init_zero_parameters(self):
        """Re-apply the zero initializations destroyed by ``reset_parameters``.

        ``reset_parameters`` xavier-initialises every ``nn.Linear``, so anything
        that must start at exactly zero has to be re-zeroed afterwards.
        """
        return

    # ------------------------------------------------------------------
    # front-end
    # ------------------------------------------------------------------
    def _append_task_tokens(self, tokens):
        if self.task_tokens is None:
            return tokens
        return torch.cat([
            tokens,
            self.task_tokens.expand(tokens.size(0), -1, -1)
        ], dim=1)

    def _encode_tokens(self, inputs):
        """Embedding -> target attention -> tokenizer -> task tokens."""
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_seq_emb[:, -1, :]
        sequence_emb = item_seq_emb[:, 0:-1, :]
        seq_pooling_emb = self.attention_layers(target_emb, sequence_emb, mask)
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        feature_embeddings = torch.cat(
            [user_context_emb, target_emb, seq_pooling_emb], dim=-1)
        if self.tokenizer_uses_field_input:
            feature_embeddings = feature_embeddings.reshape(
                batch_size, self.num_tokenizer_fields, self.embedding_dim)
        tokens = self.unified_tokenizer_layer(feature_embeddings)
        return self._append_task_tokens(tokens)

    def _shared_representation(self, inputs):
        """Return the pooled shared representation ``[B, D]``."""
        if self.bottom_type == "onetrans":
            return self._pool_tokens(self._token_representations(inputs))
        tokens = self._encode_tokens(inputs)
        if self.bottom_type == "mixer":
            tokens = self.activation_checkpoint(self.bottom_blocks, tokens)
        shared = self._pool_tokens(tokens)
        if self.bottom_mlp is not None:
            shared = self.bottom_mlp(shared)
        return shared

    def _token_representations(self, inputs):
        """Return bottom output tokens ``[B, T, D]`` (for token-level models)."""
        if self.bottom_type == "onetrans":
            batch_dict, item_dict, mask = self.get_inputs(inputs)
            batch_size = mask.shape[0]
            sequence = self.embedding_layer(item_dict, flatten_emb=True).view(
                batch_size, -1, self.item_info_dim)
            s_tokens = self.item_token_proj(sequence[:, :-1, :])
            features = torch.cat([
                self.embedding_layer(batch_dict, flatten_emb=True),
                sequence[:, -1, :]], dim=-1)
            if self.tokenizer_uses_field_input:
                features = features.reshape(
                    batch_size, self.num_tokenizer_fields, self.embedding_dim)
            tokens = self.unified_tokenizer_layer(features)
            return self.activation_checkpoint(self.bottom_blocks, s_tokens, tokens, mask)
        tokens = self._encode_tokens(inputs)
        if self.bottom_blocks is not None:
            tokens = self.activation_checkpoint(self.bottom_blocks, tokens)
        return tokens

    def _pool_tokens(self, tokens):
        if self.bottom_type == "onetrans":
            return tokens.flatten(start_dim=1)
        return tokens.mean(dim=1)

    # ------------------------------------------------------------------
    # routing (subclass hook)
    # ------------------------------------------------------------------
    def _task_representations(self, shared):
        """Map the shared representation to one representation per task."""
        return [shared for _ in range(self.num_tasks)]

    def _task_logit(self, representation, task_idx):
        return self.tower[task_idx](representation)

    def _extra_outputs(self, return_dict, shared):
        """Hook for auxiliary predictions / regularizers. Keys starting with
        ``__`` are consumed by :meth:`add_loss` and never evaluated."""
        return return_dict

    # ------------------------------------------------------------------
    # model API
    # ------------------------------------------------------------------
    def forward(self, inputs):
        shared = self._shared_representation(inputs)
        task_reprs = self._task_representations(shared)
        if len(task_reprs) != self.num_tasks:
            raise ValueError(
                f"expected {self.num_tasks} task representations, got "
                f"{len(task_reprs)}")
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            logit = self._task_logit(task_reprs[i], i)
            return_dict["{}_pred".format(labels[i])] = self.output_activation[i](logit)
        return self._extra_outputs(return_dict, shared)

    def add_loss(self, return_dict, y_true):
        labels = self.feature_map.labels
        losses = [
            self.loss_fn[i](return_dict["{}_pred".format(labels[i])],
                            y_true[i], reduction="mean")
            for i in range(self.num_tasks)
        ]
        loss = torch.stack(losses)
        if self.task_loss_weight_tensor is not None:
            weight = self.task_loss_weight_tensor.to(loss.dtype)
            loss = loss * weight / weight.mean()
        loss = torch.sum(loss)

        if self.auxiliary_loss_weight > 0:
            aux_terms = []
            for key, value in return_dict.items():
                if not key.startswith("__aux_"):
                    continue
                task_idx = int(key[len("__aux_"):].split("_")[0])
                aux_terms.append(self.loss_fn[task_idx](value, y_true[task_idx],
                                                        reduction="mean"))
            if aux_terms:
                loss = loss + self.auxiliary_loss_weight * torch.mean(
                    torch.stack(aux_terms))
        reg = return_dict.get("__reg", None)
        if reg is not None:
            loss = loss + reg
        return loss
