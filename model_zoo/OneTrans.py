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

import logging
import os

import torch
from torch import nn
from torch.func import functional_call
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, ScaledDotProductAttention
from unirank.pytorch.torch_utils import get_activation, get_optimizer
from unirank.utils import not_in_whitelist
from unirank.pytorch.layers.tokenization import build_unified_tokenizer


class OneTrans(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="OneTrans",
                 task=["binary_classification"],
                 gpu=-1,
                 dnn_activations="ReLU",
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_heads=1,
                 num_tasks=4,
                 token_dim=64,
                 num_ns_token=4,
                 tokenizer_type="Auto",
                 attention_activation_type="SoftMax",
                 reduction_ratio=0.5,
                 net_dropout=0,
                 accumulation_steps=1,
                 model_type="parallel",
                 num_loops=None,
                 share_loop_parameters=True,
                 use_behavior_embedding=True,
                 supervision_mode="progressive",
                 auxiliary_loss_weight=0.5,
                 enable_loop_diagnostics=False,
                 refiner_num_layers=1,
                 refiner_gate_init=0.1,
                 refiner_mode="recurrent",
                 memory_writer_rank=8,
                 memory_gate_init=0.01,
                 paired_fair_init=False,
                 loop_topology="linear",
                 peft_type="none",
                 peft_rank=4,
                 peft_alpha=4.0,
                 peft_dropout=0.0,
                 adapter_dim=16,
                 warm_start_checkpoint=None,
                 warm_start_mode="none",
                 base_lr_scale=0.1,
                 zero_init_refiner_output=False,
                 freeze_warm_start_towers=False,
                 task_loss_weights=None,
                 distillation_loss_weight=0.0,
                 state_preservation_weight=0.0,
                 prediction_mode="state",
                 residual_logit_scale=1.0,
                 transition_loss_weight=0.0,
                 transition_margin=0.0,
                 risk_set_loss_weight=0.0,
                 pattern_pair_loss_weight=0.0,
                 pattern_no_regret_weight=0.0,
                 topology_retention_weight=0.0,
                 protected_task_indices=None,
                 protected_state_loss_weight=0.0,
                 explicit_pair_loss_weight=0.0,
                 explicit_pair_score_bins=10,
                 explicit_pair_stage_weights=None,
                 **kwargs):
        super(OneTrans, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       **kwargs)

        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.accumulation_steps = accumulation_steps
        self.model_type = str(model_type).lower()
        self.num_loops = int(num_loops or num_tasks)
        self.share_loop_parameters = bool(share_loop_parameters)
        self.use_behavior_embedding = bool(use_behavior_embedding)
        self.supervision_mode = str(supervision_mode).lower()
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.enable_loop_diagnostics = bool(enable_loop_diagnostics)
        self.refiner_num_layers = int(refiner_num_layers)
        self.refiner_gate_init = float(refiner_gate_init)
        self.refiner_mode = str(refiner_mode).lower()
        self.memory_writer_rank = int(memory_writer_rank)
        self.memory_gate_init = float(memory_gate_init)
        self.paired_fair_init = bool(paired_fair_init)
        self.initialization_seed = int(kwargs.get("seed", torch.initial_seed()))
        self.loop_topology = str(loop_topology).lower()
        self.peft_type = str(peft_type).lower()
        self.peft_rank = int(peft_rank)
        self.peft_alpha = float(peft_alpha)
        self.peft_dropout = float(peft_dropout)
        self.adapter_dim = int(adapter_dim)
        self.warm_start_checkpoint = warm_start_checkpoint
        self.warm_start_mode = str(warm_start_mode).lower()
        self.base_lr_scale = float(base_lr_scale)
        self.zero_init_refiner_output = bool(zero_init_refiner_output)
        self.freeze_warm_start_towers = bool(freeze_warm_start_towers)
        self.task_loss_weights = (
            [1.0] * self.num_tasks
            if task_loss_weights is None else [float(x) for x in task_loss_weights]
        )
        self.distillation_loss_weight = float(distillation_loss_weight)
        self.state_preservation_weight = float(state_preservation_weight)
        self.prediction_mode = str(prediction_mode).lower()
        self.residual_logit_scale = float(residual_logit_scale)
        self.transition_loss_weight = float(transition_loss_weight)
        self.transition_margin = float(transition_margin)
        self.risk_set_loss_weight = float(risk_set_loss_weight)
        self.pattern_pair_loss_weight = float(pattern_pair_loss_weight)
        self.pattern_no_regret_weight = float(pattern_no_regret_weight)
        self.topology_retention_weight = float(topology_retention_weight)
        self.protected_task_indices = (
            [] if protected_task_indices is None
            else [int(index) for index in protected_task_indices]
        )
        self.protected_state_loss_weight = float(protected_state_loss_weight)
        self.explicit_pair_loss_weight = float(explicit_pair_loss_weight)
        self.explicit_pair_score_bins = int(explicit_pair_score_bins)
        self.explicit_pair_stage_weights = (
            [1.0] * self.num_tasks
            if explicit_pair_stage_weights is None
            else [float(value) for value in explicit_pair_stage_weights]
        )
        if self.model_type not in {
                "parallel", "looped", "whole_encoder_loop", "base_refiner"}:
            raise ValueError(f"Unsupported OneTrans model_type={model_type}")
        if self.supervision_mode not in {"progressive", "final", "deep_progressive"}:
            raise ValueError(f"Unsupported supervision_mode={supervision_mode}")
        if self.loop_topology not in {
                "linear", "branch", "context_dag_mean",
                "context_dag_learned"}:
            raise ValueError(f"Unsupported loop_topology={loop_topology}")
        if self.refiner_mode not in {
                "recurrent", "independent", "anchored_scratch",
                "decoupled_scratch", "cross_only_scratch", "s_only"}:
            raise ValueError(f"Unsupported refiner_mode={refiner_mode}")
        if (self.refiner_mode != "recurrent"
                and self.loop_topology not in {
                    "linear", "context_dag_mean", "context_dag_learned"}):
            raise ValueError(
                "non-recurrent refiner modes require linear or context-DAG topology")
        if self.memory_writer_rank <= 0:
            raise ValueError("memory_writer_rank must be positive")
        if not 0.0 < self.memory_gate_init < 1.0:
            raise ValueError("memory_gate_init must be in (0, 1)")
        if (self.loop_topology in {
                "branch", "context_dag_mean", "context_dag_learned"}
                and self.num_tasks != 4):
            raise ValueError("branch/context-DAG topology requires exactly four tasks")
        if any(index < 0 or index >= self.num_tasks
               for index in self.protected_task_indices):
            raise ValueError("protected_task_indices contains an invalid task index")
        if self.protected_state_loss_weight < 0:
            raise ValueError("protected_state_loss_weight must be non-negative")
        if (self.protected_state_loss_weight > 0
                and self.model_type != "base_refiner"):
            raise ValueError(
                "protected-state supervision requires model_type=base_refiner")
        if self.peft_type not in {"none", "film", "adapter", "lora_q", "lora_qv"}:
            raise ValueError(f"Unsupported peft_type={peft_type}")
        if self.peft_rank <= 0:
            raise ValueError("peft_rank must be positive")
        if self.adapter_dim <= 0:
            raise ValueError("adapter_dim must be positive")
        if len(self.task_loss_weights) != self.num_tasks:
            raise ValueError("task_loss_weights must have one value per task")
        if any(weight < 0 for weight in self.task_loss_weights):
            raise ValueError("task_loss_weights must be non-negative")
        if self.prediction_mode not in {"state", "residual_logit"}:
            raise ValueError(f"Unsupported prediction_mode={prediction_mode}")
        if self.prediction_mode == "residual_logit" and self.model_type != "base_refiner":
            raise ValueError("residual_logit prediction requires model_type=base_refiner")
        if any(weight < 0 for weight in (
                self.transition_loss_weight,
                self.risk_set_loss_weight,
                self.pattern_pair_loss_weight,
                self.pattern_no_regret_weight,
                self.topology_retention_weight,
                self.explicit_pair_loss_weight)):
            raise ValueError("auxiliary loss weights must be non-negative")
        if ((self.pattern_pair_loss_weight > 0
             or self.pattern_no_regret_weight > 0)
                and self.model_type != "base_refiner"):
            raise ValueError("pattern-pair supervision requires base_refiner")
        if self.topology_retention_weight > 0 and self.loop_topology != "branch":
            raise ValueError("topology retention currently requires branch topology")
        if self.explicit_pair_score_bins <= 0:
            raise ValueError("explicit_pair_score_bins must be positive")
        if len(self.explicit_pair_stage_weights) != self.num_tasks:
            raise ValueError("explicit_pair_stage_weights must have one value per task")
        if any(value < 0 for value in self.explicit_pair_stage_weights):
            raise ValueError("explicit_pair_stage_weights must be non-negative")
        if self.warm_start_mode not in {"none", "frozen_base", "low_lr_base"}:
            raise ValueError(f"Unsupported warm_start_mode={warm_start_mode}")
        if self.warm_start_mode != "none" and self.model_type != "base_refiner":
            raise ValueError("warm-start modes are only supported for base_refiner")
        if self.warm_start_mode != "none" and not self.warm_start_checkpoint:
            raise ValueError("warm_start_checkpoint is required for warm-start modes")
        if not 0.0 < self.base_lr_scale <= 1.0:
            raise ValueError("base_lr_scale must be in (0, 1]")

        # Track item and non-item feature dimensions
        self.item_info_dim = 0
        self.non_item_dim = 0
        self.num_item_fields = 0
        self.num_non_item_fields = 0
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
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

        # Non-sequential feature tokenizer: generates num_ns_token NS tokens, because the target item is also regarded as an NS token, so self.item_info_dim is added
        tokenizer_input_dim = self.non_item_dim + self.item_info_dim
        self.num_tokenizer_fields = self.num_non_item_fields + self.num_item_fields
        (self.unified_tokenizer_layer,
         self.num_ns_token,
         self.tokenizer_uses_field_input) = build_unified_tokenizer(
            tokenizer_type=tokenizer_type,
            input_dim=tokenizer_input_dim,
            field_dim=embedding_dim,
            token_dim=token_dim,
            num_tokens=num_ns_token,
            num_fields=self.num_tokenizer_fields,
        )
        self.tokenizer_type = str(tokenizer_type).strip().title()

        # Project sequence and target items to token_dim
        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        # OneTrans backbone
        if self.model_type == "parallel":
            self.unified_interaction_layers = OneTransBlock(
                input_dim=token_dim, num_heads=num_heads, num_layers=num_layers,
                num_ns_token=self.num_ns_token, dnn_activations=dnn_activations,
                expansion_factor=expansion_factor,
                attention_activation_type=attention_activation_type,
                reduction_ratio=reduction_ratio)
            self.loop_block = None
            self.base_encoder = None
            self.ns_refiner = None
            self.behavior_embedding = None
            self.refiner_gate_logits = None
        elif self.model_type in {"looped", "whole_encoder_loop"}:
            # One shared OneTrans block, invoked once per progressive stage.
            self.unified_interaction_layers = None
            self.base_encoder = None
            loop_layers = num_layers if self.model_type == "whole_encoder_loop" else 1
            self.loop_block = OneTransBlock(
                input_dim=token_dim, num_heads=num_heads, num_layers=loop_layers,
                num_ns_token=self.num_ns_token, dnn_activations=dnn_activations,
                expansion_factor=expansion_factor,
                attention_activation_type=attention_activation_type,
                reduction_ratio=reduction_ratio)
            self.ns_refiner = None
            self.behavior_embedding = (
                nn.Embedding(self.num_loops, token_dim)
                if self.use_behavior_embedding else None)
            self.refiner_gate_logits = None
        else:
            # Preserve the complete OneTrans encoder, then recurrently refine
            # only the fixed-width NS state against a fixed sequence memory.
            self.unified_interaction_layers = None
            self.loop_block = None
            self.base_encoder = OneTransBlock(
                input_dim=token_dim, num_heads=num_heads, num_layers=num_layers,
                num_ns_token=self.num_ns_token, dnn_activations=dnn_activations,
                expansion_factor=expansion_factor,
                attention_activation_type=attention_activation_type,
                reduction_ratio=reduction_ratio)
            self.ns_refiner = (
                DecoupledNSRefinementBlock(
                    input_dim=token_dim,
                    num_heads=num_heads,
                    num_layers=self.refiner_num_layers,
                    num_ns_token=self.num_ns_token,
                    dnn_activations=dnn_activations,
                    expansion_factor=expansion_factor,
                    use_self_attention=(
                        self.refiner_mode == "decoupled_scratch"),
                ) if self.refiner_mode in {
                    "decoupled_scratch", "cross_only_scratch"} else
                NSRefinementBlock(
                    input_dim=token_dim,
                    num_heads=num_heads,
                    num_layers=self.refiner_num_layers,
                    num_ns_token=self.num_ns_token,
                    num_stages=self.num_loops,
                    dnn_activations=dnn_activations,
                    expansion_factor=expansion_factor,
                    peft_type=self.peft_type,
                    peft_rank=self.peft_rank,
                    peft_alpha=self.peft_alpha,
                    peft_dropout=self.peft_dropout,
                )
            )
            self.s_memory_writer = (
                LowRankSMemoryWriter(
                    input_dim=token_dim,
                    rank=self.memory_writer_rank,
                ) if self.refiner_mode == "s_only" else None
            )
            self.task_adapters = (
                nn.ModuleList([
                    TaskResidualAdapter(token_dim, self.adapter_dim, dnn_activations)
                    for _ in range(self.num_loops)
                ]) if self.peft_type == "adapter" else None
            )
            self.behavior_embedding = (
                nn.Embedding(self.num_loops, token_dim)
                if self.use_behavior_embedding else None)
            gate_init = min(max(self.refiner_gate_init, 1.0e-4), 1.0 - 1.0e-4)
            gate_logit = torch.logit(torch.tensor(gate_init, dtype=torch.float32))
            self.refiner_gate_logits = nn.Parameter(gate_logit.repeat(self.num_loops))
            if self.refiner_mode == "s_only":
                memory_gate_logit = torch.logit(torch.tensor(
                    self.memory_gate_init, dtype=torch.float32))
                self.memory_gate_logits = nn.Parameter(
                    memory_gate_logit.repeat(self.num_loops))
            else:
                self.register_parameter("memory_gate_logits", None)
        if self.model_type != "base_refiner":
            self.task_adapters = None
            self.s_memory_writer = None
            self.register_parameter("memory_gate_logits", None)
        if (self.model_type == "base_refiner"
                and self.loop_topology in {"branch", "context_dag_learned"}):
            self.branch_fusion_logits = nn.Parameter(torch.zeros(2))
        else:
            self.register_parameter("branch_fusion_logits", None)

        # Finally, only NS tokens are used for prediction.
        self.tower = nn.ModuleList([MLP_Block(input_dim=token_dim * self.num_ns_token,
                                              output_dim=1,
                                              hidden_units=tower_hidden_units,
                                              hidden_activations=tower_activations,
                                              output_activation=None,
                                              dropout_rates=net_dropout)
                                    for _ in range(num_tasks)])
        if isinstance(task, list):
            assert len(task) == num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]
            )

        self.compile(kwargs.get("dense_optimizer"), kwargs["loss"], kwargs.get("dense_learning_rate"))
        self.reset_parameters()
        if self.paired_fair_init:
            self._reset_shared_parameters_for_paired_comparison()
        if self.zero_init_refiner_output and self.ns_refiner is not None:
            self.ns_refiner.zero_init_outputs()
        if self.warm_start_mode != "none":
            self._load_parallel_warm_start(self.warm_start_checkpoint)
            self._configure_warm_start_optimizer()
        self.model_to_device()

    @staticmethod
    def _reset_module_with_seed(module, seed):
        if module is None:
            return
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            for submodule in module.modules():
                reset = type(submodule).__dict__.get("reset_parameters")
                if reset is not None:
                    reset(submodule)
            for submodule in module.modules():
                if hasattr(submodule, "init_weights"):
                    submodule.init_weights()

    def _reset_shared_parameters_for_paired_comparison(self):
        """Make common OneTrans tensors invariant to variant-only module RNG."""
        base = self.initialization_seed * 100
        self._reset_module_with_seed(self.embedding_layer, base + 1)
        self._reset_module_with_seed(self.unified_tokenizer_layer, base + 2)
        self._reset_module_with_seed(self.item_token_proj, base + 3)
        encoder = (
            self.unified_interaction_layers
            if self.model_type == "parallel" else self.base_encoder
        )
        self._reset_module_with_seed(encoder, base + 4)
        self._reset_module_with_seed(self.tower, base + 5)
        self._reset_module_with_seed(self.behavior_embedding, base + 6)

    def _warm_start_base_modules(self):
        return (
            self.embedding_layer,
            self.unified_tokenizer_layer,
            self.item_token_proj,
            self.base_encoder,
        )

    def _load_parallel_warm_start(self, checkpoint):
        checkpoint = os.path.abspath(os.path.expanduser(str(checkpoint)))
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint}")
        source = torch.load(checkpoint, map_location="cpu", weights_only=True)
        target = self.state_dict()
        mapped = {}
        unexpected = []
        for key, value in source.items():
            clean_key = key.replace("._orig_mod.", ".")
            if clean_key.startswith("unified_interaction_layers."):
                clean_key = "base_encoder." + clean_key[len("unified_interaction_layers."):]
            if clean_key in target and target[clean_key].shape == value.shape:
                mapped[clean_key] = value
            else:
                unexpected.append(key)
        required_prefixes = (
            "embedding_layer.", "unified_tokenizer_layer.",
            "item_token_proj.", "base_encoder.", "tower.",
        )
        missing_required = [
            key for key in target
            if key.startswith(required_prefixes) and key not in mapped
        ]
        if missing_required:
            raise RuntimeError(
                "Warm-start checkpoint is missing required tensors: {}"
                .format(missing_required[:10]))
        self.load_state_dict(mapped, strict=False)
        self.warm_start_audit = {
            "checkpoint": checkpoint,
            "loaded_tensors": len(mapped),
            "ignored_source_tensors": len(unexpected),
            "missing_required_tensors": len(missing_required),
        }
        logging.info("OneTrans warm-start audit: %s", self.warm_start_audit)

    def _configure_warm_start_optimizer(self):
        base_param_ids = {
            id(param)
            for module in self._warm_start_base_modules()
            for param in module.parameters()
        }
        if self.warm_start_mode == "frozen_base":
            for module in self._warm_start_base_modules():
                for param in module.parameters():
                    param.requires_grad = False
            if self.freeze_warm_start_towers:
                for param in self.tower.parameters():
                    param.requires_grad = False
            self._compile_optimizers(
                self.dense_optimizer_name, self.dense_learning_rate)
            logging.info(
                "OneTrans warm-start: frozen base representation modules; "
                "freeze_towers=%s", self.freeze_warm_start_towers)
            return

        sparse_param_ids = {
            id(param)
            for module in self.modules() if isinstance(module, nn.Embedding)
            for param in module.parameters(recurse=False) if param.requires_grad
        }
        dense_base, dense_new, sparse_base, sparse_new = [], [], [], []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            is_base = id(param) in base_param_ids
            is_sparse = id(param) in sparse_param_ids
            if is_sparse:
                (sparse_base if is_base else sparse_new).append(param)
            else:
                (dense_base if is_base else dense_new).append(param)

        dense_groups = []
        if dense_new:
            dense_groups.append({"params": dense_new, "lr": self.dense_learning_rate})
        if dense_base:
            dense_groups.append({
                "params": dense_base,
                "lr": self.dense_learning_rate * self.base_lr_scale,
            })
        sparse_groups = []
        if sparse_new:
            sparse_groups.append({"params": sparse_new, "lr": self.sparse_learning_rate})
        if sparse_base:
            sparse_groups.append({
                "params": sparse_base,
                "lr": self.sparse_learning_rate * self.base_lr_scale,
            })
        self.dense_optimizer = get_optimizer(
            self.dense_optimizer_name, dense_groups, self.dense_learning_rate,
            weight_decay=self.dense_weight_decay)
        self.sparse_optimizer = get_optimizer(
            self.sparse_optimizer_name, sparse_groups, self.sparse_learning_rate,
            weight_decay=0)
        self.optimizers = [
            optimizer for optimizer in (self.dense_optimizer, self.sparse_optimizer)
            if optimizer is not None
        ]
        self.optimizer = self.dense_optimizer or self.sparse_optimizer
        self._dense_params = dense_new + dense_base
        self._sparse_params = sparse_new + sparse_base
        logging.info(
            "OneTrans warm-start: base LR scale=%s, dense base/new=%d/%d, "
            "sparse base/new=%d/%d",
            self.base_lr_scale,
            sum(p.numel() for p in dense_base), sum(p.numel() for p in dense_new),
            sum(p.numel() for p in sparse_base), sum(p.numel() for p in sparse_new),
        )

    def _run_looped_backbone(self, s_tokens, ns_tokens, mask):
        states = []
        h_s, h_ns, h_mask = s_tokens, ns_tokens, mask
        for stage in range(self.num_loops):
            if self.behavior_embedding is not None:
                h_ns = h_ns + self.behavior_embedding.weight[stage].view(1, 1, -1)
            h_ns, h_s, h_mask = self.loop_block(
                h_s, h_ns, h_mask, return_sequence=True)
            states.append(h_ns)
        if self.enable_loop_diagnostics and len(states) > 1:
            with torch.no_grad():
                flat = [x.detach().float().flatten(1) for x in states]
                self.last_loop_diagnostics = {
                    "delta_norm": [float((flat[i] - flat[i - 1]).norm(dim=1).mean())
                                   for i in range(1, len(flat))],
                    "cosine": [float(torch.nn.functional.cosine_similarity(
                        flat[i - 1], flat[i], dim=1).mean())
                               for i in range(1, len(flat))],
                }
        return states

    def _refiner_update(self, h_ns, memory_s, memory_mask, stage):
        conditioned = h_ns
        if self.behavior_embedding is not None:
            conditioned = conditioned + self.behavior_embedding.weight[stage].view(1, 1, -1)
        candidate = self.ns_refiner(memory_s, conditioned, memory_mask, stage=stage)
        if self.task_adapters is not None:
            candidate = candidate + self.task_adapters[stage](conditioned)
        delta = candidate - conditioned
        gate = torch.sigmoid(self.refiner_gate_logits[stage]).to(delta.dtype)
        return h_ns + gate * delta

    def _contextual_independent_update(self, base_ns, context_residual,
                                       memory_s, memory_mask, stage):
        """Read parent residuals as context while keeping an independent base anchor."""
        conditioned = base_ns + context_residual
        if self.behavior_embedding is not None:
            conditioned = conditioned + self.behavior_embedding.weight[stage].view(
                1, 1, -1)
        candidate = self.ns_refiner(
            memory_s, conditioned, memory_mask, stage=stage)
        delta = candidate - conditioned
        gate = torch.sigmoid(self.refiner_gate_logits[stage]).to(delta.dtype)
        return base_ns + gate * delta

    def _anchored_scratch_update(self, base_ns, scratch_ns, memory_s,
                                 memory_mask, stage):
        """Update a recurrent correction without rewriting the base encoder path."""
        conditioned = base_ns.detach() + scratch_ns
        if self.behavior_embedding is not None:
            conditioned = conditioned + self.behavior_embedding.weight[stage].view(
                1, 1, -1)
        candidate = self.ns_refiner(
            memory_s.detach(), conditioned, memory_mask, stage=stage)
        delta = candidate - conditioned
        gate = torch.sigmoid(self.refiner_gate_logits[stage]).to(delta.dtype)
        next_scratch = scratch_ns + gate * delta
        return next_scratch, base_ns + next_scratch

    def _s_memory_update(self, memory_s, base_ns, memory_mask, stage):
        conditioned_ns = base_ns
        if self.behavior_embedding is not None:
            conditioned_ns = conditioned_ns + self.behavior_embedding.weight[
                stage].view(1, 1, -1)
        delta_s = self.s_memory_writer(
            memory_s, conditioned_ns, memory_mask)
        gate = torch.sigmoid(self.memory_gate_logits[stage]).to(delta_s.dtype)
        return memory_s + gate * delta_s

    def _run_base_refiner(self, s_tokens, ns_tokens, mask):
        base_ns, memory_s, memory_mask = self.activation_checkpoint(
            self.base_encoder, s_tokens, ns_tokens, mask, True)
        if self.loop_topology == "linear":
            states = []
            parent_states = []
            if self.refiner_mode == "recurrent":
                h_ns = base_ns
                for stage in range(self.num_loops):
                    parent_states.append(h_ns)
                    h_ns = self._refiner_update(
                        h_ns, memory_s, memory_mask, stage)
                    states.append(h_ns)
            elif self.refiner_mode == "independent":
                for stage in range(self.num_loops):
                    parent_states.append(base_ns)
                    states.append(self._refiner_update(
                        base_ns, memory_s, memory_mask, stage))
            elif self.refiner_mode in {
                    "anchored_scratch", "decoupled_scratch",
                    "cross_only_scratch"}:
                scratch_ns = torch.zeros_like(base_ns)
                parent_state = base_ns
                for stage in range(self.num_loops):
                    parent_states.append(parent_state)
                    scratch_ns, parent_state = self._anchored_scratch_update(
                        base_ns, scratch_ns, memory_s, memory_mask, stage)
                    states.append(parent_state)
            else:
                current_memory = memory_s
                for stage in range(self.num_loops):
                    parent_states.append(base_ns)
                    current_memory = self._s_memory_update(
                        current_memory, base_ns, memory_mask, stage)
                    states.append(self._refiner_update(
                        base_ns, current_memory, memory_mask, stage))
                memory_s = current_memory
        elif self.loop_topology == "branch":
            h_click = self._refiner_update(base_ns, memory_s, memory_mask, 0)
            h_cart = self._refiner_update(h_click, memory_s, memory_mask, 1)
            h_fav = self._refiner_update(h_click, memory_s, memory_mask, 2)
            fusion_gates = torch.sigmoid(self.branch_fusion_logits).to(h_click.dtype)
            h_fused = (
                h_click
                + fusion_gates[0] * (h_cart - h_click)
                + fusion_gates[1] * (h_fav - h_click)
            )
            h_buy = self._refiner_update(h_fused, memory_s, memory_mask, 3)
            states = [h_click, h_cart, h_fav, h_buy]
            parent_states = [base_ns, h_click, h_click, h_fused]
        else:
            zero_context = torch.zeros_like(base_ns)
            h_click = self._contextual_independent_update(
                base_ns, zero_context, memory_s, memory_mask, 0)
            click_residual = h_click - base_ns
            h_cart = self._contextual_independent_update(
                base_ns, click_residual, memory_s, memory_mask, 1)
            h_fav = self._contextual_independent_update(
                base_ns, click_residual, memory_s, memory_mask, 2)
            cart_residual = h_cart - base_ns
            fav_residual = h_fav - base_ns
            if self.loop_topology == "context_dag_learned":
                fusion = torch.softmax(
                    self.branch_fusion_logits, dim=0).to(base_ns.dtype)
                buy_context = (
                    fusion[0] * cart_residual + fusion[1] * fav_residual)
            else:
                buy_context = 0.5 * (cart_residual + fav_residual)
            h_buy = self._contextual_independent_update(
                base_ns, buy_context, memory_s, memory_mask, 3)
            states = [h_click, h_cart, h_fav, h_buy]
            parent_states = [base_ns, base_ns, base_ns, base_ns]
        self._record_loop_diagnostics(states)
        return states, base_ns, parent_states, memory_s, memory_mask

    def _record_loop_diagnostics(self, states):
        if not (self.enable_loop_diagnostics and len(states) > 1):
            return
        with torch.no_grad():
            flat = [x.detach().float().flatten(1) for x in states]
            self.last_loop_diagnostics = {
                "delta_norm": [float((flat[i] - flat[i - 1]).norm(dim=1).mean())
                               for i in range(1, len(flat))],
                "cosine": [float(torch.nn.functional.cosine_similarity(
                    flat[i - 1], flat[i], dim=1).mean())
                           for i in range(1, len(flat))],
            }

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]

        # Reshape flattened embeddings to B x (T+1) x item_info_dim
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_seq_emb[:, 0:-1, :]  # B x S x item_info_dim

        # S-tokens
        s_tokens = self.item_token_proj(sequence_emb)  # B x S x token_dim

        # target item as an NS token
        # Other non-sequential features -> NS tokens
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)       # B x non_item_dim
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)
        if self.tokenizer_uses_field_input:
            feature_embeddings = feature_embeddings.reshape(
                batch_size, self.num_tokenizer_fields, self.embedding_dim
            )
        unified_tokens = self.unified_tokenizer_layer(feature_embeddings)           # B x num_ns_token x token_dim

        # unified OneTrans
        if self.model_type == "parallel":
            unified_tokens = self.activation_checkpoint(
                self.unified_interaction_layers, s_tokens, unified_tokens, mask)
            states = [unified_tokens]
        elif self.model_type in {"looped", "whole_encoder_loop"}:
            states = self._run_looped_backbone(s_tokens, unified_tokens, mask)
        else:
            (states,
             base_reference,
             parent_states,
             refinement_memory,
             refinement_memory_mask) = self._run_base_refiner(
                s_tokens, unified_tokens, mask)

        # Predict from the NS tokens
        if self.supervision_mode == "final" or self.model_type == "parallel":
            prediction_states = [states[-1]] * self.num_tasks
        else:
            prediction_states = [states[min(i, len(states) - 1)]
                                 for i in range(self.num_tasks)]
        if self.prediction_mode == "residual_logit":
            tower_output = []
            transition_logits = []
            base_logits = []
            for task_idx in range(self.num_tasks):
                base_logit = self.tower[task_idx](
                    base_reference.flatten(start_dim=1))
                parent_logit = self.tower[task_idx](
                    parent_states[task_idx].flatten(start_dim=1))
                stage_logit = self.tower[task_idx](
                    states[task_idx].flatten(start_dim=1))
                delta_logit = stage_logit - parent_logit
                tower_output.append(
                    base_logit + self.residual_logit_scale * delta_logit)
                transition_logits.append(delta_logit)
                base_logits.append(base_logit)
        else:
            tower_output = [self.tower[i](prediction_states[i].flatten(start_dim=1))
                            for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        if self.prediction_mode == "residual_logit":
            return_dict["__transition_logits"] = torch.stack(
                transition_logits, dim=1)
            return_dict["__base_logits"] = torch.stack(base_logits, dim=1)
        if self.model_type == "base_refiner" and (
                self.pattern_pair_loss_weight > 0
                or self.pattern_no_regret_weight > 0):
            pair_stage_logits = []
            pair_parent_logits = []
            for task_idx in range(self.num_tasks):
                # Recompute only the local stage transition from detached
                # parent/memory. The pair auxiliary therefore updates the
                # shared refiner, stage PEFT/embedding/gate, and target tower,
                # but cannot rewrite earlier loop states or the base encoder.
                local_state = self._refiner_update(
                    parent_states[task_idx].detach(),
                    refinement_memory.detach(),
                    refinement_memory_mask,
                    task_idx,
                )
                pair_stage_logits.append(self.tower[task_idx](
                    local_state.flatten(start_dim=1)))
                pair_parent_logits.append(self.tower[task_idx](
                    parent_states[task_idx].detach().flatten(start_dim=1)).detach())
            return_dict["__pair_stage_logits"] = torch.stack(
                pair_stage_logits, dim=1)
            return_dict["__pair_parent_logits"] = torch.stack(
                pair_parent_logits, dim=1)
        if self.model_type == "base_refiner" and self.explicit_pair_loss_weight > 0:
            explicit_stage_logits = []
            explicit_parent_logits = []
            explicit_base_logits = []
            for task_idx in range(self.num_tasks):
                local_state = self._peft_only_refiner_update(
                    parent_states[task_idx],
                    refinement_memory,
                    refinement_memory_mask,
                    task_idx,
                )
                explicit_stage_logits.append(self._detached_tower_forward(
                    task_idx, local_state))
                explicit_parent_logits.append(self._detached_tower_forward(
                    task_idx, parent_states[task_idx].detach()).detach())
                explicit_base_logits.append(self._detached_tower_forward(
                    task_idx, base_reference.detach()).detach())
            return_dict["__explicit_stage_logits"] = torch.stack(
                explicit_stage_logits, dim=1)
            return_dict["__explicit_parent_logits"] = torch.stack(
                explicit_parent_logits, dim=1)
            return_dict["__explicit_base_logits"] = torch.stack(
                explicit_base_logits, dim=1)
        if self.topology_retention_weight > 0:
            retention_predictions = []
            # (state index, retained task index, local coefficient)
            retention_spec = (
                (1, 0, 0.10),  # cart state retains click
                (2, 0, 0.10),  # favorite state retains click
                (3, 0, 0.05),  # buy state weakly retains click
                (3, 1, 0.10),  # buy state retains cart
                (3, 2, 0.10),  # buy state retains favorite
            )
            for state_idx, task_idx, coefficient in retention_spec:
                prediction = self.output_activation[task_idx](
                    self.tower[task_idx](states[state_idx].flatten(start_dim=1)))
                retention_predictions.append(
                    (task_idx, coefficient, prediction))
            return_dict["__retention_predictions"] = retention_predictions
        if self.protected_state_loss_weight > 0:
            protected_predictions = []
            for task_idx in self.protected_task_indices:
                for stage_idx in range(task_idx + 1, len(states)):
                    prediction = self.output_activation[task_idx](
                        self.tower[task_idx](
                            states[stage_idx].flatten(start_dim=1)))
                    protected_predictions.append((task_idx, prediction))
            return_dict["__protected_state_predictions"] = protected_predictions
        if self.model_type == "base_refiner" and (
                self.distillation_loss_weight > 0
                or self.state_preservation_weight > 0):
            return_dict["__base_reference"] = base_reference.detach()
            return_dict["__loop_states"] = torch.stack(states, dim=1)
            if self.distillation_loss_weight > 0:
                for task_idx in range(self.num_tasks):
                    teacher_logit = self.tower[task_idx](
                        base_reference.detach().flatten(start_dim=1))
                    return_dict[f"__teacher_{task_idx}"] = self.output_activation[
                        task_idx](teacher_logit).detach()
        if self.supervision_mode == "deep_progressive" and self.model_type != "parallel":
            for task_idx in range(self.num_tasks):
                for stage_idx in range(task_idx + 1, len(states)):
                    aux = self.output_activation[task_idx](
                        self.tower[task_idx](states[stage_idx].flatten(start_dim=1)))
                    return_dict[f"__aux_{task_idx}_{stage_idx}"] = aux
        return return_dict

    def add_loss(self, return_dict, y_true):
        primary = [self.loss_fn[i](
            return_dict[f"{self.feature_map.labels[i]}_pred"], y_true[i], reduction="mean")
            for i in range(self.num_tasks)]
        weights = torch.as_tensor(
            self.task_loss_weights, device=primary[0].device,
            dtype=primary[0].dtype)
        total = (torch.stack(primary) * weights).sum()
        if self.transition_loss_weight > 0:
            transition_losses = []
            deltas = return_dict["__transition_logits"]
            for task_idx in range(self.num_tasks):
                labels = y_true[task_idx]
                signs = labels * 2.0 - 1.0
                per_sample = torch.nn.functional.softplus(
                    self.transition_margin - signs * deltas[:, task_idx])
                transition_losses.append(self._balanced_masked_mean(
                    per_sample, labels))
            total = total + self.transition_loss_weight * torch.stack(
                transition_losses).mean()
        if self.risk_set_loss_weight > 0:
            risk_losses = []
            risk_masks = [
                (y_true[0] > 0.5),
                (y_true[0] > 0.5) | (y_true[1] > 0.5),
                (y_true[0] > 0.5) | (y_true[2] > 0.5),
                (y_true[1] > 0.5) | (y_true[2] > 0.5) | (y_true[3] > 0.5),
            ]
            for task_idx in range(1, self.num_tasks):
                prediction = return_dict[
                    f"{self.feature_map.labels[task_idx]}_pred"]
                labels = y_true[task_idx]
                per_sample = torch.nn.functional.binary_cross_entropy(
                    prediction, labels, reduction="none")
                risk_losses.append(self._balanced_masked_mean(
                    per_sample, labels, risk_masks[task_idx]))
            if risk_losses:
                total = total + self.risk_set_loss_weight * torch.stack(
                    risk_losses).mean()
        if (self.pattern_pair_loss_weight > 0
                or self.pattern_no_regret_weight > 0):
            pair_loss, no_regret_loss = self._pattern_pair_losses(
                return_dict["__pair_stage_logits"],
                return_dict["__pair_parent_logits"],
                y_true,
            )
            total = total + self.pattern_pair_loss_weight * pair_loss
            total = total + self.pattern_no_regret_weight * no_regret_loss
        if self.explicit_pair_loss_weight > 0:
            explicit_loss = self._explicit_pattern_pair_loss(
                return_dict["__explicit_stage_logits"],
                return_dict["__explicit_parent_logits"],
                return_dict["__explicit_base_logits"],
                y_true,
            )
            total = total + self.explicit_pair_loss_weight * explicit_loss
        if self.topology_retention_weight > 0:
            retention = []
            for task_idx, coefficient, prediction in return_dict[
                    "__retention_predictions"]:
                retention.append(coefficient * self.loss_fn[task_idx](
                    prediction.float(), y_true[task_idx], reduction="mean"))
            if retention:
                total = total + self.topology_retention_weight * torch.stack(
                    retention).sum()
        if self.protected_state_loss_weight > 0:
            protected_losses = []
            for task_idx, prediction in return_dict[
                    "__protected_state_predictions"]:
                protected_losses.append(self.loss_fn[task_idx](
                    prediction.float(), y_true[task_idx], reduction="mean"))
            if protected_losses:
                total = total + self.protected_state_loss_weight * torch.stack(
                    protected_losses).mean()
        if self.distillation_loss_weight > 0:
            distillation = []
            for task_idx in range(self.num_tasks):
                student = return_dict[f"{self.feature_map.labels[task_idx]}_pred"]
                teacher = return_dict[f"__teacher_{task_idx}"]
                distillation.append(torch.nn.functional.binary_cross_entropy(
                    student, teacher, reduction="mean"))
            total = total + self.distillation_loss_weight * torch.stack(
                distillation).mean()
        if self.state_preservation_weight > 0:
            states = return_dict["__loop_states"]
            reference = return_dict["__base_reference"].unsqueeze(1)
            state_loss = (states - reference).float().square().mean()
            total = total + self.state_preservation_weight * state_loss
        if self.supervision_mode == "deep_progressive" and self.model_type != "parallel":
            auxiliary = []
            for task_idx in range(self.num_tasks):
                for stage_idx in range(task_idx + 1, self.num_loops):
                    key = f"__aux_{task_idx}_{stage_idx}"
                    if key in return_dict:
                        auxiliary.append(self.loss_fn[task_idx](
                            return_dict[key], y_true[task_idx], reduction="mean"))
            if auxiliary:
                total = total + self.auxiliary_loss_weight * torch.stack(auxiliary).mean()
        return total

    def _peft_only_refiner_update(self, parent, memory, memory_mask, stage):
        """Run one local stage while routing auxiliary gradients to PEFT only."""
        conditioned = parent.detach()
        if self.behavior_embedding is not None:
            conditioned = conditioned + self.behavior_embedding.weight[stage].view(
                1, 1, -1)
        parameters = {
            name: (parameter if "lora" in name else parameter.detach())
            for name, parameter in self.ns_refiner.named_parameters()
        }
        candidate = functional_call(
            self.ns_refiner,
            parameters,
            (memory.detach(), conditioned, memory_mask),
            {"stage": stage},
            strict=False,
        )
        delta = candidate - conditioned
        gate = torch.sigmoid(self.refiner_gate_logits[stage]).to(delta.dtype)
        return parent.detach() + gate * delta

    def _detached_tower_forward(self, task_idx, state):
        tower = self.tower[task_idx]
        parameters = {
            name: parameter.detach()
            for name, parameter in tower.named_parameters()
        }
        return functional_call(
            tower, parameters, (state.flatten(start_dim=1),), strict=False)

    @staticmethod
    def _balanced_masked_mean(values, labels, mask=None):
        valid = torch.ones_like(labels, dtype=torch.bool)
        if mask is not None:
            valid = valid & mask.bool()
        positive = valid & (labels > 0.5)
        negative = valid & (labels <= 0.5)
        parts = []
        if positive.any():
            parts.append(values[positive].mean())
        if negative.any():
            parts.append(values[negative].mean())
        if not parts:
            return values.sum() * 0.0
        return torch.stack(parts).mean()

    def _pattern_pair_losses(self, stage_logits, parent_logits, y_true):
        """Exact-pattern in-batch delta ranking with detached parent logits."""
        labels = torch.cat(y_true, dim=1)
        binary = (labels > 0.5).long()
        pair_losses = []
        no_regret_losses = []
        paired_positive = []
        total_positive = []
        for task_idx in range(self.num_tasks):
            other_tasks = [index for index in range(self.num_tasks)
                           if index != task_idx]
            group_code = torch.zeros(
                binary.size(0), dtype=torch.long, device=binary.device)
            for bit, other_idx in enumerate(other_tasks):
                group_code = group_code + (binary[:, other_idx] << bit)

            stage = stage_logits[:, task_idx, 0]
            parent = parent_logits[:, task_idx, 0]
            delta = stage - parent
            task_positive = binary[:, task_idx].bool()
            task_pair_losses = []
            task_regret_losses = []
            matched_count = 0
            for code in range(1 << len(other_tasks)):
                same_pattern = group_code == code
                positive = same_pattern & task_positive
                negative = same_pattern & ~task_positive
                if not (positive.any() and negative.any()):
                    continue
                matched_count += int(positive.sum().detach().cpu())
                delta_margin = delta[positive].mean() - delta[negative].mean()
                task_pair_losses.append(torch.nn.functional.softplus(-delta_margin))

                parent_margin = (
                    parent[positive].mean() - parent[negative].mean()).detach()
                stage_margin = stage[positive].mean() - stage[negative].mean()
                task_regret_losses.append(torch.nn.functional.relu(
                    parent_margin - stage_margin))
            if task_pair_losses:
                pair_losses.append(torch.stack(task_pair_losses).mean())
                no_regret_losses.append(torch.stack(task_regret_losses).mean())
            paired_positive.append(matched_count)
            total_positive.append(int(task_positive.sum().detach().cpu()))

        zero = stage_logits.sum() * 0.0
        pair_loss = torch.stack(pair_losses).mean() if pair_losses else zero
        regret_loss = (
            torch.stack(no_regret_losses).mean() if no_regret_losses else zero)
        self.last_pattern_pair_stats = {
            "paired_positive": paired_positive,
            "total_positive": total_positive,
        }
        return pair_loss, regret_loss

    def _explicit_pattern_pair_loss(
            self, stage_logits, parent_logits, base_logits, y_true):
        """One-to-one exact-label pairs matched inside detached base-score bins."""
        labels = torch.cat(y_true, dim=1)
        binary = (labels > 0.5).long()
        losses = []
        paired_positive = []
        total_positive = []
        for task_idx in range(self.num_tasks):
            other_tasks = [index for index in range(self.num_tasks)
                           if index != task_idx]
            group_code = torch.zeros(
                binary.size(0), dtype=torch.long, device=binary.device)
            for bit, other_idx in enumerate(other_tasks):
                group_code = group_code + (binary[:, other_idx] << bit)

            score = torch.sigmoid(base_logits[:, task_idx, 0].detach())
            if score.numel() > 1:
                boundaries = torch.quantile(
                    score.float(),
                    torch.linspace(
                        0, 1, self.explicit_pair_score_bins + 1,
                        device=score.device,
                    )[1:-1],
                ).unique()
                score_bin = torch.bucketize(score.float(), boundaries)
            else:
                score_bin = torch.zeros_like(group_code)
            match_code = group_code * self.explicit_pair_score_bins + score_bin

            delta = (stage_logits[:, task_idx, 0]
                     - parent_logits[:, task_idx, 0])
            positive_label = binary[:, task_idx].bool()
            task_losses = []
            matched = 0
            for code in match_code.unique(sorted=True):
                same = match_code == code
                positive_idx = torch.nonzero(
                    same & positive_label, as_tuple=False).flatten()
                negative_idx = torch.nonzero(
                    same & ~positive_label, as_tuple=False).flatten()
                if positive_idx.numel() == 0 or negative_idx.numel() == 0:
                    continue
                # Deterministic one-to-one matching. Negatives may be reused
                # only when a stratum has fewer controls than positives.
                control_idx = negative_idx[
                    torch.arange(positive_idx.numel(), device=delta.device)
                    % negative_idx.numel()]
                margins = delta[positive_idx] - delta[control_idx]
                task_losses.append(torch.nn.functional.softplus(-margins).mean())
                matched += int(positive_idx.numel())
            if task_losses and self.explicit_pair_stage_weights[task_idx] > 0:
                losses.append(
                    self.explicit_pair_stage_weights[task_idx]
                    * torch.stack(task_losses).mean())
            paired_positive.append(matched)
            total_positive.append(int(positive_label.sum().detach().cpu()))

        self.last_explicit_pair_stats = {
            "paired_positive": paired_positive,
            "total_positive": total_positive,
        }
        if not losses:
            return stage_logits.sum() * 0.0
        return torch.stack(losses).sum() / max(
            1.0, sum(self.explicit_pair_stage_weights))


class OneTransBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_heads,
                 num_layers,
                 num_ns_token,
                 dnn_activations='ReLU',
                 expansion_factor=4,
                 attention_activation_type="SoftMax",
                 reduction_ratio=0.5):
        super(OneTransBlock, self).__init__()
        self.num_layers = num_layers
        self.reduction_ratio = float(reduction_ratio)
        if not 0.0 <= self.reduction_ratio < 1.0:
            raise ValueError(
                "reduction_ratio must be in [0, 1), got {}"
                .format(reduction_ratio)
            )

        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])

        self.mha_layers = nn.ModuleList([
            MixedMHA(
                input_dim=input_dim,
                num_heads=num_heads,
                num_ns_token=num_ns_token,
                attention_activation_type=attention_activation_type,
            ) for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            MixedFFN(
                input_dim=input_dim,
                num_ns_token=num_ns_token,
                dnn_activations=dnn_activations,
                expansion_factor=expansion_factor
            ) for _ in range(num_layers)
        ])

    def forward(self, s_tokens, ns_tokens, mask=None, return_sequence=False):
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()
            q_mask = mask
        else:
            q_mask = None

        ps_tokens = s_tokens
        for i in range(self.num_layers):
            # Prune sequence queries only. NS tokens stay in the separate
            # ns_tokens stream and keep a fixed token count across all layers.
            num_sequence_tokens = ps_tokens.size(1)
            sequence_start = int(num_sequence_tokens * self.reduction_ratio)
            ps_tokens = ps_tokens[:, sequence_start:, :]
            if q_mask is not None:
                q_mask = q_mask[:, sequence_start:]
            norm_s = self.norm1_layers[i](s_tokens)
            norm_ps = self.norm1_layers[i](ps_tokens)
            norm_ns = self.norm1_layers[i](ns_tokens)

            delta_ps, delta_ns = self.mha_layers[i](norm_s, norm_ps, norm_ns, kv_mask=mask, q_mask=q_mask)
            ps_tokens = ps_tokens + delta_ps
            ns_tokens = ns_tokens + delta_ns

            norm_ps = self.norm2_layers[i](ps_tokens)
            norm_ns = self.norm2_layers[i](ns_tokens)

            delta_ps, delta_ns = self.ffn_layers[i](norm_ps, norm_ns, q_mask)
            ps_tokens = ps_tokens + delta_ps
            ns_tokens = ns_tokens + delta_ns

            s_tokens = ps_tokens
            mask = q_mask
        if return_sequence:
            return ns_tokens, ps_tokens, q_mask
        return ns_tokens


class NSRefinementBlock(nn.Module):
    """Refine fixed-width NS tokens while keeping sequence memory unchanged."""
    def __init__(self,
                 input_dim,
                 num_heads,
                 num_layers,
                 num_ns_token,
                 num_stages,
                 dnn_activations="ReLU",
                 expansion_factor=4,
                 peft_type="none",
                 peft_rank=4,
                 peft_alpha=4.0,
                 peft_dropout=0.0):
        super(NSRefinementBlock, self).__init__()
        self.num_layers = int(num_layers)
        self.peft_type = str(peft_type).lower()
        self.norm_memory_layers = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(self.num_layers)
        ])
        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(self.num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(self.num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            NSMemoryAttention(
                input_dim=input_dim,
                num_heads=num_heads,
                num_ns_token=num_ns_token,
                num_stages=num_stages,
                peft_type=self.peft_type,
                peft_rank=peft_rank,
                peft_alpha=peft_alpha,
                peft_dropout=peft_dropout,
            ) for _ in range(self.num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            NSTokenFFN(
                input_dim=input_dim,
                num_ns_token=num_ns_token,
                dnn_activations=dnn_activations,
                expansion_factor=expansion_factor,
            ) for _ in range(self.num_layers)
        ])
        if self.peft_type == "film":
            self.film_gamma = nn.Parameter(torch.zeros(
                num_stages, self.num_layers, num_ns_token, input_dim))
            self.film_beta = nn.Parameter(torch.zeros(
                num_stages, self.num_layers, num_ns_token, input_dim))
        else:
            self.register_parameter("film_gamma", None)
            self.register_parameter("film_beta", None)

    def forward(self, memory_s, ns_tokens, memory_mask=None, stage=0):
        for index in range(self.num_layers):
            norm_memory = self.norm_memory_layers[index](memory_s)
            norm_ns = self.norm1_layers[index](ns_tokens)
            if self.film_gamma is not None:
                gamma = self.film_gamma[stage, index].unsqueeze(0).to(norm_ns.dtype)
                beta = self.film_beta[stage, index].unsqueeze(0).to(norm_ns.dtype)
                norm_ns = norm_ns * (1.0 + gamma) + beta
            ns_tokens = ns_tokens + self.attention_layers[index](
                norm_memory, norm_ns, memory_mask, stage=stage)
            norm_ns = self.norm2_layers[index](ns_tokens)
            ns_tokens = ns_tokens + self.ffn_layers[index](norm_ns)
        return ns_tokens

    def zero_init_outputs(self):
        for attention in self.attention_layers:
            nn.init.zeros_(attention.W_o.weight)
        for ffn in self.ffn_layers:
            nn.init.zeros_(ffn.W2)
            nn.init.zeros_(ffn.b2)


class DecoupledNSRefinementBlock(nn.Module):
    """Read S memory first, then mix NS slots without an artificial causal order."""
    def __init__(self, input_dim, num_heads, num_layers, num_ns_token,
                 dnn_activations="ReLU", expansion_factor=4,
                 use_self_attention=True):
        super().__init__()
        self.num_layers = int(num_layers)
        self.use_self_attention = bool(use_self_attention)
        self.cross_norm_memory = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(self.num_layers)
        ])
        self.cross_norm_ns = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(self.num_layers)
        ])
        self.cross_attention = nn.ModuleList([
            NSCrossMemoryAttention(input_dim, num_heads, num_ns_token)
            for _ in range(self.num_layers)
        ])
        # Keep the same registered tensors for paired initialization. The
        # cross-only variant skips these modules in forward, isolating the
        # effect of NS self-attention without changing later FFN RNG state.
        self.self_norm_ns = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(self.num_layers)
        ])
        self.self_attention = nn.ModuleList([
            NSFullSelfAttention(input_dim, num_heads, num_ns_token)
            for _ in range(self.num_layers)
        ])
        self.ffn_norm_ns = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(self.num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            NSTokenFFN(input_dim, num_ns_token, dnn_activations,
                       expansion_factor=expansion_factor)
            for _ in range(self.num_layers)
        ])

    def forward(self, memory_s, ns_tokens, memory_mask=None, stage=0):
        del stage
        for index in range(self.num_layers):
            ns_tokens = ns_tokens + self.cross_attention[index](
                self.cross_norm_memory[index](memory_s),
                self.cross_norm_ns[index](ns_tokens),
                memory_mask,
            )
            if self.use_self_attention:
                ns_tokens = ns_tokens + self.self_attention[index](
                    self.self_norm_ns[index](ns_tokens))
            ns_tokens = ns_tokens + self.ffn_layers[index](
                self.ffn_norm_ns[index](ns_tokens))
        return ns_tokens

    def zero_init_outputs(self):
        for attention in self.cross_attention:
            nn.init.zeros_(attention.W_o.weight)
        for attention in self.self_attention:
            nn.init.zeros_(attention.W_o.weight)
        for ffn in self.ffn_layers:
            nn.init.zeros_(ffn.W2)
            nn.init.zeros_(ffn.b2)


class NSCrossMemoryAttention(nn.Module):
    """NS queries read only sequence memory."""
    def __init__(self, input_dim, num_heads, num_ns_token):
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim={input_dim} is not divisible by num_heads={num_heads}")
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.W_q_ns = nn.Parameter(torch.empty(
            num_ns_token, input_dim, input_dim))
        self.W_k_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_o = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.xavier_uniform_(self.W_q_ns)

    def forward(self, memory_s, ns_tokens, memory_mask=None):
        batch_size, memory_length, _ = memory_s.shape
        ns_length = ns_tokens.size(1)
        if memory_mask is not None:
            memory_s = memory_s * memory_mask.unsqueeze(-1).to(memory_s.dtype)
        query = torch.einsum("bld,ldh->blh", ns_tokens, self.W_q_ns)
        key = self.W_k_s(memory_s)
        value = self.W_v_s(memory_s)
        query = query.view(
            batch_size, ns_length, self.num_heads,
            self.head_dim).transpose(1, 2)
        key = key.view(
            batch_size, memory_length, self.num_heads,
            self.head_dim).transpose(1, 2)
        value = value.view(
            batch_size, memory_length, self.num_heads,
            self.head_dim).transpose(1, 2)
        attention_mask = None
        if memory_mask is not None:
            attention_mask = memory_mask.bool().view(
                batch_size, 1, 1, memory_length).expand(
                    -1, 1, ns_length, -1)
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=0.0, scale=self.head_dim ** -0.5)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, ns_length, self.input_dim)
        return self.W_o(output)


class NSFullSelfAttention(nn.Module):
    """Full self-attention among latent NS slots."""
    def __init__(self, input_dim, num_heads, num_ns_token):
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim={input_dim} is not divisible by num_heads={num_heads}")
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.W_q = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_k = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_v = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_o = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, ns_tokens):
        batch_size, ns_length, _ = ns_tokens.shape
        query = torch.einsum("bld,ldh->blh", ns_tokens, self.W_q)
        key = torch.einsum("bld,ldh->blh", ns_tokens, self.W_k)
        value = torch.einsum("bld,ldh->blh", ns_tokens, self.W_v)
        query = query.view(
            batch_size, ns_length, self.num_heads,
            self.head_dim).transpose(1, 2)
        key = key.view(
            batch_size, ns_length, self.num_heads,
            self.head_dim).transpose(1, 2)
        value = value.view(
            batch_size, ns_length, self.num_heads,
            self.head_dim).transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0,
            scale=self.head_dim ** -0.5)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, ns_length, self.input_dim)
        return self.W_o(output)


class LowRankSMemoryWriter(nn.Module):
    """Fixed-length, low-rank S update conditioned on the current feedback state."""
    def __init__(self, input_dim, rank):
        super().__init__()
        self.rank = int(rank)
        self.norm_s = nn.LayerNorm(input_dim)
        self.norm_ns = nn.LayerNorm(input_dim)
        self.q_down = nn.Linear(input_dim, self.rank, bias=False)
        self.k_down = nn.Linear(input_dim, self.rank, bias=False)
        self.v_down = nn.Linear(input_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, input_dim, bias=False)

    def forward(self, memory_s, ns_tokens, memory_mask=None):
        query = self.q_down(self.norm_s(memory_s))
        key = self.k_down(self.norm_ns(ns_tokens))
        value = self.v_down(self.norm_ns(ns_tokens))
        attention = torch.matmul(query, key.transpose(-1, -2))
        attention = torch.softmax(attention * (self.rank ** -0.5), dim=-1)
        delta = self.up(torch.matmul(attention, value))
        if memory_mask is not None:
            delta = delta * memory_mask.unsqueeze(-1).to(delta.dtype)
        return delta


class NSMemoryAttention(nn.Module):
    """NS queries attend to fixed sequence memory and causally ordered NS slots."""
    def __init__(self, input_dim, num_heads, num_ns_token, num_stages,
                 peft_type="none", peft_rank=4, peft_alpha=4.0,
                 peft_dropout=0.0):
        super(NSMemoryAttention, self).__init__()
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim={input_dim} is not divisible by num_heads={num_heads}")
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_ns_token = num_ns_token
        self.head_dim = input_dim // num_heads
        self.peft_type = str(peft_type).lower()
        self.peft_rank = int(peft_rank)
        self.peft_scale = float(peft_alpha) / self.peft_rank
        self.peft_dropout = float(peft_dropout)
        self.W_k_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_q_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_k_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_v_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_o = nn.Linear(input_dim, input_dim, bias=False)
        if self.peft_type in {"lora_q", "lora_qv"}:
            self.q_lora_A = nn.Parameter(torch.empty(
                num_stages, num_ns_token, input_dim, self.peft_rank))
            self.q_lora_B = nn.Parameter(torch.zeros(
                num_stages, num_ns_token, self.peft_rank, input_dim))
        else:
            self.register_parameter("q_lora_A", None)
            self.register_parameter("q_lora_B", None)
        if self.peft_type == "lora_qv":
            self.memory_v_lora_A = nn.Parameter(torch.empty(
                num_stages, input_dim, self.peft_rank))
            self.memory_v_lora_B = nn.Parameter(torch.zeros(
                num_stages, self.peft_rank, input_dim))
            self.ns_v_lora_A = nn.Parameter(torch.empty(
                num_stages, num_ns_token, input_dim, self.peft_rank))
            self.ns_v_lora_B = nn.Parameter(torch.zeros(
                num_stages, num_ns_token, self.peft_rank, input_dim))
        else:
            self.register_parameter("memory_v_lora_A", None)
            self.register_parameter("memory_v_lora_B", None)
            self.register_parameter("ns_v_lora_A", None)
            self.register_parameter("ns_v_lora_B", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q_ns)
        nn.init.xavier_uniform_(self.W_k_ns)
        nn.init.xavier_uniform_(self.W_v_ns)
        if self.q_lora_A is not None:
            # PEFT initialization must not perturb the RNG stream used by the
            # shared refiner, behavior embeddings, and task towers.
            with torch.random.fork_rng(devices=[]):
                nn.init.xavier_uniform_(self.q_lora_A)
            nn.init.zeros_(self.q_lora_B)
        if self.memory_v_lora_A is not None:
            with torch.random.fork_rng(devices=[]):
                nn.init.xavier_uniform_(self.memory_v_lora_A)
                nn.init.xavier_uniform_(self.ns_v_lora_A)
            nn.init.zeros_(self.memory_v_lora_B)
            nn.init.zeros_(self.ns_v_lora_B)

    def _dropout(self, inputs):
        return torch.nn.functional.dropout(
            inputs, p=self.peft_dropout, training=self.training)

    def forward(self, memory_s, ns_tokens, memory_mask=None, stage=0):
        batch_size, memory_length, _ = memory_s.shape
        ns_length = ns_tokens.size(1)
        if memory_mask is not None:
            memory_s = memory_s * memory_mask.unsqueeze(-1).to(memory_s.dtype)

        q_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_q_ns)
        k_s = self.W_k_s(memory_s)
        v_s = self.W_v_s(memory_s)
        k_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_k_ns)
        v_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_v_ns)
        if self.q_lora_A is not None:
            q_low = torch.einsum(
                "bld,ldr->blr", self._dropout(ns_tokens), self.q_lora_A[stage])
            q_ns = q_ns + self.peft_scale * torch.einsum(
                "blr,lrd->bld", q_low, self.q_lora_B[stage])
        if self.memory_v_lora_A is not None:
            memory_low = torch.einsum(
                "bld,dr->blr", self._dropout(memory_s), self.memory_v_lora_A[stage])
            v_s = v_s + self.peft_scale * torch.einsum(
                "blr,rd->bld", memory_low, self.memory_v_lora_B[stage])
            ns_low = torch.einsum(
                "bld,ldr->blr", self._dropout(ns_tokens), self.ns_v_lora_A[stage])
            v_ns = v_ns + self.peft_scale * torch.einsum(
                "blr,lrd->bld", ns_low, self.ns_v_lora_B[stage])
        key = torch.cat([k_s, k_ns], dim=1)
        value = torch.cat([v_s, v_ns], dim=1)

        q_ns = q_ns.view(
            batch_size, ns_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(
            batch_size, memory_length + ns_length,
            self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(
            batch_size, memory_length + ns_length,
            self.num_heads, self.head_dim).transpose(1, 2)

        ns_causal = torch.ones(
            ns_length, ns_length, dtype=torch.bool, device=ns_tokens.device).tril()
        memory_valid = torch.ones(
            batch_size, ns_length, memory_length,
            dtype=torch.bool, device=ns_tokens.device)
        if memory_mask is not None:
            memory_valid = memory_mask.bool().unsqueeze(1).expand(-1, ns_length, -1)
        ns_valid = ns_causal.unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = torch.cat([memory_valid, ns_valid], dim=-1).unsqueeze(1)

        output = torch.nn.functional.scaled_dot_product_attention(
            q_ns,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.head_dim ** -0.5,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, ns_length, self.input_dim)
        return self.W_o(output)


class NSTokenFFN(nn.Module):
    """Token-specific FFN for NS latent slots only."""
    def __init__(self, input_dim, num_ns_token, dnn_activations, expansion_factor=4):
        super(NSTokenFFN, self).__init__()
        hidden_dim = input_dim * expansion_factor
        self.activation = get_activation(dnn_activations)
        self.W1 = nn.Parameter(torch.empty(num_ns_token, input_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_ns_token, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(num_ns_token, hidden_dim, input_dim))
        self.b2 = nn.Parameter(torch.zeros(num_ns_token, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, ns_tokens):
        hidden = torch.einsum("bld,ldh->blh", ns_tokens, self.W1)
        hidden = self.activation(hidden + self.b1.unsqueeze(0))
        return (torch.einsum("blh,lhd->bld", hidden, self.W2)
                + self.b2.unsqueeze(0))


class TaskResidualAdapter(nn.Module):
    """Zero-initialized bottleneck adapter for one feedback stage."""
    def __init__(self, input_dim, adapter_dim, activation="ReLU"):
        super(TaskResidualAdapter, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.down_weight = nn.Parameter(torch.empty(input_dim, adapter_dim))
        self.down_bias = nn.Parameter(torch.zeros(adapter_dim))
        self.activation = get_activation(activation)
        self.up_weight = nn.Parameter(torch.zeros(adapter_dim, input_dim))
        self.up_bias = nn.Parameter(torch.zeros(input_dim))
        with torch.random.fork_rng(devices=[]):
            nn.init.xavier_uniform_(self.down_weight)

    def init_weights(self):
        nn.init.zeros_(self.up_weight)
        nn.init.zeros_(self.up_bias)

    def forward(self, inputs):
        hidden = torch.matmul(self.norm(inputs), self.down_weight) + self.down_bias
        return torch.matmul(self.activation(hidden), self.up_weight) + self.up_bias


class MixedMHA(nn.Module):
    def __init__(self, input_dim, num_heads, num_ns_token,
                 attention_activation_type="SoftMax"):
        super(MixedMHA, self).__init__()
        assert input_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(input_dim, num_heads)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_ns_token = num_ns_token
        self.head_dim = input_dim // num_heads

        # shared projections for S-tokens
        self.W_q_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v_s = nn.Linear(input_dim, input_dim, bias=False)

        # token-specific projections for NS-tokens
        # shape: [num_ns_token, input_dim, input_dim]
        self.W_q_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_k_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_v_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_o = nn.Linear(input_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(
            attention_activation_type=attention_activation_type
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q_ns)
        nn.init.xavier_uniform_(self.W_k_ns)
        nn.init.xavier_uniform_(self.W_v_ns)

    def forward(self, s_tokens, ps_tokens, ns_tokens, kv_mask=None, q_mask=None):
        """
        s_tokens:  B x Ls x D
        ps_tokens: B x qLs x D
        ns_tokens: B x Lns x D
        kv_mask:   B x Ls, valid mask for full S key/value
        q_mask:    B x qLs, valid mask for pruned S query
        Keep SDPA mask-free to preserve FlashAttention eligibility, accepting minor attention leakage.
        Torch cannot use the causal_lower_right mask, so full attention is used here to simulate pyramid reduction.
        """
        B, Ls, D = s_tokens.shape
        _, qLs, _ = ps_tokens.shape
        _, Lns, _ = ns_tokens.shape

        if kv_mask is not None:
            s_tokens = s_tokens * kv_mask.unsqueeze(-1).float()

        # shared QKV for S-tokens
        q_s = self.W_q_s(s_tokens)    # B x Ls x D
        k_s = self.W_k_s(s_tokens)    # B x Ls x D
        v_s = self.W_v_s(s_tokens)    # B x Ls x D

        # token-specific QKV for NS-tokens
        q_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_q_ns)
        k_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_k_ns)
        v_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_v_ns)

        q = torch.cat([q_s, q_ns], dim=1)   # B x (Ls+Lns) x D
        k = torch.cat([k_s, k_ns], dim=1)   # B x (Ls+Lns) x D
        v = torch.cat([v_s, v_ns], dim=1)   # B x (Ls+Lns) x D

        L = Ls + Lns

        # split heads
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # B x H x L x h
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # B x H x L x h
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        output, _ = self.dot_attention(
            q,
            k,
            v,
            scale=self.head_dim ** 0.5,
            is_causal=True,
        )

        # concat heads
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.W_o(output)

        ps_out = output[:, Ls - qLs:Ls, :]
        ns_out = output[:, Ls:, :]

        if q_mask is not None:
            ps_out = ps_out * q_mask.unsqueeze(-1).float()

        return ps_out, ns_out


class MixedFFN(nn.Module):
    def __init__(self, input_dim, num_ns_token, dnn_activations, expansion_factor=4):
        super(MixedFFN, self).__init__()
        hidden_dim = input_dim * expansion_factor
        self.num_ns_token = num_ns_token
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = get_activation(dnn_activations)

        # shared FFN for S-tokens
        self.ffn_s = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(dnn_activations),
            nn.Linear(hidden_dim, input_dim)
        )

        # token-specific FFN for NS-tokens using einsum-style parameters
        self.W1_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, hidden_dim))
        self.b1_ns = nn.Parameter(torch.zeros(num_ns_token, hidden_dim))
        self.W2_ns = nn.Parameter(torch.empty(num_ns_token, hidden_dim, input_dim))
        self.b2_ns = nn.Parameter(torch.zeros(num_ns_token, input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1_ns)
        nn.init.xavier_uniform_(self.W2_ns)
        nn.init.zeros_(self.b1_ns)
        nn.init.zeros_(self.b2_ns)

    def forward(self, s_tokens, ns_tokens, mask=None):
        """
        s_tokens:  B x Ls x D
        ns_tokens: B x Lns x D
        """
        # shared FFN for S-tokens
        s_out = self.ffn_s(s_tokens)

        # token-specific FFN for NS-tokens
        # layer 1: B x Lns x D  @  Lns x D x H -> B x Lns x H
        ns_hidden = torch.einsum("bld,ldh->blh", ns_tokens, self.W1_ns) + self.b1_ns.unsqueeze(0)
        ns_hidden = self.activation(ns_hidden)
        # layer 2: B x Lns x H  @  Lns x H x D -> B x Lns x D
        ns_out = torch.einsum("blh,lhd->bld", ns_hidden, self.W2_ns) + self.b2_ns.unsqueeze(0)

        if mask is not None:
            s_out = s_out * mask.unsqueeze(-1).float()

        return s_out, ns_out
