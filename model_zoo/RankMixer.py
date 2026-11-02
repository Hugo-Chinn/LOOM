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

import torch
from torch import nn
import torch.nn.functional as F
import math
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, MultiHeadTargetAttention, PerTokenFeedForward, MultiHeadTokenMixing
from unirank.pytorch.torch_utils import get_activation
from unirank.utils import not_in_whitelist
from unirank.pytorch.layers.tokenization import build_unified_tokenizer


class RecurrenceConditioningLinear(nn.Linear):
    """Linear excluded from UniRank's exact-type default reset traversal.

    Variant-only recurrence modules are initialized in an isolated RNG scope.
    Keeping them out of ``MultiTaskModel.reset_parameters`` prevents their
    presence from shifting the later custom reinitialization of shared sparse
    embeddings in matched-seed experiments.
    """


class RankMixer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="RankMixer",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 attention_dropout=0,
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=64,
                 attention_dim=None,
                 num_ns_token=4,
                 tokenizer_type="Chunk",
                 net_dropout=0,
                 accumulation_steps=1,
                 attention_activation_type="SoftMax",
                 model_type="parallel",
                 num_loops=None,
                 share_loop_parameters=True,
                 use_behavior_embedding=True,
                 loop_scope="refinement",
                 loop_block_layers=None,
                 supervision_mode="progressive",
                 auxiliary_loss_weight=0.5,
                 task_token_mode="none",
                 enable_loop_diagnostics=False,
                 loop_topology="linear",
                 use_loop_gate=False,
                 loop_gate_init=0.5,
                 use_depth_loop_gate=False,
                 depth_gate_init=0.5,
                 depth_gate_slope_init=0.1,
                 loop_step_reg_weight=0.0,
                 loop_step_margin=0.0,
                 loop_variance_reg_weight=0.0,
                 loop_variance_floor=0.5,
                 behavior_graph=None,
                 dag_parent_fusion="mean",
                 disable_branch_merge=False,
                 num_preloops=0,
                 use_iteration_depth_embedding=False,
                 behavior_embedding_init="default",
                 behavior_embedding_init_std=0.02,
                 loop_depth_conditioning="none",
                 depth_conditioning_dim=16,
                 behavior_adapter_dim=0,
                 behavior_embedding_permutation=None,
                 sample_trust_gate="none",
                 sample_trust_hidden_dim=16,
                 sample_trust_init=0.9,
                 sample_trust_reg_weight=0.0,
                 use_base_state_bypass=False,
                 base_bypass_init=0.5,
                 base_bypass_mode="convex",
                 base_bypass_tasks=None,
                 task_loss_weights=None,
                 child_adapter_dim=0,
                 child_adapter_nodes="multi_parent",
                 branch_parent_fusion="static",
                 head_sharing="independent",
                 head_groups=None,
                 head_adapter_dim=16,
                 preference_margin_mode="none",
                 preference_margin_weight=0.0,
                 preference_margin_start=0.0,
                 preference_margin_target=0.2,
                 preference_positive_label="buy",
                 preference_negative_mode="click_only",
                 **kwargs):
        super(RankMixer, self).__init__(feature_map,
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
        self.loop_scope = str(loop_scope).lower()
        self.loop_block_layers = (
            None if loop_block_layers is None else int(loop_block_layers)
        )
        if self.loop_block_layers is not None and self.loop_block_layers < 1:
            raise ValueError("loop_block_layers must be >= 1")
        self.supervision_mode = str(supervision_mode).lower()
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.task_token_mode = str(task_token_mode).lower()
        self.enable_loop_diagnostics = bool(enable_loop_diagnostics)
        self.loop_topology = str(loop_topology).lower()
        self.use_loop_gate = bool(use_loop_gate)
        self.use_depth_loop_gate = bool(use_depth_loop_gate)
        self.depth_gate_init = float(depth_gate_init)
        self.depth_gate_slope_init = float(depth_gate_slope_init)
        self.loop_step_reg_weight = float(loop_step_reg_weight)
        self.loop_step_margin = float(loop_step_margin)
        self.loop_variance_reg_weight = float(loop_variance_reg_weight)
        self.loop_variance_floor = float(loop_variance_floor)
        self.behavior_graph = behavior_graph
        self.dag_parent_fusion = str(dag_parent_fusion).lower()
        self.disable_branch_merge = bool(disable_branch_merge)
        self.num_preloops = int(num_preloops)
        self.use_iteration_depth_embedding = bool(use_iteration_depth_embedding)
        self.behavior_embedding_init = str(behavior_embedding_init).lower()
        self.behavior_embedding_init_std = float(behavior_embedding_init_std)
        self.loop_depth_conditioning = str(loop_depth_conditioning).lower()
        self.depth_conditioning_dim = int(depth_conditioning_dim)
        self.behavior_adapter_dim = int(behavior_adapter_dim)
        self.behavior_embedding_permutation = behavior_embedding_permutation
        self.sample_trust_gate = str(sample_trust_gate).lower()
        self.sample_trust_hidden_dim = int(sample_trust_hidden_dim)
        self.sample_trust_init = float(sample_trust_init)
        self.sample_trust_reg_weight = float(sample_trust_reg_weight)
        self.use_base_state_bypass = bool(use_base_state_bypass)
        self.base_bypass_mode = str(base_bypass_mode).lower()
        self.base_bypass_tasks = base_bypass_tasks
        self.task_loss_weights = task_loss_weights
        self.child_adapter_dim = int(child_adapter_dim)
        self.child_adapter_nodes = str(child_adapter_nodes).lower()
        if self.child_adapter_nodes not in {"multi_parent", "all"}:
            raise ValueError(
                "child_adapter_nodes must be 'multi_parent' or 'all', got "
                f"{child_adapter_nodes}"
            )
        self.branch_parent_fusion = str(branch_parent_fusion).lower()
        self.head_sharing = str(head_sharing).lower()
        self.head_groups = head_groups
        self.head_adapter_dim = int(head_adapter_dim)
        self.preference_margin_mode = str(preference_margin_mode).lower()
        self.preference_margin_weight = float(preference_margin_weight)
        self.preference_margin_start = float(preference_margin_start)
        self.preference_margin_target = float(preference_margin_target)
        self.preference_positive_label = str(preference_positive_label)
        self.preference_negative_mode = str(preference_negative_mode).lower()
        if self.model_type not in {"parallel", "sequential", "looped"}:
            raise ValueError(f"Unsupported model_type={model_type}")
        if self.loop_scope not in {"refinement", "whole_encoder"}:
            raise ValueError(f"Unsupported loop_scope={loop_scope}")
        if self.supervision_mode not in {"progressive", "final", "deep_progressive"}:
            raise ValueError(f"Unsupported supervision_mode={supervision_mode}")
        if self.task_token_mode not in {"none", "single", "multi"}:
            raise ValueError(f"Unsupported task_token_mode={task_token_mode}")
        if self.loop_topology not in {"linear", "branch", "dag"}:
            raise ValueError(f"Unsupported loop_topology={loop_topology}")
        if self.dag_parent_fusion not in {
                "mean", "softmax", "dynamic_softmax", "dynamic_sigmoid"}:
            raise ValueError(f"Unsupported dag_parent_fusion={dag_parent_fusion}")
        if self.num_preloops < 0:
            raise ValueError("num_preloops must be non-negative")
        if self.behavior_embedding_init not in {
                "default", "zero", "small_normal", "scaled_default"}:
            raise ValueError(
                f"Unsupported behavior_embedding_init={behavior_embedding_init}"
            )
        if self.behavior_embedding_init_std < 0:
            raise ValueError("behavior_embedding_init_std must be non-negative")
        if self.loop_depth_conditioning not in {
                "none", "channel_scale", "sinusoidal", "behavior_sinusoidal"}:
            raise ValueError(
                f"Unsupported loop_depth_conditioning={loop_depth_conditioning}"
            )
        if self.depth_conditioning_dim <= 0:
            raise ValueError("depth_conditioning_dim must be positive")
        if self.behavior_adapter_dim < 0:
            raise ValueError("behavior_adapter_dim must be non-negative")
        if self.sample_trust_gate not in {"none", "stage", "shared"}:
            raise ValueError(f"Unsupported sample_trust_gate={sample_trust_gate}")
        if self.sample_trust_hidden_dim <= 0:
            raise ValueError("sample_trust_hidden_dim must be positive")
        if not 0.0 < self.sample_trust_init < 1.0:
            raise ValueError("sample_trust_init must lie strictly between 0 and 1")
        if self.sample_trust_reg_weight < 0:
            raise ValueError("sample_trust_reg_weight must be non-negative")
        if self.use_loop_gate and self.use_depth_loop_gate:
            raise ValueError("use_loop_gate and use_depth_loop_gate are mutually exclusive")
        if self.depth_gate_slope_init <= 0:
            raise ValueError("depth_gate_slope_init must be positive")
        if self.child_adapter_dim < 0:
            raise ValueError("child_adapter_dim must be non-negative")
        if self.base_bypass_mode not in {"convex", "residual"}:
            raise ValueError(f"Unsupported base_bypass_mode={base_bypass_mode}")
        if self.branch_parent_fusion not in {"static", "dynamic"}:
            raise ValueError(f"Unsupported branch_parent_fusion={branch_parent_fusion}")
        if self.branch_parent_fusion == "dynamic" and self.loop_topology != "branch":
            raise ValueError("branch_parent_fusion=dynamic requires loop_topology=branch")
        if self.head_sharing not in {
                "independent", "shared_first", "grouped", "grouped_first",
                "grouped_trunk", "grouped_trunk_adapter", "shared_trunk", "shared"}:
            raise ValueError(f"Unsupported head_sharing={head_sharing}")
        if self.head_sharing == "shared_first" and not tower_hidden_units:
            raise ValueError("head_sharing=shared_first requires at least one tower hidden layer")
        if self.head_sharing.startswith("grouped"):
            if not isinstance(head_groups, list) or len(head_groups) != num_tasks:
                raise ValueError(
                    "grouped head sharing requires head_groups with one group id per task"
                )
        if self.head_sharing in {"grouped_first", "grouped_trunk",
                                 "grouped_trunk_adapter"} and not tower_hidden_units:
            raise ValueError(f"head_sharing={self.head_sharing} requires hidden layers")
        if self.head_adapter_dim <= 0:
            raise ValueError("head_adapter_dim must be positive")
        if self.preference_margin_mode not in {
                "none", "fixed", "progressive", "reverse_progressive"}:
            raise ValueError(
                f"Unsupported preference_margin_mode={preference_margin_mode}"
            )
        if self.preference_margin_weight < 0:
            raise ValueError("preference_margin_weight must be non-negative")
        if self.preference_margin_start < 0 or self.preference_margin_target < 0:
            raise ValueError("preference margins must be non-negative")
        if self.preference_negative_mode not in {"click_only", "engaged_non_buy"}:
            raise ValueError(
                f"Unsupported preference_negative_mode={preference_negative_mode}"
            )
        if (self.preference_margin_mode != "none"
                and self.model_type != "looped"):
            raise ValueError("preference margin supervision requires model_type=looped")
        if self.loop_topology == "branch" and self.num_tasks != 4:
            raise ValueError("branch topology currently requires four tasks: click/cart/fav/buy")
        self.graph_parents = None
        self.graph_topological_order = None
        self.graph_descendants = None
        self.graph_depths = None
        if self.loop_topology == "dag":
            self._initialize_behavior_graph(behavior_graph)

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
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim
        self.num_tokenizer_fields = self.num_non_item_fields + 2 * self.num_item_fields
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

        if self.task_token_mode == "single":
            self.task_tokens = nn.Parameter(torch.zeros(1, 1, token_dim))
            mixer_num_tokens = self.num_ns_token + 1
        elif self.task_token_mode == "multi":
            self.task_tokens = nn.Parameter(torch.zeros(1, num_tasks, token_dim))
            mixer_num_tokens = self.num_ns_token + num_tasks
        else:
            self.register_parameter("task_tokens", None)
            mixer_num_tokens = self.num_ns_token

        self.attention_layers = MultiHeadTargetAttention(
            input_dim=self.item_info_dim,
            attention_dim=token_dim if attention_dim is None else attention_dim,
            dropout_rate=attention_dropout,
            attention_activation_type=attention_activation_type,
        )
        if self.model_type == "parallel":
            self.unified_interaction_layers = RankMixerBlock(
                input_dim=token_dim, num_ns_token=mixer_num_tokens,
                num_layers=num_layers, expand=expansion_factor,
                net_dropout=net_dropout)
            self.loop_block = None
            self.loop_blocks = None
        else:
            block_layers = self._resolve_loop_block_layers(
                num_layers, num_layers if self.loop_scope == "whole_encoder" else 1
            )
            if self.model_type == "looped" and self.share_loop_parameters:
                # A single registered module is invoked repeatedly. There is no
                # per-stage copy of its parameters.
                self.loop_block = RankMixerBlock(
                    input_dim=token_dim, num_ns_token=mixer_num_tokens,
                    num_layers=block_layers, expand=expansion_factor,
                    net_dropout=net_dropout)
                self.loop_blocks = None
            else:
                if self.model_type == "sequential" and self.share_loop_parameters:
                    raise ValueError(
                        "model_type=sequential requires share_loop_parameters=false; "
                        "use model_type=looped for exact parameter sharing."
                    )
                self.loop_block = None
                self.loop_blocks = nn.ModuleList([
                    RankMixerBlock(
                        input_dim=token_dim, num_ns_token=mixer_num_tokens,
                        num_layers=block_layers, expand=expansion_factor,
                        net_dropout=net_dropout)
                    for _ in range(self.num_loops + self.num_preloops)
                ])
            self.unified_interaction_layers = None

        self.behavior_embedding = (
            nn.Embedding(self.num_loops, token_dim)
            if self.model_type != "parallel" and self.use_behavior_embedding
            else None
        )
        if self.model_type != "parallel" and self.use_iteration_depth_embedding:
            max_behavior_depth = self._max_behavior_depth()
            self.iteration_depth_embedding = nn.Embedding(
                self.num_preloops + max_behavior_depth + 1, token_dim
            )
        else:
            self.iteration_depth_embedding = None
        if self.model_type == "looped" and self.use_depth_loop_gate:
            gate_init = min(max(self.depth_gate_init, 1e-4), 1.0 - 1e-4)
            self.depth_gate_base_logit = nn.Parameter(
                torch.logit(torch.tensor(gate_init, dtype=torch.float32))
            )
            slope = torch.tensor(self.depth_gate_slope_init, dtype=torch.float32)
            self.depth_gate_slope_raw = nn.Parameter(torch.log(torch.expm1(slope)))
        else:
            self.register_parameter("depth_gate_base_logit", None)
            self.register_parameter("depth_gate_slope_raw", None)
        if self.model_type == "looped" and self.use_loop_gate:
            gate_init = min(max(float(loop_gate_init), 1e-4), 1.0 - 1e-4)
            gate_logit = torch.logit(torch.tensor(gate_init, dtype=torch.float32))
            self.loop_gate_logits = nn.Parameter(gate_logit.repeat(self.num_loops))
        else:
            self.register_parameter("loop_gate_logits", None)
        if self.model_type == "looped" and self.loop_topology == "branch":
            self.branch_fusion_logits = nn.Parameter(torch.zeros(2))
            if self.branch_parent_fusion == "dynamic":
                self.branch_fusion_router = nn.Linear(3 * token_dim, 2)
                nn.init.zeros_(self.branch_fusion_router.weight)
                nn.init.zeros_(self.branch_fusion_router.bias)
            else:
                self.branch_fusion_router = None
        else:
            self.register_parameter("branch_fusion_logits", None)
            self.branch_fusion_router = None
        if (self.model_type == "looped" and self.loop_topology == "dag"
                and self.dag_parent_fusion == "softmax"):
            # A dense node-by-parent parameter matrix avoids registering a
            # ParameterDict, which torch.compile may wrap as an OptimizedModule.
            # Only entries selected by graph_parents participate in forward.
            # Zero initialization exactly reproduces mean fusion initially.
            self.dag_parent_fusion_logits = nn.Parameter(
                torch.zeros(self.num_tasks, self.num_tasks))
        else:
            self.register_parameter("dag_parent_fusion_logits", None)
        if (self.model_type == "looped" and self.loop_topology == "dag"
                and self.dag_parent_fusion.startswith("dynamic_")):
            self.dag_parent_routers = nn.ModuleList([
                nn.Linear((len(parents) + 1) * token_dim, len(parents))
                if len(parents) > 1 else nn.Identity()
                for parents in self.graph_parents
            ])
            for node, parents in enumerate(self.graph_parents):
                if len(parents) > 1:
                    nn.init.zeros_(self.dag_parent_routers[node].weight)
                    nn.init.zeros_(self.dag_parent_routers[node].bias)
        else:
            self.dag_parent_routers = None
        if self.model_type == "looped" and self.num_preloops > 0 and self.use_loop_gate:
            gate_init = min(max(float(loop_gate_init), 1e-4), 1.0 - 1e-4)
            gate_logit = torch.logit(torch.tensor(gate_init, dtype=torch.float32))
            self.preloop_gate_logits = nn.Parameter(gate_logit.repeat(self.num_preloops))
        else:
            self.register_parameter("preloop_gate_logits", None)

        if self.model_type == "looped" and self.use_base_state_bypass:
            if self.base_bypass_mode == "residual":
                # tanh(0)=0, so this mode starts exactly from the no-bypass model.
                self.base_bypass_logits = nn.Parameter(torch.zeros(num_tasks))
            else:
                bypass_init = min(max(float(base_bypass_init), 1e-4), 1.0 - 1e-4)
                self.base_bypass_logits = nn.Parameter(
                    torch.logit(torch.tensor(bypass_init, dtype=torch.float32)).repeat(num_tasks)
                )
            self.register_buffer(
                "base_bypass_mask", self._resolve_base_bypass_mask(num_tasks)
            )
        else:
            self.register_parameter("base_bypass_logits", None)
            self.register_buffer("base_bypass_mask", None)
        if self.task_loss_weights is None:
            self.register_buffer("task_loss_weight_tensor", None)
        else:
            weights = torch.tensor(
                [float(w) for w in self.task_loss_weights], dtype=torch.float32
            )
            if weights.numel() != num_tasks:
                raise ValueError(
                    "task_loss_weights must have one weight per task: "
                    f"got {weights.numel()} for {num_tasks} tasks"
                )
            if bool((weights <= 0).any()):
                raise ValueError("task_loss_weights must be positive")
            self.register_buffer("task_loss_weight_tensor", weights)

        if (self.model_type == "looped" and self.loop_topology == "dag"
                and self.child_adapter_dim > 0):
            self.child_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(token_dim, self.child_adapter_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(self.child_adapter_dim, token_dim, bias=False),
                ) if self._node_has_adapter(len(parents)) else nn.Identity()
                for parents in self.graph_parents
            ])
            # Start exactly from the no-adapter DAG and learn only residuals.
            for node, parents in enumerate(self.graph_parents):
                if self._node_has_adapter(len(parents)):
                    nn.init.zeros_(self.child_adapters[node][-1].weight)
        else:
            self.child_adapters = None

        # Keep explicit per-task taps so diagnostics can capture the exact
        # representation entering each readout under every sharing mode.
        self.task_head_input_taps = nn.ModuleList([nn.Identity() for _ in range(num_tasks)])
        self.tower = nn.ModuleList()
        self.shared_first_layer = None
        self.task_head_tails = None
        self.grouped_towers = None
        self.grouped_first_layers = None
        self.grouped_task_tails = None
        self.grouped_trunks = None
        self.grouped_task_outputs = None
        self.grouped_task_adapters = None
        self.task_to_head_group = None
        self.shared_head_trunk = None
        self.shared_tower = None
        self.task_output_layers = None
        if self.head_sharing == "independent":
            self.tower = nn.ModuleList([MLP_Block(input_dim=token_dim,
                                                  output_dim=1,
                                                  hidden_units=tower_hidden_units,
                                                  hidden_activations=tower_activations,
                                                  output_activation=None,
                                                  dropout_rates=net_dropout)
                                        for _ in range(num_tasks)])
        elif self.head_sharing == "shared_first":
            if isinstance(tower_activations, list):
                first_activations = tower_activations[:1]
                tail_activations = tower_activations[1:]
            else:
                first_activations = tower_activations
                tail_activations = tower_activations
            self.shared_first_layer = MLP_Block(
                input_dim=token_dim,
                output_dim=None,
                hidden_units=tower_hidden_units[:1],
                hidden_activations=first_activations,
                output_activation=None,
                dropout_rates=net_dropout,
            )
            self.task_head_tails = nn.ModuleList([
                MLP_Block(
                    input_dim=tower_hidden_units[0],
                    output_dim=1,
                    hidden_units=tower_hidden_units[1:],
                    hidden_activations=tail_activations,
                    output_activation=None,
                    dropout_rates=net_dropout,
                )
                for _ in range(num_tasks)
            ])
        elif self.head_sharing.startswith("grouped"):
            unique_groups = []
            for group in head_groups:
                if group not in unique_groups:
                    unique_groups.append(group)
            group_to_index = {group: index for index, group in enumerate(unique_groups)}
            self.task_to_head_group = [group_to_index[group] for group in head_groups]
            if self.head_sharing == "grouped":
                self.grouped_towers = nn.ModuleList([
                    MLP_Block(input_dim=token_dim,
                              output_dim=1,
                              hidden_units=tower_hidden_units,
                              hidden_activations=tower_activations,
                              output_activation=None,
                              dropout_rates=net_dropout)
                    for _ in unique_groups
                ])
            elif self.head_sharing == "grouped_first":
                if isinstance(tower_activations, list):
                    first_activations = tower_activations[:1]
                    tail_activations = tower_activations[1:]
                else:
                    first_activations = tower_activations
                    tail_activations = tower_activations
                self.grouped_first_layers = nn.ModuleList([
                    MLP_Block(
                        input_dim=token_dim,
                        output_dim=None,
                        hidden_units=tower_hidden_units[:1],
                        hidden_activations=first_activations,
                        output_activation=None,
                        dropout_rates=net_dropout,
                    )
                    for _ in unique_groups
                ])
                self.grouped_task_tails = nn.ModuleList([
                    MLP_Block(
                        input_dim=tower_hidden_units[0],
                        output_dim=1,
                        hidden_units=tower_hidden_units[1:],
                        hidden_activations=tail_activations,
                        output_activation=None,
                        dropout_rates=net_dropout,
                    )
                    for _ in range(num_tasks)
                ])
            else:
                trunk_output_dim = tower_hidden_units[-1]
                self.grouped_trunks = nn.ModuleList([
                    MLP_Block(
                        input_dim=token_dim,
                        output_dim=None,
                        hidden_units=tower_hidden_units,
                        hidden_activations=tower_activations,
                        output_activation=None,
                        dropout_rates=net_dropout,
                    )
                    for _ in unique_groups
                ])
                self.grouped_task_outputs = nn.ModuleList([
                    nn.Linear(trunk_output_dim, 1) for _ in range(num_tasks)
                ])
                if self.head_sharing == "grouped_trunk_adapter":
                    group_sizes = [self.task_to_head_group.count(index)
                                   for index in range(len(unique_groups))]
                    self.grouped_task_adapters = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(trunk_output_dim, self.head_adapter_dim, bias=False),
                            nn.ReLU(),
                            nn.Linear(self.head_adapter_dim, trunk_output_dim, bias=False),
                        ) if group_sizes[self.task_to_head_group[task_idx]] > 1
                        else nn.Identity()
                        for task_idx in range(num_tasks)
                    ])
        elif self.head_sharing == "shared_trunk":
            self.shared_head_trunk = MLP_Block(
                input_dim=token_dim,
                output_dim=None,
                hidden_units=tower_hidden_units,
                hidden_activations=tower_activations,
                output_activation=None,
                dropout_rates=net_dropout,
            )
            trunk_output_dim = tower_hidden_units[-1] if tower_hidden_units else token_dim
            self.task_output_layers = nn.ModuleList([
                nn.Linear(trunk_output_dim, 1) for _ in range(num_tasks)
            ])
        else:
            self.shared_tower = MLP_Block(
                input_dim=token_dim,
                output_dim=1,
                hidden_units=tower_hidden_units,
                hidden_activations=tower_activations,
                output_activation=None,
                dropout_rates=net_dropout,
            )
        if isinstance(task, list):
            assert len(task) == num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]
            )

        # Register experimental recurrence-conditioning modules after the
        # baseline model so their construction cannot perturb the initialization
        # of existing RankMixer and head parameters.
        max_iteration_depth = self.num_preloops + self._max_behavior_depth()
        if (self.model_type == "looped"
                and self.loop_depth_conditioning == "channel_scale"):
            self.depth_channel_scale = nn.Parameter(
                torch.zeros(max_iteration_depth + 1, token_dim)
            )
        else:
            self.register_parameter("depth_channel_scale", None)
        if (self.model_type == "looped"
                and self.loop_depth_conditioning in {
                    "sinusoidal", "behavior_sinusoidal"}):
            modulation_input_dim = self.depth_conditioning_dim
            if self.loop_depth_conditioning == "behavior_sinusoidal":
                modulation_input_dim += token_dim
            # Variant-only modules must not shift initialization of the
            # baseline RankMixer tensors in matched-seed comparisons.
            with torch.random.fork_rng(devices=[]):
                self.depth_modulation = nn.Sequential(
                    RecurrenceConditioningLinear(modulation_input_dim, token_dim),
                    nn.SiLU(),
                    RecurrenceConditioningLinear(token_dim, 3 * token_dim),
                )
                # Initialize before UniRank may wrap this module with
                # torch.compile; OptimizedModule is not subscriptable.
                nn.init.zeros_(self.depth_modulation[-1].weight)
                nn.init.zeros_(self.depth_modulation[-1].bias)
        else:
            self.depth_modulation = None
        if self.model_type == "looped" and self.behavior_adapter_dim > 0:
            with torch.random.fork_rng(devices=[]):
                self.behavior_adapters = nn.ModuleList([
                    nn.Sequential(
                        RecurrenceConditioningLinear(
                            token_dim, self.behavior_adapter_dim, bias=False
                        ),
                        nn.SiLU(),
                        RecurrenceConditioningLinear(
                            self.behavior_adapter_dim, token_dim, bias=False
                        ),
                    )
                    for _ in range(self.num_loops)
                ])
                for adapter in self.behavior_adapters:
                    nn.init.zeros_(adapter[-1].weight)
        else:
            self.behavior_adapters = None
        trust_input_dim = 3 * token_dim
        if self.model_type == "looped" and self.sample_trust_gate != "none":
            with torch.random.fork_rng(devices=[]):
                def make_trust_router():
                    router = nn.Sequential(
                        RecurrenceConditioningLinear(
                            trust_input_dim, self.sample_trust_hidden_dim
                        ),
                        nn.SiLU(),
                        RecurrenceConditioningLinear(
                            self.sample_trust_hidden_dim, 1
                        ),
                    )
                    nn.init.zeros_(router[-1].weight)
                    initial_logit = torch.logit(torch.tensor(self.sample_trust_init))
                    nn.init.constant_(router[-1].bias, float(initial_logit))
                    return router

                if self.sample_trust_gate == "stage":
                    self.sample_trust_routers = nn.ModuleList([
                        make_trust_router() for _ in range(self.num_loops)
                    ])
                    self.shared_sample_trust_router = None
                else:
                    self.sample_trust_routers = None
                    self.shared_sample_trust_router = make_trust_router()
        else:
            self.sample_trust_routers = None
            self.shared_sample_trust_router = None

        # A single scalar head gives every loop depth a comparable preference
        # scale. Construct it in an isolated RNG scope so enabling the margin
        # objective cannot shift initialization of the matched BCE baseline.
        if (self.model_type == "looped"
                and self.preference_margin_mode != "none"
                and self.preference_margin_weight > 0):
            with torch.random.fork_rng(devices=[]):
                self.preference_score_head = RecurrenceConditioningLinear(
                    token_dim, 1
                )
        else:
            self.preference_score_head = None

        self.compile(kwargs.get("dense_optimizer"), kwargs["loss"], kwargs.get("dense_learning_rate"))
        self.reset_parameters()
        if self.grouped_task_adapters is not None:
            for adapter in self.grouped_task_adapters:
                if isinstance(adapter, nn.Sequential):
                    nn.init.zeros_(adapter[-1].weight)
        if self.task_tokens is not None:
            nn.init.normal_(self.task_tokens, mean=0.0, std=0.02)
        if self.iteration_depth_embedding is not None:
            nn.init.normal_(self.iteration_depth_embedding.weight, mean=0.0, std=0.02)
        if (self.behavior_embedding is not None
                and self.behavior_embedding_permutation is not None):
            # Structural control: keep the graph, but let each node inject the
            # behavior code of a different stage. If the design is right, this
            # must hurt even though the topology and parameter count are fixed.
            perm = torch.tensor([int(i) for i in self.behavior_embedding_permutation])
            rows = self.behavior_embedding.weight.shape[0]
            if perm.numel() != rows or sorted(perm.tolist()) != list(range(rows)):
                raise ValueError(
                    "behavior_embedding_permutation must be a permutation of "
                    f"range({rows})"
                )
            with torch.no_grad():
                self.behavior_embedding.weight.copy_(
                    self.behavior_embedding.weight[perm]
                )
        if self.behavior_embedding is not None:
            if self.behavior_embedding_init == "zero":
                nn.init.zeros_(self.behavior_embedding.weight)
            elif self.behavior_embedding_init == "small_normal":
                nn.init.normal_(self.behavior_embedding.weight, mean=0.0,
                                std=self.behavior_embedding_init_std)
            elif self.behavior_embedding_init == "scaled_default":
                self.behavior_embedding.weight.data.mul_(
                    self.behavior_embedding_init_std
                )
        self.model_to_device()

    def _append_task_tokens(self, tokens):
        if self.task_tokens is None:
            return tokens
        return torch.cat([
            tokens,
            self.task_tokens.expand(tokens.size(0), -1, -1)
        ], dim=1)

    def _initialize_behavior_graph(self, graph):
        labels = list(self.feature_map.labels)
        if not isinstance(graph, dict):
            raise ValueError("loop_topology=dag requires behavior_graph as a label->parents mapping")
        unknown_nodes = set(graph) - set(labels)
        missing_nodes = set(labels) - set(graph)
        if unknown_nodes or missing_nodes:
            raise ValueError(
                f"behavior_graph labels mismatch: unknown={sorted(unknown_nodes)}, "
                f"missing={sorted(missing_nodes)}"
            )
        label_to_index = {label: index for index, label in enumerate(labels)}
        parents = []
        for label in labels:
            parent_labels = graph[label] or []
            if not isinstance(parent_labels, list):
                raise ValueError(f"behavior_graph[{label}] must be a list")
            if label in parent_labels:
                raise ValueError(f"behavior_graph contains a self-loop at {label}")
            unknown_parents = set(parent_labels) - set(labels)
            if unknown_parents:
                raise ValueError(
                    f"behavior_graph[{label}] has unknown parents={sorted(unknown_parents)}"
                )
            parents.append([label_to_index[parent] for parent in parent_labels])

        indegree = [len(node_parents) for node_parents in parents]
        children = [[] for _ in labels]
        for child, node_parents in enumerate(parents):
            for parent in node_parents:
                children[parent].append(child)
        queue = [index for index, degree in enumerate(indegree) if degree == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(labels):
            raise ValueError("behavior_graph must be acyclic")

        descendants = []
        for node in range(len(labels)):
            seen = set()
            stack = list(children[node])
            while stack:
                child = stack.pop()
                if child in seen:
                    continue
                seen.add(child)
                stack.extend(children[child])
            descendants.append(sorted(seen, key=order.index))
        self.graph_parents = parents
        self.graph_topological_order = order
        self.graph_descendants = descendants
        depths = [0] * len(labels)
        for node in order:
            if parents[node]:
                depths[node] = max(depths[parent] for parent in parents[node]) + 1
        self.graph_depths = depths

    def _max_behavior_depth(self):
        if self.loop_topology == "linear":
            return self.num_loops - 1
        if self.loop_topology == "branch":
            return 2
        return max(self.graph_depths)

    def _behavior_iteration_depth(self, stage):
        if self.loop_topology == "linear":
            behavior_depth = stage
        elif self.loop_topology == "branch":
            behavior_depth = (0, 1, 1, 2)[stage]
        else:
            behavior_depth = self.graph_depths[stage]
        return self.num_preloops + behavior_depth

    def _condition_on_iteration_depth(self, state, depth):
        if self.iteration_depth_embedding is None:
            return state
        return state + self.iteration_depth_embedding.weight[depth].view(1, 1, -1)

    def _sinusoidal_depth_code(self, depth, device, dtype):
        half = self.depth_conditioning_dim // 2
        if half == 0:
            return torch.zeros(1, self.depth_conditioning_dim,
                               device=device, dtype=dtype)
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angle = torch.as_tensor(float(depth), device=device) * frequencies
        code = torch.cat([torch.sin(angle), torch.cos(angle)], dim=0)
        if code.numel() < self.depth_conditioning_dim:
            code = F.pad(code, (0, self.depth_conditioning_dim - code.numel()))
        return code.view(1, -1).to(dtype)

    def _depth_modulation_values(self, depth, stage, reference):
        if self.depth_modulation is None:
            return None
        code = self._sinusoidal_depth_code(
            depth, reference.device, reference.dtype
        )
        if self.loop_depth_conditioning == "behavior_sinusoidal":
            if stage is None or self.behavior_embedding is None:
                behavior = torch.zeros(
                    1, self.token_dim, device=reference.device,
                    dtype=reference.dtype,
                )
            else:
                behavior = self.behavior_embedding.weight[stage].view(1, -1)
                behavior = behavior.to(reference.dtype)
            code = torch.cat([code, behavior], dim=-1)
        scale, shift, residual_scale = self.depth_modulation(code).chunk(3, dim=-1)
        return (
            scale.view(1, 1, -1),
            shift.view(1, 1, -1),
            residual_scale.view(1, 1, -1),
        )

    def _apply_recurrence_conditioning(self, conditioned, depth, stage):
        modulation = self._depth_modulation_values(
            depth, stage, conditioned
        )
        if modulation is None:
            return conditioned, None
        scale, shift, residual_scale = modulation
        return conditioned * (1.0 + scale) + shift, residual_scale

    def _apply_conditioned_update(self, previous, candidate, depth,
                                  residual_scale=None):
        delta = candidate - previous
        if self.depth_channel_scale is not None:
            scale = self.depth_channel_scale[depth].view(1, 1, -1)
            delta = delta * (1.0 + scale.to(delta.dtype))
        if residual_scale is not None:
            delta = delta * (1.0 + residual_scale.to(delta.dtype))
        return previous + delta

    def _depth_loop_gate(self, depth, dtype):
        max_depth = self.num_preloops + self._max_behavior_depth()
        normalized_depth = float(depth) / max(max_depth, 1)
        slope = F.softplus(self.depth_gate_slope_raw)
        gate = torch.sigmoid(
            self.depth_gate_base_logit - slope * normalized_depth
        )
        return gate.to(dtype)

    def _sample_trust_value(self, previous, proposal, stage):
        if (self.sample_trust_routers is None
                and self.shared_sample_trust_router is None):
            return None
        delta = proposal - previous
        router_input = torch.cat([
            previous.mean(dim=1),
            proposal.mean(dim=1),
            delta.mean(dim=1),
        ], dim=-1)
        router = (
            self.sample_trust_routers[stage]
            if self.sample_trust_routers is not None
            else self.shared_sample_trust_router
        )
        raw = router(router_input.float()).squeeze(-1)
        # Exactly one at initialization, while retaining a non-zero derivative
        # that permits only attenuation of harmful proposals.
        base_trust = torch.sigmoid(raw)
        calibrated = (1.0 + base_trust - self.sample_trust_init).clamp(0.0, 1.0)
        # Straight-through calibration: the forward value is exactly bounded
        # and equals one at initialization, while backward follows the smooth
        # sigmoid path instead of the zero derivative of clamp at its boundary.
        trust = calibrated.detach() + base_trust - base_trust.detach()
        return trust.to(proposal.dtype)

    def _apply_sample_trust(self, previous, proposal, stage):
        trust = self._sample_trust_value(previous, proposal, stage)
        if trust is None:
            return proposal, None
        return (
            previous + trust.view(-1, 1, 1) * (proposal - previous),
            trust,
        )

    def _state_representation(self, state, task_idx):
        if self.task_token_mode == "single":
            return state[:, -1, :]
        if self.task_token_mode == "multi":
            return state[:, self.num_ns_token + task_idx, :]
        return state.mean(dim=1)

    def _loop_update(self, h, stage):
        conditioned = h
        if self.behavior_embedding is not None:
            conditioned = conditioned + self.behavior_embedding.weight[stage].view(1, 1, -1)
        depth = self._behavior_iteration_depth(stage)
        conditioned = self._condition_on_iteration_depth(conditioned, depth)
        if self.behavior_adapters is not None:
            conditioned = conditioned + self.behavior_adapters[stage](conditioned)
        conditioned, residual_scale = self._apply_recurrence_conditioning(
            conditioned, depth, stage
        )
        block = self.loop_block if self.loop_block is not None else self.loop_blocks[stage]
        candidate = self.activation_checkpoint(block, conditioned)
        candidate = self._apply_conditioned_update(
            h, candidate, depth, residual_scale
        )
        if self.depth_gate_base_logit is not None:
            gate = self._depth_loop_gate(
                depth, candidate.dtype
            )
            proposal = h + gate * (candidate - h)
        elif self.loop_gate_logits is None:
            proposal = candidate
        else:
            gate = torch.sigmoid(self.loop_gate_logits[stage]).to(candidate.dtype)
            proposal = h + gate * (candidate - h)
        return self._apply_sample_trust(h, proposal, stage)

    def _run_preloops(self, tokens):
        """Run optional behavior-agnostic refinement with the same loop block."""
        h = tokens
        for index in range(self.num_preloops):
            conditioned = self._condition_on_iteration_depth(h, index)
            conditioned, residual_scale = self._apply_recurrence_conditioning(
                conditioned, index, None
            )
            block = (self.loop_block if self.loop_block is not None
                     else self.loop_blocks[self.num_loops + index])
            candidate = self.activation_checkpoint(block, conditioned)
            candidate = self._apply_conditioned_update(
                h, candidate, index, residual_scale
            )
            if self.depth_gate_base_logit is not None:
                gate = self._depth_loop_gate(index, candidate.dtype)
                h = h + gate * (candidate - h)
            elif self.preloop_gate_logits is None:
                h = candidate
            else:
                gate = torch.sigmoid(self.preloop_gate_logits[index]).to(candidate.dtype)
                h = h + gate * (candidate - h)
        return h

    def _fuse_dag_parents(self, node, states_by_node, base_state):
        parents = self.graph_parents[node]
        parent_states = [states_by_node[parent] for parent in parents]
        if self.disable_branch_merge and len(parent_states) > 1:
            # Keep every branch computation but route only the first declared parent.
            return parent_states[0]
        if len(parent_states) == 1 or self.dag_parent_fusion == "mean":
            return torch.stack(parent_states, dim=0).mean(dim=0)
        stacked = torch.stack(parent_states, dim=0)
        if self.dag_parent_fusion == "softmax":
            weights = torch.softmax(
                self.dag_parent_fusion_logits[node, parents], dim=0
            ).to(parent_states[0].dtype)
            return (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)

        # Sample-level, child-specific routing. Pooling is used only to predict
        # weights; the routed values remain full RankMixer token tensors.
        router_input = torch.cat(
            [state.mean(dim=1) for state in parent_states] + [base_state.mean(dim=1)],
            dim=-1,
        )
        logits = self.dag_parent_routers[node](router_input).float()
        if self.dag_parent_fusion == "dynamic_softmax":
            weights = torch.softmax(logits, dim=-1)
        else:
            weights = torch.sigmoid(logits)
        weights = weights.to(parent_states[0].dtype)
        return (stacked * weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

    @staticmethod
    def _relative_step(next_state, previous_state):
        delta = (next_state - previous_state).float().flatten(1).norm(dim=1)
        base = previous_state.float().flatten(1).norm(dim=1).clamp_min(1e-6)
        return delta / base

    def _run_progressive_encoder(self, tokens):
        states = []
        steps = []
        trusts = []
        tokens = self._run_preloops(tokens)
        base_state = tokens
        if self.loop_topology == "linear":
            h = tokens
            for stage in range(self.num_loops):
                next_h, trust = self._loop_update(h, stage)
                steps.append(self._relative_step(next_h, h))
                trusts.append(trust)
                h = next_h
                states.append(h)
        elif self.loop_topology == "branch":
            h_click, trust_click = self._loop_update(tokens, 0)
            h_cart, trust_cart = self._loop_update(h_click, 1)
            h_fav, trust_fav = self._loop_update(h_click, 2)
            if self.branch_fusion_router is None:
                fusion_logits = self.branch_fusion_logits
            else:
                fusion_input = torch.cat([
                    h_click.mean(dim=1), h_cart.mean(dim=1), h_fav.mean(dim=1)
                ], dim=-1)
                fusion_logits = self.branch_fusion_router(fusion_input).float()
            fusion_gates = torch.sigmoid(fusion_logits).to(h_click.dtype)
            if fusion_gates.ndim == 1:
                cart_gate, fav_gate = fusion_gates[0], fusion_gates[1]
            else:
                cart_gate = fusion_gates[:, 0].view(-1, 1, 1)
                fav_gate = fusion_gates[:, 1].view(-1, 1, 1)
            h_fused = (
                h_click
                + cart_gate * (h_cart - h_click)
                + fav_gate * (h_fav - h_click)
            )
            if self.disable_branch_merge:
                h_fused = h_cart
            h_buy, trust_buy = self._loop_update(h_fused, 3)
            states = [h_click, h_cart, h_fav, h_buy]
            trusts = [trust_click, trust_cart, trust_fav, trust_buy]
            steps = [
                self._relative_step(h_click, tokens),
                self._relative_step(h_cart, h_click),
                self._relative_step(h_fav, h_click),
                self._relative_step(h_buy, h_fused),
            ]
        else:
            states_by_node = [None] * self.num_tasks
            steps_by_node = [None] * self.num_tasks
            trusts_by_node = [None] * self.num_tasks
            for node in self.graph_topological_order:
                node_parents = self.graph_parents[node]
                if node_parents:
                    parent_state = self._fuse_dag_parents(node, states_by_node, base_state)
                else:
                    parent_state = tokens
                node_state, node_trust = self._loop_update(parent_state, node)
                if self.child_adapters is not None and self._node_has_adapter(
                    len(node_parents)
                ):
                    node_state = node_state + self.child_adapters[node](parent_state)
                states_by_node[node] = node_state
                steps_by_node[node] = self._relative_step(node_state, parent_state)
                trusts_by_node[node] = node_trust
            states = states_by_node
            steps = steps_by_node
            trusts = trusts_by_node
        if self.enable_loop_diagnostics and len(states) > 1:
            with torch.no_grad():
                flat = [x.detach().float().flatten(1) for x in states]
                self.last_loop_diagnostics = {
                    "delta_norm": [float((flat[i] - flat[i - 1]).norm(dim=1).mean())
                                   for i in range(1, len(flat))],
                    "cosine": [float(F.cosine_similarity(flat[i - 1], flat[i], dim=1).mean())
                               for i in range(1, len(flat))],
                }
        trust_tensor = None
        if trusts and all(trust is not None for trust in trusts):
            trust_tensor = torch.stack(trusts, dim=1)
        return states, torch.stack(steps, dim=1), base_state, trust_tensor

    def _node_has_adapter(self, num_parents):
        """Whether a DAG node owns a residual adapter.

        `multi_parent` gives adapters only to nodes that fuse several parents
        (on KuaiRand these are exactly the gradient-starved rare tasks);
        `all` gives every behaviour node its own residual capacity.
        """
        if self.child_adapter_nodes == "all":
            return True
        return num_parents > 1

    def _resolve_loop_block_layers(self, num_layers, default_layers):
        """Layers inside one loop step.

        Default: 1 for refinement-only loops, `num_layers` for whole-encoder
        loops. `loop_block_layers` overrides both, which decouples the
        refinement depth from the parallel backbone depth.
        """
        if self.loop_block_layers is not None:
            return self.loop_block_layers
        return default_layers

    def _resolve_base_bypass_mask(self, num_tasks):
        """One-hot-ish mask limiting the bypass to a task subset (1 = bypass)."""
        if self.base_bypass_tasks is None:
            return torch.ones(num_tasks)
        labels = list(self.feature_map.labels)
        selected = set()
        for item in self.base_bypass_tasks:
            if isinstance(item, int):
                if not 0 <= item < num_tasks:
                    raise ValueError(
                        f"base_bypass_tasks index out of range: {item}"
                    )
                selected.add(int(item))
            else:
                if str(item) not in labels:
                    raise ValueError(
                        f"base_bypass_tasks contains unknown label: {item}"
                    )
                selected.add(labels.index(str(item)))
        if not selected:
            raise ValueError("base_bypass_tasks resolved to an empty task set")
        return torch.tensor(
            [1.0 if index in selected else 0.0 for index in range(num_tasks)]
        )

    def _head_representation(self, state, task_idx, base_state):
        representation = self._state_representation(state, task_idx)
        if self.base_bypass_logits is None:
            return representation
        base_representation = self._state_representation(base_state, task_idx)
        mask = self.base_bypass_mask[task_idx].to(representation.dtype)
        if self.base_bypass_mode == "residual":
            weight = torch.tanh(self.base_bypass_logits[task_idx]).to(representation.dtype)
            weight = weight * mask
            return representation + weight * (base_representation - representation)
        weight = torch.sigmoid(self.base_bypass_logits[task_idx]).to(representation.dtype)
        weight = weight * mask
        return weight * base_representation + (1.0 - weight) * representation

    def _task_logit(self, representation, task_idx):
        representation = self.task_head_input_taps[task_idx](representation)
        if self.head_sharing == "independent":
            return self.tower[task_idx](representation)
        if self.head_sharing == "shared_first":
            return self.task_head_tails[task_idx](
                self.shared_first_layer(representation)
            )
        if self.head_sharing == "grouped":
            return self.grouped_towers[
                self.task_to_head_group[task_idx]
            ](representation)
        if self.head_sharing == "grouped_first":
            group_idx = self.task_to_head_group[task_idx]
            return self.grouped_task_tails[task_idx](
                self.grouped_first_layers[group_idx](representation)
            )
        if self.head_sharing in {"grouped_trunk", "grouped_trunk_adapter"}:
            group_idx = self.task_to_head_group[task_idx]
            hidden = self.grouped_trunks[group_idx](representation)
            if self.grouped_task_adapters is not None:
                hidden = hidden + self.grouped_task_adapters[task_idx](hidden)
            return self.grouped_task_outputs[task_idx](hidden)
        if self.head_sharing == "shared_trunk":
            return self.task_output_layers[task_idx](
                self.shared_head_trunk(representation)
            )
        return self.shared_tower(representation)

    def _preference_depth_logits(self, states):
        """Return shared-scale logits grouped by feedback-graph depth."""
        stage_logits = [
            self.preference_score_head(
                self._state_representation(state, min(index, self.num_tasks - 1))
            ).squeeze(-1)
            for index, state in enumerate(states)
        ]
        if self.loop_topology == "branch":
            # Click -> {Cart, Fav} -> Buy. Cart and Fav are peers, not an
            # artificial ordinal chain.
            return torch.stack([
                stage_logits[0],
                0.5 * (stage_logits[1] + stage_logits[2]),
                stage_logits[3],
            ], dim=1)
        if self.loop_topology == "dag":
            depth_logits = []
            for depth in sorted(set(self.graph_depths)):
                nodes = [index for index, value in enumerate(self.graph_depths)
                         if value == depth]
                depth_logits.append(
                    torch.stack([stage_logits[index] for index in nodes], dim=1).mean(dim=1)
                )
            return torch.stack(depth_logits, dim=1)
        return torch.stack(stage_logits, dim=1)

    def _preference_margins(self, num_depths, reference):
        if self.preference_margin_mode == "fixed":
            return reference.new_tensor([self.preference_margin_target])
        margins = torch.linspace(
            self.preference_margin_start,
            self.preference_margin_target,
            steps=num_depths,
            device=reference.device,
            dtype=reference.dtype,
        )
        if self.preference_margin_mode == "reverse_progressive":
            margins = margins.flip(0)
        return margins

    def _preference_pair_masks(self, y_true):
        labels = list(self.feature_map.labels)
        if self.preference_positive_label not in labels:
            raise ValueError(
                f"preference_positive_label={self.preference_positive_label} "
                f"is not in labels={labels}"
            )
        positive_index = labels.index(self.preference_positive_label)
        positive = y_true[positive_index].reshape(-1) > 0.5
        non_purchase = ~positive
        if self.preference_negative_mode == "engaged_non_buy":
            engaged = torch.zeros_like(positive)
            for index in range(positive_index):
                engaged |= y_true[index].reshape(-1) > 0.5
            negative = engaged & non_purchase
        else:
            if "is_click" not in labels:
                raise ValueError("click_only negatives require an is_click label")
            click = y_true[labels.index("is_click")].reshape(-1) > 0.5
            other_feedback = torch.zeros_like(positive)
            for index, label in enumerate(labels):
                if label not in {"is_click", self.preference_positive_label}:
                    other_feedback |= y_true[index].reshape(-1) > 0.5
            negative = click & non_purchase & ~other_feedback
        return positive, negative

    def _preference_margin_loss(self, depth_logits, y_true, group_ids):
        positive, negative = self._preference_pair_masks(y_true)
        if not positive.any() or not negative.any():
            # Keep a differentiable zero for sparse mini-batches with no pair.
            return depth_logits.sum() * 0.0
        if self.preference_margin_mode == "fixed":
            selected = depth_logits[:, -1:]
        else:
            selected = depth_logits
        margins = self._preference_margins(selected.size(1), selected)
        positive_scores = selected[positive]
        negative_scores = selected[negative]
        pairwise_gap = positive_scores[:, None, :] - negative_scores[None, :, :]
        positive_groups = group_ids.reshape(-1)[positive]
        negative_groups = group_ids.reshape(-1)[negative]
        valid_pairs = positive_groups[:, None] == negative_groups[None, :]
        if not valid_pairs.any():
            return depth_logits.sum() * 0.0
        losses = F.softplus(margins.view(1, 1, -1) - pairwise_gap)
        return losses[valid_pairs].mean()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]
        # item_dict contains [history_items..., target_item]
        # Reshape flattened embeddings to B x (T+1) x item_info_dim
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_seq_emb[:, 0:-1, :]  # B x T x item_info_dim

        seq_pooling_emb = self.attention_layers(target_emb, sequence_emb, mask) # B x embedding_dim

        # Other non-sequential features -> NS tokens
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)       # B x non_item_dim
        feature_embeddings = torch.cat([user_context_emb, target_emb, seq_pooling_emb], dim=-1)
        if self.tokenizer_uses_field_input:
            feature_embeddings = feature_embeddings.reshape(
                batch_size, self.num_tokenizer_fields, self.embedding_dim
            )
        unified_tokens = self.unified_tokenizer_layer(feature_embeddings)           # B x num_ns_token x token_dim
        unified_tokens = self._append_task_tokens(unified_tokens)


        # unified model
        if self.model_type == "parallel":
            unified_tokens = self.activation_checkpoint(
                self.unified_interaction_layers, unified_tokens)
            states = [unified_tokens]
            base_state = unified_tokens
        else:
            states, loop_steps, base_state, sample_trust = self._run_progressive_encoder(
                unified_tokens
            )

        if self.supervision_mode == "final" or self.model_type == "parallel":
            prediction_states = [states[-1]] * self.num_tasks
        else:
            prediction_states = [states[min(i, len(states) - 1)]
                                 for i in range(self.num_tasks)]
        tower_output = [
            self._task_logit(
                self._head_representation(prediction_states[i], i, base_state), i
            )
            for i in range(self.num_tasks)
        ]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        if self.supervision_mode == "deep_progressive" and self.model_type != "parallel":
            # Each behavior remains supervised at all deeper refinement stages.
            # These auxiliary outputs are consumed only by add_loss().
            for task_idx in range(self.num_tasks):
                if self.loop_topology == "dag":
                    auxiliary_stages = self.graph_descendants[task_idx]
                else:
                    auxiliary_stages = range(task_idx + 1, len(states))
                for stage_idx in auxiliary_stages:
                    rep = self._head_representation(states[stage_idx], task_idx, base_state)
                    aux = self.output_activation[task_idx](
                        self._task_logit(rep, task_idx)
                    )
                    return_dict[f"__aux_{task_idx}_{stage_idx}"] = aux
        if self.model_type == "looped":
            return_dict["__loop_steps"] = loop_steps
            if sample_trust is not None:
                return_dict["__sample_trust"] = sample_trust
            return_dict["__loop_representations"] = torch.stack([
                self._state_representation(state, min(index, self.num_tasks - 1))
                for index, state in enumerate(states)
            ], dim=1)
        if self.preference_score_head is not None:
            return_dict["__preference_depth_logits"] = self._preference_depth_logits(states)
            return_dict["__preference_group_ids"] = batch_dict[
                self.feature_map.group_id
            ]
        return return_dict

    def add_loss(self, return_dict, y_true):
        primary = [
            self.loss_fn[i](return_dict[f"{self.feature_map.labels[i]}_pred"],
                            y_true[i], reduction="mean")
            for i in range(self.num_tasks)
        ]
        stacked = torch.stack(primary)
        if self.task_loss_weight_tensor is not None:
            stacked = stacked * self.task_loss_weight_tensor.to(stacked.dtype)
        total = stacked.sum()
        if self.supervision_mode == "deep_progressive" and self.model_type != "parallel":
            auxiliary = []
            for task_idx in range(self.num_tasks):
                if self.loop_topology == "dag":
                    auxiliary_stages = self.graph_descendants[task_idx]
                else:
                    auxiliary_stages = range(task_idx + 1, self.num_loops)
                for stage_idx in auxiliary_stages:
                    key = f"__aux_{task_idx}_{stage_idx}"
                    if key in return_dict:
                        auxiliary.append(self.loss_fn[task_idx](
                            return_dict[key], y_true[task_idx], reduction="mean"))
            if auxiliary:
                total = total + self.auxiliary_loss_weight * torch.stack(auxiliary).mean()
        if self.model_type == "looped" and self.loop_step_reg_weight > 0:
            steps = return_dict["__loop_steps"]
            if self.loop_topology == "linear":
                penalties = [
                    F.relu(steps[:, index] - steps[:, index - 1] + self.loop_step_margin)
                    for index in range(1, steps.size(1))
                ]
            elif self.loop_topology == "branch":
                middle = 0.5 * (steps[:, 1] + steps[:, 2])
                penalties = [
                    F.relu(middle - steps[:, 0] + self.loop_step_margin),
                    F.relu(steps[:, 3] - middle + self.loop_step_margin),
                ]
            else:
                penalties = []
                for child, parents in enumerate(self.graph_parents):
                    if not parents:
                        continue
                    parent_step = torch.stack(
                        [steps[:, parent] for parent in parents], dim=1
                    ).mean(dim=1)
                    penalties.append(
                        F.relu(steps[:, child] - parent_step + self.loop_step_margin)
                    )
            if penalties:
                total = total + self.loop_step_reg_weight * torch.stack(penalties).mean()
        if self.model_type == "looped" and self.loop_variance_reg_weight > 0:
            representations = return_dict["__loop_representations"].float()
            feature_std = representations.std(dim=0, unbiased=False)
            variance_penalty = F.relu(self.loop_variance_floor - feature_std).mean()
            total = total + self.loop_variance_reg_weight * variance_penalty
        if (self.model_type == "looped" and self.sample_trust_reg_weight > 0
                and "__sample_trust" in return_dict):
            trust = return_dict["__sample_trust"].float()
            total = total + self.sample_trust_reg_weight * (1.0 - trust).mean()
        if self.preference_score_head is not None:
            margin_loss = self._preference_margin_loss(
                return_dict["__preference_depth_logits"], y_true,
                return_dict["__preference_group_ids"],
            )
            total = total + self.preference_margin_weight * margin_loss
        return total

    def loop_parameter_ids(self):
        """Return registered parameter identities for sharing sanity tests."""
        if self.loop_block is not None:
            return {id(p) for p in self.loop_block.parameters()}
        if self.loop_blocks is not None:
            return [{id(p) for p in block.parameters()} for block in self.loop_blocks]
        return set()

class RankMixerBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_ns_token,
                 num_layers,
                 expand=2,
                 net_dropout=0.0):
        super(RankMixerBlock, self).__init__()
        self.num_layers = num_layers

        self.mixer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim)
            for _ in range(num_layers)
        ])
        self.pffn_norms = nn.ModuleList([
            nn.LayerNorm(input_dim)
            for _ in range(num_layers)
        ])

        self.mixer_layers = nn.ModuleList([
            MultiHeadTokenMixing(input_dim=input_dim, num_token=num_ns_token)
            for _ in range(num_layers)
        ])
        self.pffn_layers = nn.ModuleList([
            PerTokenFeedForward(input_dim=input_dim,
                           num_token=num_ns_token,
                           expand=expand,
                           net_dropout=net_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        for i in range(self.num_layers):
            x = self.mixer_layers[i](self.mixer_norms[i](x)) + x
            x = self.pffn_layers[i](self.pffn_norms[i](x)) + x
        return x
