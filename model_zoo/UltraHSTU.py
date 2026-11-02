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

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block
from unirank.pytorch.torch_utils import disable_torch_compile


class UltraHSTU(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="UltraHSTU",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 truncation_start_layer=1,
                 truncation_ratio=0.5,
                 k1=[32, 32, 16],
                 k2=[20, 10, 10],
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=64,
                 num_heads=2,
                 net_dropout=0,
                 accumulation_steps=1,
                 model_type="parallel",
                 num_loops=None,
                 use_behavior_embedding=True,
                 supervision_mode="progressive",
                 auxiliary_loss_weight=0.5,
                 **kwargs):
        super(UltraHSTU, self).__init__(feature_map,
                                        model_id=model_id,
                                        gpu=gpu,
                                        **kwargs)
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.accumulation_steps = accumulation_steps
        self.truncation_start_layer = truncation_start_layer
        self.truncation_ratio = truncation_ratio
        self.model_type = str(model_type).lower()
        self.num_loops = int(num_loops or num_tasks)
        self.use_behavior_embedding = bool(use_behavior_embedding)
        self.supervision_mode = str(supervision_mode).lower()
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        if self.model_type not in {"parallel", "looped"}:
            raise ValueError(f"Unsupported UltraHSTU model_type={model_type}")
        if self.supervision_mode not in {"progressive", "final", "deep_progressive"}:
            raise ValueError(f"Unsupported supervision_mode={supervision_mode}")

        assert len(k1) == len(k2) and truncation_start_layer <= len(k1)
        num_layers = len(k1)

        # Track item and non-item feature dimensions
        self.item_info_dim = 0
        self.user_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item"]:
                self.item_info_dim += emb_dim
            elif spec.get("source") not in ["item", "action"]:
                self.user_info_dim += emb_dim

        # The action is embedding alone, and the other features are embedding conventionally.
        self.action_embedding_layer = FeatureEmbedding(
            feature_map, self.item_info_dim, required_feature_columns=['action']
        )
        self.feature_embedding_layer = FeatureEmbedding(
            feature_map, embedding_dim, not_required_feature_columns=['action']
        )

        self.unified_tokenizer_layer = UltraHSTUTokenizer(
            user_info_dim=self.user_info_dim,
            item_info_dim=self.item_info_dim,
            token_dim=token_dim
        )
        if self.model_type == "parallel":
            self.unified_interaction_layers = UnifiedInteractionBlocks(
                token_dim=token_dim, num_layers=num_layers,
                truncation_start_layer=truncation_start_layer,
                truncation_ratio=truncation_ratio, k1=k1, k2=k2,
                num_heads=num_heads, expansion_factor=expansion_factor,
                net_dropout=net_dropout)
            self.loop_block = None
        else:
            self.unified_interaction_layers = None
            self.loop_block = SequentialTransductionUnit(
                token_dim=token_dim, num_heads=num_heads,
                expansion_factor=expansion_factor, net_dropout=net_dropout)
        self.loop_k1 = list(k1)
        self.loop_k2 = list(k2)
        self.behavior_embedding = (
            nn.Embedding(self.num_loops, token_dim)
            if self.model_type == "looped" and self.use_behavior_embedding else None)
        self.tower = nn.ModuleList([
            MLP_Block(input_dim=token_dim,
                      output_dim=1,
                      hidden_units=tower_hidden_units,
                      hidden_activations=tower_activations,
                      output_activation=None,
                      dropout_rates=net_dropout)
            for _ in range(num_tasks)
        ])
        if isinstance(task, list):
            assert len(task) == num_tasks
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(str(t)) for t in task]
            )
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]
            )

        self.compile(kwargs.get("dense_optimizer"), kwargs["loss"], kwargs.get("dense_learning_rate"))
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]

        item_seq_emb = self.feature_embedding_layer(item_dict, flatten_emb=True)
        action_emb = self.action_embedding_layer(item_dict, flatten_emb=True)
        sequence_emb = item_seq_emb + action_emb                              # B x (T+1) x item_info_dim

        sequence_emb = sequence_emb.view(batch_size, -1, self.item_info_dim)   # B x (T+1) x item_info_dim

        # user context
        user_context_emb = self.feature_embedding_layer(batch_dict, flatten_emb=True) # B x user_info_dim

        # [user_token][sequence_tokens|candidate]
        unified_tokens = self.unified_tokenizer_layer(user_context_emb, sequence_emb) # B x (1 + T+1) x D

        states = []
        if self.model_type == "parallel":
            unified_tokens = self.activation_checkpoint(
                self.unified_interaction_layers, unified_tokens, mask)
            states.append(unified_tokens[:, -1, :])
        else:
            valid_mask = self._expand_loop_mask(unified_tokens, mask)
            for stage in range(self.num_loops):
                if self.behavior_embedding is not None:
                    unified_tokens = unified_tokens.clone()
                    unified_tokens[:, -1, :] = (
                        unified_tokens[:, -1, :] + self.behavior_embedding.weight[stage])
                k1 = self.loop_k1[min(stage, len(self.loop_k1) - 1)]
                k2 = self.loop_k2[min(stage, len(self.loop_k2) - 1)]
                unified_tokens = self.loop_block(
                    unified_tokens, valid_mask=valid_mask, k1=k1, k2=k2) + unified_tokens
                states.append(unified_tokens[:, -1, :])

        if self.supervision_mode == "final" or self.model_type == "parallel":
            selected = [states[-1]] * self.num_tasks
        else:
            selected = [states[min(i, len(states) - 1)] for i in range(self.num_tasks)]
        tower_output = [self.tower[i](selected[i]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict[f"{labels[i]}_pred"] = y_pred[i]
        if self.supervision_mode == "deep_progressive" and self.model_type == "looped":
            for task_idx in range(self.num_tasks):
                for stage_idx in range(task_idx + 1, len(states)):
                    aux = self.output_activation[task_idx](self.tower[task_idx](states[stage_idx]))
                    return_dict[f"__aux_{task_idx}_{stage_idx}"] = aux
        return return_dict

    def _expand_loop_mask(self, tokens, history_mask):
        if history_mask is None:
            return None
        batch_size = tokens.size(0)
        user_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
        candidate_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
        return torch.cat([user_valid, history_mask.bool(), candidate_valid], dim=1)

    def add_loss(self, return_dict, y_true):
        total = torch.stack([
            self.loss_fn[i](return_dict[f"{self.feature_map.labels[i]}_pred"],
                            y_true[i], reduction="mean")
            for i in range(self.num_tasks)
        ]).sum()
        if self.supervision_mode == "deep_progressive" and self.model_type == "looped":
            auxiliary = [
                self.loss_fn[t](return_dict[f"__aux_{t}_{s}"], y_true[t], reduction="mean")
                for t in range(self.num_tasks) for s in range(t + 1, self.num_loops)
            ]
            if auxiliary:
                total = total + self.auxiliary_loss_weight * torch.stack(auxiliary).mean()
        return total


class UltraHSTUTokenizer(nn.Module):
    def __init__(self, user_info_dim, item_info_dim, token_dim):
        super(UltraHSTUTokenizer, self).__init__()
        self.mlp_u = MLP_Block(
            input_dim=user_info_dim,
            hidden_units=[token_dim],
            hidden_activations="SiLU",
            layer_norm=True,
        )
        self.mlp_i = MLP_Block(
            input_dim=item_info_dim,
            hidden_units=[token_dim],
            hidden_activations="SiLU",
            layer_norm=True,
        )

    def forward(self, user_context_emb, sequence_emb):
        user_token = self.mlp_u(user_context_emb).unsqueeze(1)
        seq_tokens = self.mlp_i(sequence_emb)
        return torch.cat([user_token, seq_tokens], dim=1)


class UnifiedInteractionBlocks(nn.Module):
    def __init__(self,
                 token_dim,
                 num_layers,
                 truncation_start_layer,
                 truncation_ratio,
                 k1,
                 k2,
                 num_heads=2,
                 expansion_factor=4,
                 net_dropout=0.0):
        super(UnifiedInteractionBlocks, self).__init__()
        self.num_layers = num_layers
        self.truncation_start_layer = truncation_start_layer
        self.truncation_ratio = truncation_ratio
        self.k1 = k1
        self.k2 = k2

        self.stu_layers = nn.ModuleList([
            SequentialTransductionUnit(
                token_dim=token_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                net_dropout=net_dropout
            )
            for _ in range(num_layers)
        ])

    def _expand_valid_mask(self, x, valid_mask):
        """
        The input valid_mask only overwrites the history T.
        The actual length of sequence_emb is T+1 (the last one is the candidate).
        Unified token requires an additional user token.
        """
        if valid_mask is None:
            return None

        B = x.size(0)
        device = x.device
        valid_mask = valid_mask.bool()

        user_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
        candidate_valid = torch.ones(B, 1, dtype=torch.bool, device=device)

        # [user][history][candidate]
        return torch.cat([user_valid, valid_mask, candidate_valid], dim=1)

    def truncate_tail(self, x, valid_mask):
        """
        reserve:
        - 1 user token
        - recent truncation_ratio ratio of sequence tokens
        """
        if self.truncation_ratio is None:
            return x, valid_mask

        B, S, D = x.shape
        seq_len = S - 1
        if seq_len <= 1:
            return x, valid_mask

        keep_seq_len = max(1, int(seq_len * self.truncation_ratio))
        keep_seq_len = min(keep_seq_len, seq_len)

        x = torch.cat([x[:, :1, :], x[:, -keep_seq_len:, :]], dim=1)

        if valid_mask is not None:
            valid_mask = torch.cat([valid_mask[:, :1], valid_mask[:, -keep_seq_len:]], dim=1)

        return x, valid_mask

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor):
        valid_mask = self._expand_valid_mask(x, valid_mask)

        for i in range(self.num_layers):
            if i == self.truncation_start_layer:
                x, valid_mask = self.truncate_tail(x, valid_mask)

            x = self.stu_layers[i](
                x,
                valid_mask=valid_mask,
                k1=self.k1[i],
                k2=self.k2[i]
            ) + x

        return x


class SequentialTransductionUnit(nn.Module):
    def __init__(self, token_dim, num_heads=2, expansion_factor=4, net_dropout=0.0):
        super(SequentialTransductionUnit, self).__init__()
        assert token_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.token_dim = token_dim

        self.pre_norm = nn.LayerNorm(token_dim)
        self.attn_out_norm = nn.LayerNorm(token_dim)

        hidden_dim = token_dim * expansion_factor

        self.pre_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(net_dropout),
            nn.Linear(hidden_dim, token_dim * 4, bias=False)
        )

        self.post_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(net_dropout),
            nn.Linear(hidden_dim, token_dim)
        )
        self._block_mask_cache = {}

    @disable_torch_compile
    def _get_sparse_block_mask(self, S, device, k1, k2):
        cache_key = (S, device.type, device.index, int(k1), int(k2))
        if cache_key in self._block_mask_cache:
            return self._block_mask_cache[cache_key]

        seq_len = S - 1

        def mask_mod(b, h, q_idx, kv_idx):
            is_user_query = (q_idx == 0)
            is_user_key = (kv_idx == 0)

            user_rule = (kv_idx == 0)

            q_seq_idx = q_idx - 1
            kv_seq_idx = kv_idx - 1

            is_causal = kv_seq_idx <= q_seq_idx
            in_local_window = (q_seq_idx - kv_seq_idx) < k1
            in_recent_queries = q_seq_idx >= (seq_len - k2)

            seq_rule = is_causal & (in_local_window | in_recent_queries)

            return torch.where(
                is_user_query,
                user_rule,
                is_user_key | seq_rule
        )

        try:
            block_mask = create_block_mask(mask_mod, None, None, S, S, device=device, _compile=True)
        except TypeError:
            block_mask = create_block_mask(mask_mod, None, None, S, S, device=device)
        self._block_mask_cache[cache_key] = block_mask
        return block_mask

    def forward(self, x: torch.Tensor, valid_mask=None, k1=32, k2=20):
        B, S, D = x.shape

        norm_x = self.pre_norm(x)

        U, Q, K, V = self.pre_proj(norm_x).chunk(4, dim=-1)

        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        sparse_block_mask = self._get_sparse_block_mask(
            S=S,
            device=x.device,
            k1=k1,
            k2=k2
        )

        A = flex_attention(
            Q,
            K,
            V,
            block_mask=sparse_block_mask,
            scale=1.0 / math.sqrt(self.head_dim)
        )

        A = A.transpose(1, 2).contiguous().view(B, S, D)

        Y = self.post_proj(self.attn_out_norm(A) * F.silu(U))

        if valid_mask is not None:
            Y = Y * valid_mask.detach().unsqueeze(-1).to(dtype=Y.dtype)
        return Y
