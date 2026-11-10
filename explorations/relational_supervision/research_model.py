"""RelSup on the historical Native RankMixer, by subclassing (file untouched).

Config keys (all optional, flat, in the experiment's YAML block):
  relsup_sibling_margin_weight, relsup_sibling_margin,
  relsup_sibling_decor_weight, relsup_edge_gain_weight, relsup_edge_gain_margin,
  relsup_edge_distill_weight, relsup_detach_reference, relsup_diagnostics,
  relsup_log_every
With every weight at zero the training loss equals RankMixer's.
"""
import logging
from pathlib import Path
import sys

import torch

_RESEARCH = str(Path(__file__).resolve().parents[2] / "research")
if _RESEARCH not in sys.path:
    sys.path.insert(0, _RESEARCH)

from model_zoo.RankMixer import RankMixer  # noqa: E402

from .losses import (RelSupConfig, collect, decode_keys, encode_key,  # noqa: E402
                     relational_loss, validate_applicable)
from .structure import relations, topology_parents  # noqa: E402


class RankMixerRelSup(RankMixer):
    def __init__(self, feature_map,
                 relsup_sibling_margin_weight=0.0, relsup_sibling_margin=0.2,
                 relsup_sibling_decor_weight=0.0,
                 relsup_edge_gain_weight=0.0, relsup_edge_gain_margin=0.2,
                 relsup_edge_distill_weight=0.0,
                 relsup_detach_reference=True, relsup_diagnostics=True,
                 relsup_log_every=200, **kwargs):
        super().__init__(feature_map, **kwargs)
        if self.model_type != "looped":
            raise ValueError("RankMixerRelSup requires model_type=looped")
        if self.supervision_mode == "final":
            raise ValueError("RelSup needs per-behavior states; supervision_mode=final "
                             "reads every task from the last state")
        self.relsup = RelSupConfig(
            sibling_margin_weight=float(relsup_sibling_margin_weight),
            sibling_margin=float(relsup_sibling_margin),
            sibling_decor_weight=float(relsup_sibling_decor_weight),
            edge_gain_weight=float(relsup_edge_gain_weight),
            edge_gain_margin=float(relsup_edge_gain_margin),
            edge_distill_weight=float(relsup_edge_distill_weight),
            detach_reference=bool(relsup_detach_reference),
            diagnostics=bool(relsup_diagnostics))
        self.relsup_relations = relations(topology_parents(
            self.loop_topology, self.num_tasks, getattr(self, "graph_parents", None)))
        validate_applicable(self.relsup, self.relsup_relations)
        self.relsup_log_every = int(relsup_log_every)
        self._relsup_cache = None
        self._relsup_step = 0
        logging.info("[RelSup] config=%s edges=%s siblings=%s", self.relsup.to_dict(),
                     self.relsup_relations.edges, self.relsup_relations.siblings)

    def _run_progressive_encoder(self, tokens):
        result = super()._run_progressive_encoder(tokens)
        self._relsup_cache = (result[0], result[2])  # states, base_state
        return result

    def forward(self, inputs):
        self._relsup_cache = None
        return_dict = super().forward(inputs)
        cache, self._relsup_cache = self._relsup_cache, None
        if not self.training or cache is None:
            return return_dict
        if not (self.relsup.active or self.relsup.diagnostics):
            return return_dict
        states, base = cache

        def logit_fn(task, node, detached):
            state = states[node].detach() if detached else states[node]
            return self._task_logit(
                self._head_representation(state, task, base), task).reshape(-1)

        def delta_fn(node, parent):
            return (self._state_representation(states[node], node)
                    - self._state_representation(states[parent], node))

        # Computed here so head evaluations share the forward autocast context;
        # train_step casts every returned tensor to float32 before add_loss.
        for key, value in collect(self.relsup, self.relsup_relations,
                                  logit_fn, delta_fn).items():
            return_dict[encode_key(key)] = value
        return return_dict

    def add_loss(self, return_dict, y_true):
        total = super().add_loss(return_dict, y_true)
        tensors = decode_keys(return_dict)
        if not tensors:
            return total
        targets = torch.cat([y.reshape(-1, 1) for y in y_true], dim=1).float()
        extra, logs = relational_loss(self.relsup, self.relsup_relations,
                                      tensors, targets)
        self._relsup_step += 1
        if self.relsup_log_every > 0 and self._relsup_step % self.relsup_log_every == 0:
            logging.info("[RelSup] step=%d %s", self._relsup_step, " ".join(
                f"{name}={float(value):.6f}" for name, value in sorted(logs.items())))
        return total if extra is None else total + extra
