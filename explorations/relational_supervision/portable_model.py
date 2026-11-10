"""RelSup on the portable core, by subclassing `loom.LOOM` (core untouched)."""
import torch
from torch import nn

from loom import LOOM

from .losses import RelSupConfig, collect, relational_loss, validate_applicable
from .structure import relations


class RelationalLOOM(LOOM):
    """`loom.LOOM` plus optional sibling and parent/child loss terms.

    With every weight at zero the logits and the loss equal the core's.
    States are captured through the `_update` hook, so graph execution,
    merging, gating and readouts remain the core implementation.
    """

    def __init__(self, config, relsup=None):
        super().__init__(config)
        self.relsup = relsup or RelSupConfig()
        self.relations = relations(self.parents)
        validate_applicable(self.relsup, self.relations)
        self._states = None
        self._base = None
        self.last_relational = {}

    def _update(self, previous, stage, depth, behavior=True):
        result = super()._update(previous, stage, depth, behavior)
        if self._states is not None:
            if behavior:
                self._states[stage] = result
            else:
                self._base = result
        return result

    def forward(self, tokens, return_auxiliary=None, return_states=False):
        self._states, self._base = [None] * len(self.tasks), None
        try:
            output = super().forward(tokens, return_auxiliary)
            states = self._states
            base = self._base if self.config.num_preloops else tokens
        finally:
            self._states, self._base = None, None
        wanted = self.training if return_auxiliary is None else return_auxiliary
        if wanted and (self.relsup.active or self.relsup.diagnostics):
            output["relational"] = collect(
                self.relsup, self.relations,
                lambda task, node, detached: self._readout(
                    states[node].detach() if detached else states[node], task, base),
                lambda node, parent: states[node].mean(1) - states[parent].mean(1))
        if return_states:
            output["states"], output["base"] = states, base
        return output

    def loss(self, output, targets):
        total = super().loss(output, targets)
        self.last_relational = {}
        if "relational" in output:
            extra, logs = relational_loss(self.relsup, self.relations,
                                          output["relational"], targets,
                                          torch.isfinite(targets))
            self.last_relational = {name: float(value) for name, value in logs.items()}
            if extra is not None:
                total = total + extra
        return total


class RelationalTabular(nn.Module):
    """Same example encoder and construction order as `loom.cli.TabularLOOM`."""

    def __init__(self, input_dim, config, relsup=None):
        super().__init__()
        self.encoder = nn.Linear(input_dim, config.num_tokens * config.token_dim)
        self.core = RelationalLOOM(config, relsup)

    def forward(self, x, **kwargs):
        c = self.core.config
        return self.core(self.encoder(x).reshape(-1, c.num_tokens, c.token_dim), **kwargs)
