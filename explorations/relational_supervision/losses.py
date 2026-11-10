"""Relational supervision (RelSup) loss terms. Pure torch; no model imports.

Tensor dictionary keys produced by `collect` and consumed by `relational_loss`:
  ("logit", task, node, detached) -> [B] logit of head `task` on state `node`
  ("delta", node, parent)         -> [B, D] pooled increment H_node - H_parent
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

WEIGHTS = ("sibling_margin_weight", "sibling_decor_weight",
           "edge_gain_weight", "edge_distill_weight")
KEY_PREFIX = "__relsup"


@dataclass
class RelSupConfig:
    sibling_margin_weight: float = 0.0
    sibling_margin: float = 0.2
    sibling_decor_weight: float = 0.0
    edge_gain_weight: float = 0.0
    edge_gain_margin: float = 0.2
    edge_distill_weight: float = 0.0
    detach_reference: bool = True
    diagnostics: bool = True

    def __post_init__(self):
        for name in WEIGHTS + ("sibling_margin", "edge_gain_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("detach_reference", "diagnostics"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

    @property
    def active(self):
        return any(getattr(self, name) > 0 for name in WEIGHTS)

    def to_dict(self):
        return asdict(self)


def validate_applicable(config, rel):
    """Reject arms that are N/A on this graph instead of silently doing nothing."""
    if (config.sibling_margin_weight > 0 or config.sibling_decor_weight > 0) \
            and not rel.siblings:
        raise ValueError("sibling terms are N/A: this graph has no sibling pair")
    if (config.edge_gain_weight > 0 or config.edge_distill_weight > 0) and not rel.edges:
        raise ValueError("edge terms are N/A: this graph has no parent/child edge")


def _wanted(config, name):
    return config.diagnostics or getattr(config, name) > 0


def collect(config, rel, logit_fn, delta_fn):
    """Compute every tensor the enabled terms need, inside the model forward.

    logit_fn(task, node, detached) -> [B] logit of head `task` on state `node`,
        with the state detached when `detached` is true.
    delta_fn(node, parent) -> [B, D] pooled state increment.
    """
    tensors = {}
    reference_detached = config.detach_reference

    def need(task, node, detached):
        key = ("logit", task, node, detached)
        if key not in tensors:
            tensors[key] = logit_fn(task, node, detached)

    if _wanted(config, "sibling_margin_weight"):
        for a, b, _ in rel.siblings:
            for own, other in ((a, b), (b, a)):
                need(own, own, False)
                need(own, other, reference_detached)
    if _wanted(config, "edge_gain_weight"):
        for child, parent in rel.edges:
            need(child, child, False)
            need(child, parent, reference_detached)
    if _wanted(config, "edge_distill_weight"):
        for child, parent in rel.edges:
            need(parent, child, False)
            need(parent, parent, False)
    if _wanted(config, "sibling_decor_weight"):
        for a, b, parent in rel.siblings:
            for node in (a, b):
                key = ("delta", node, parent)
                if key not in tensors:
                    tensors[key] = delta_fn(node, parent)
    return tensors


def cross_correlation(x, y, floor=1e-4):
    """Mean squared cross-correlation between batch-standardized features.

    Centering removes constant offsets (for example a behavior embedding), so
    the value cannot be reduced by shifting one sibling. Standardizing makes it
    scale-free, so it cannot be reduced by shrinking increments either.
    """
    x, y = x.float(), y.float()
    if x.shape[0] < 2:
        return None
    x = (x - x.mean(0)) / x.std(0, unbiased=False).clamp_min(floor)
    y = (y - y.mean(0)) / y.std(0, unbiased=False).clamp_min(floor)
    return (x.transpose(0, 1) @ y / x.shape[0]).pow(2).mean()


def relational_loss(config, rel, tensors, targets, observed=None):
    """Return (weighted extra loss or None, {name: detached scalar}).

    targets: [B, K] in [0, 1]; NaN allowed only where `observed` is false.
    Each term is the mean over graph pairs of a per-pair masked mean.
    Terms with zero weight are evaluated without gradient when
    `config.diagnostics` is true, so a baseline run reports the same scalars.
    """
    if observed is None:
        observed = torch.ones_like(targets, dtype=torch.bool)
    reference_detached = config.detach_reference

    def logit(task, node, detached):
        return tensors["logit", task, node, detached].float().reshape(-1)

    logs = {}

    def margin_term(pairs, margin, gap_name):
        losses, gaps = [], []
        for task, reference in pairs:
            mask = observed[:, task]
            if not mask.any():
                continue
            sign = 2 * targets[mask, task].float() - 1
            gap = sign * (logit(task, task, False)[mask]
                          - logit(task, reference, reference_detached)[mask])
            losses.append(F.softplus(margin - gap).mean())
            gaps.append(gap.detach().mean())
        if not losses:
            return None
        logs[gap_name] = torch.stack(gaps).mean()
        return torch.stack(losses).mean()

    def sibling_margin():
        pairs = [pair for a, b, _ in rel.siblings for pair in ((a, b), (b, a))]
        return margin_term(pairs, config.sibling_margin, "sibling_gap")

    def edge_gain():
        return margin_term(list(rel.edges), config.edge_gain_margin, "edge_gap")

    def edge_distill():
        values = []
        for child, parent in rel.edges:
            teacher_logit = logit(parent, parent, False).detach()
            teacher = torch.sigmoid(teacher_logit)
            student = logit(parent, child, False)
            # KL(teacher || student): cross entropy minus teacher entropy.
            values.append(
                F.binary_cross_entropy_with_logits(student, teacher)
                - F.binary_cross_entropy_with_logits(teacher_logit, teacher))
        return torch.stack(values).mean() if values else None

    def sibling_decor():
        values = [cross_correlation(tensors["delta", a, parent],
                                    tensors["delta", b, parent])
                  for a, b, parent in rel.siblings]
        values = [value for value in values if value is not None]
        return torch.stack(values).mean() if values else None

    total = None
    for name, function in (("sibling_margin", sibling_margin),
                           ("sibling_decor", sibling_decor),
                           ("edge_gain", edge_gain),
                           ("edge_distill", edge_distill)):
        weight = getattr(config, name + "_weight")
        if weight > 0:
            value = function()
        elif config.diagnostics:
            with torch.no_grad():
                value = function()
        else:
            continue
        if value is None:
            continue
        logs[name] = value.detach()
        if weight > 0:
            total = weight * value if total is None else total + weight * value
    return total, logs


def encode_key(key):
    """Tuple key -> string key for frameworks whose outputs are string dicts."""
    return "|".join([KEY_PREFIX] + [str(int(p)) if not isinstance(p, str) else p
                                    for p in key])


def decode_keys(mapping):
    """Extract RelSup tensors from a string-keyed output dictionary."""
    tensors = {}
    for name, value in mapping.items():
        if not isinstance(name, str) or not name.startswith(KEY_PREFIX + "|"):
            continue
        parts = name.split("|")[1:]
        if parts[0] == "logit":
            tensors["logit", int(parts[1]), int(parts[2]), bool(int(parts[3]))] = value
        else:
            tensors["delta", int(parts[1]), int(parts[2])] = value
    return tensors
