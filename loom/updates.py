"""Named, shape-preserving update factories for architecture exploration."""
from collections.abc import Callable

from torch import nn


_FACTORIES: dict[str, Callable] = {}


def register_update(name, factory):
    """Register factory(config) -> fresh nn.Module before constructing LOOM.

    Custom registrations must also run before loading a checkpoint. Never
    import arbitrary Python modules named in untrusted YAML.
    """
    if not isinstance(name, str) or not name or not callable(factory):
        raise ValueError("an update needs a nonempty name and callable factory")
    if name in _FACTORIES:
        raise ValueError(f"update already registered: {name}")
    _FACTORIES[name] = factory


def available_updates():
    return tuple(sorted(_FACTORIES))


def build_update(config):
    if config.update_type not in _FACTORIES:
        raise ValueError(f"unknown update {config.update_type!r}; "
                         f"registered: {available_updates()}")
    block = _FACTORIES[config.update_type](config)
    if not isinstance(block, nn.Module):
        raise TypeError("update factory must return an nn.Module")
    return block


def _mixer(config):
    if config.update_options:
        raise ValueError("mixer has no update_options")
    from .core import TokenMixerBlock
    return TokenMixerBlock(config)


class AttentionUpdate(nn.Module):
    """Exploratory pre-norm self-attention update, NOT a validated LOOM recipe."""

    def __init__(self, config):
        super().__init__()
        if set(config.update_options) - {"num_heads"}:
            raise ValueError("attention update_options only accepts num_heads")
        heads = config.update_options.get("num_heads", 4)
        if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
            raise ValueError("num_heads must be a positive integer")
        if config.token_dim % heads:
            raise ValueError("attention token_dim must be divisible by num_heads")
        d, h = config.token_dim, config.token_dim * config.expansion_factor
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(config.update_layers)])
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d, heads, dropout=config.dropout, batch_first=True)
            for _ in range(config.update_layers)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(config.update_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(config.dropout),
                          nn.Linear(h, d)) for _ in range(config.update_layers)])

    def forward(self, x):
        for norm, attention, ffn_norm, ffn in zip(
                self.norms, self.attention, self.ffn_norms, self.ffns):
            normalized = norm(x)
            x = x + attention(normalized, normalized, normalized, need_weights=False)[0]
            x = x + ffn(ffn_norm(x))
        return x


register_update("mixer", _mixer)
register_update("attention", AttentionUpdate)
