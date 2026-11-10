"""Run with: python -m pytest explorations/relational_supervision/tests -q"""
import pytest
import torch

from loom import LOOM, LOOMConfig
from explorations.relational_supervision.arms import ARMS, expand
from explorations.relational_supervision.losses import (
    RelSupConfig, cross_correlation, decode_keys, encode_key)
from explorations.relational_supervision.portable_model import RelationalLOOM
from explorations.relational_supervision.structure import relations, topology_parents

GRAPH = {"click": [], "cart": ["click"], "fav": ["click"], "buy": ["cart", "fav"]}


def config(**overrides):
    values = dict(tasks=["click", "cart", "fav", "buy"], token_dim=8, num_tokens=3,
                  graph=GRAPH)
    values.update(overrides)
    return LOOMConfig(**values)


def batch(n=16):
    torch.manual_seed(0)
    return torch.randn(n, 3, 8), torch.randint(0, 2, (n, 4)).float()


def test_relations():
    rel = relations([[], [0], [0], [1, 2]])
    assert rel.edges == ((1, 0), (2, 0), (3, 1), (3, 2))
    assert rel.siblings == ((1, 2, 0),)
    assert relations(topology_parents("linear", 4)).siblings == ()
    assert relations([[], [0], [0, 1]]).siblings == ()  # ancestor, not sibling


@pytest.mark.parametrize("preloops", [0, 2])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_zero_weights_equal_core(preloops, diagnostics):
    base = LOOM(config(num_preloops=preloops, base_bypass_tasks=["buy"]))
    model = RelationalLOOM(config(num_preloops=preloops, base_bypass_tasks=["buy"]),
                           RelSupConfig(diagnostics=diagnostics))
    model.load_state_dict(base.state_dict())
    x, y = batch()
    a, b = base(x), model(x)
    torch.testing.assert_close(a["logits"], b["logits"])
    torch.testing.assert_close(base.loss(a, y), model.loss(b, y))
    assert ("relational" in b) == diagnostics
    model.eval()
    assert "relational" not in model(x)


@pytest.mark.parametrize("arm", [name for name in ARMS if name != "full"])
def test_each_arm_is_finite_and_changes_the_loss(arm):
    x, y = batch()
    y[0, 1] = float("nan")
    reference = RelationalLOOM(config(), RelSupConfig())
    model = RelationalLOOM(config(), RelSupConfig(**ARMS[arm]))
    model.load_state_dict(reference.state_dict())
    loss = model.loss(model(x), y)
    assert torch.isfinite(loss)
    assert loss > reference.loss(reference(x), y)
    loss.backward()
    assert model.behavior_embedding.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)
    assert model.last_relational


def test_reference_state_is_detached_by_default():
    x, _ = batch()
    model = RelationalLOOM(config(), RelSupConfig(sibling_margin_weight=1.0))
    reference = model(x)["relational"]["logit", 1, 2, True]
    grads = torch.autograd.grad(reference.sum(), model.behavior_embedding.weight,
                                allow_unused=True)
    assert grads[0] is None
    live = RelationalLOOM(config(), RelSupConfig(sibling_margin_weight=1.0,
                                                 detach_reference=False))
    reference = live(x)["relational"]["logit", 1, 2, False]
    assert torch.autograd.grad(reference.sum(), live.behavior_embedding.weight)[0] is not None


def test_not_applicable_graphs_are_rejected():
    with pytest.raises(ValueError, match="N/A"):
        RelationalLOOM(config(topology="linear", graph={}),
                       RelSupConfig(sibling_margin_weight=0.1))
    RelationalLOOM(config(topology="linear", graph={}), RelSupConfig(edge_gain_weight=0.1))


def test_cross_correlation_ignores_offsets_and_scale():
    torch.manual_seed(0)
    x, noise = torch.randn(512, 8), torch.randn(512, 8)
    same = cross_correlation(x, 3 * x + 5)
    assert same > 5 * cross_correlation(x, noise)
    torch.testing.assert_close(same, cross_correlation(x, x), rtol=1e-4, atol=1e-6)


def test_states_exposed_for_diagnostics():
    model = RelationalLOOM(config(num_preloops=1))
    model.eval()
    out = model(batch()[0], return_states=True)
    assert len(out["states"]) == 4 and out["base"].shape == (16, 3, 8)
    torch.testing.assert_close(
        out["logits"][:, 3], model._readout(out["states"][3], 3, out["base"]))


def test_key_round_trip_and_arm_expansion():
    keys = [("logit", 1, 2, True), ("logit", 3, 3, False), ("delta", 2, 0)]
    encoded = {encode_key(key): index for index, key in enumerate(keys)}
    encoded["click_pred"] = -1
    assert decode_keys(encoded) == {key: index for index, key in enumerate(keys)}
    arms = expand(["full", "sib_margin"], (1.0, 3.0), margin=0.5)
    assert set(arms) == {"full", "sib_margin_m0.5", "sib_margin_x3_m0.5"}
    assert arms["sib_margin_x3_m0.5"] == {"sibling_margin_weight": 0.3,
                                          "sibling_margin": 0.5}
    with pytest.raises(ValueError):
        RelSupConfig(sibling_margin_weight=-1.0)
