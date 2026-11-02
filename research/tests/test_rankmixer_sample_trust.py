import torch
from torch import nn

from model_zoo.RankMixer import RankMixer, RecurrenceConditioningLinear


def trust_stub(mode="stage", num_loops=3, token_dim=8, init=0.9):
    model = object.__new__(RankMixer)
    nn.Module.__init__(model)
    model.num_loops = num_loops
    model.token_dim = token_dim
    model.sample_trust_init = init
    model.sample_trust_gate = mode

    def router():
        module = nn.Sequential(
            RecurrenceConditioningLinear(3 * token_dim, 4),
            nn.SiLU(),
            RecurrenceConditioningLinear(4, 1),
        )
        nn.init.zeros_(module[-1].weight)
        nn.init.constant_(module[-1].bias, float(torch.logit(torch.tensor(init))))
        return module

    if mode == "stage":
        model.sample_trust_routers = nn.ModuleList([
            router() for _ in range(num_loops)
        ])
        model.shared_sample_trust_router = None
    elif mode == "shared":
        model.sample_trust_routers = None
        model.shared_sample_trust_router = router()
    else:
        model.sample_trust_routers = None
        model.shared_sample_trust_router = None
    return model


def test_sample_trust_is_exact_identity_at_initialization():
    previous = torch.randn(5, 4, 8)
    proposal = torch.randn(5, 4, 8)
    for mode in ("stage", "shared"):
        model = trust_stub(mode)
        updated, trust = model._apply_sample_trust(previous, proposal, stage=1)
        torch.testing.assert_close(trust, torch.ones_like(trust))
        torch.testing.assert_close(updated, proposal)


def test_sample_trust_output_layer_receives_gradient_at_identity_boundary():
    model = trust_stub("stage")
    previous = torch.randn(5, 4, 8)
    proposal = torch.randn(5, 4, 8)
    updated, _ = model._apply_sample_trust(previous, proposal, stage=1)
    updated.square().mean().backward()
    router = model.sample_trust_routers[1]
    assert router[-1].weight.grad is not None
    assert router[-1].weight.grad.abs().sum() > 0
    assert router[-1].bias.grad is not None
    assert router[-1].bias.grad.abs().sum() > 0


def test_sample_trust_can_only_attenuate_proposal():
    model = trust_stub("shared")
    with torch.no_grad():
        model.shared_sample_trust_router[-1].bias.fill_(-20.0)
    previous = torch.randn(5, 4, 8)
    proposal = torch.randn(5, 4, 8)
    updated, trust = model._apply_sample_trust(previous, proposal, stage=0)
    assert bool(torch.all((trust >= 0) & (trust <= 1)))
    original_step = (proposal - previous).flatten(1).norm(dim=1)
    trusted_step = (updated - previous).flatten(1).norm(dim=1)
    assert bool(torch.all(trusted_step <= original_step + 1e-6))


def test_no_sample_trust_preserves_proposal_and_returns_none():
    model = trust_stub("none")
    previous = torch.randn(5, 4, 8)
    proposal = torch.randn(5, 4, 8)
    updated, trust = model._apply_sample_trust(previous, proposal, stage=0)
    assert trust is None
    torch.testing.assert_close(updated, proposal)

