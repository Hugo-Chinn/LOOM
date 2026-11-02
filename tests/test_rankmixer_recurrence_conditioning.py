import torch
from torch import nn

from model_zoo.RankMixer import RankMixer


def recurrence_stub(mode="none", behavior_adapter_dim=0):
    model = object.__new__(RankMixer)
    nn.Module.__init__(model)
    model.token_dim = 8
    model.depth_conditioning_dim = 6
    model.loop_depth_conditioning = mode
    model.behavior_embedding = nn.Embedding(3, model.token_dim)
    model.depth_channel_scale = (
        nn.Parameter(torch.zeros(4, model.token_dim))
        if mode == "channel_scale" else None
    )
    if mode in {"sinusoidal", "behavior_sinusoidal"}:
        input_dim = model.depth_conditioning_dim
        if mode == "behavior_sinusoidal":
            input_dim += model.token_dim
        model.depth_modulation = nn.Sequential(
            nn.Linear(input_dim, model.token_dim),
            nn.SiLU(),
            nn.Linear(model.token_dim, 3 * model.token_dim),
        )
        nn.init.zeros_(model.depth_modulation[-1].weight)
        nn.init.zeros_(model.depth_modulation[-1].bias)
    else:
        model.depth_modulation = None
    if behavior_adapter_dim:
        model.behavior_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model.token_dim, behavior_adapter_dim, bias=False),
                nn.SiLU(),
                nn.Linear(behavior_adapter_dim, model.token_dim, bias=False),
            )
            for _ in range(3)
        ])
        for adapter in model.behavior_adapters:
            nn.init.zeros_(adapter[-1].weight)
    else:
        model.behavior_adapters = None
    return model


def test_sinusoidal_code_is_deterministic_and_depth_specific():
    model = recurrence_stub("sinusoidal")
    code0 = model._sinusoidal_depth_code(0, torch.device("cpu"), torch.float32)
    code1 = model._sinusoidal_depth_code(1, torch.device("cpu"), torch.float32)
    assert code0.shape == code1.shape == (1, 6)
    assert torch.equal(
        code1,
        model._sinusoidal_depth_code(1, torch.device("cpu"), torch.float32),
    )
    assert not torch.equal(code0, code1)


def test_zero_initialized_depth_modulation_is_identity_preserving():
    conditioned = torch.randn(4, 5, 8)
    previous = torch.randn(4, 5, 8)
    candidate = torch.randn(4, 5, 8)
    for mode in ("sinusoidal", "behavior_sinusoidal"):
        model = recurrence_stub(mode)
        modulated, residual_scale = model._apply_recurrence_conditioning(
            conditioned, depth=2, stage=1
        )
        updated = model._apply_conditioned_update(
            previous, candidate, depth=2, residual_scale=residual_scale
        )
        torch.testing.assert_close(modulated, conditioned)
        torch.testing.assert_close(updated, candidate)


def test_channel_scale_starts_from_baseline_and_receives_gradient():
    model = recurrence_stub("channel_scale")
    previous = torch.randn(4, 5, 8)
    candidate = torch.randn(4, 5, 8)
    updated = model._apply_conditioned_update(previous, candidate, depth=2)
    torch.testing.assert_close(updated, candidate)
    updated.square().mean().backward()
    assert model.depth_channel_scale.grad is not None
    assert model.depth_channel_scale.grad[2].abs().sum() > 0
    assert model.depth_channel_scale.grad[:2].abs().sum() == 0


def test_modulation_output_layer_receives_gradient_at_identity_init():
    model = recurrence_stub("behavior_sinusoidal")
    conditioned = torch.randn(4, 5, 8)
    previous = torch.randn(4, 5, 8)
    candidate = torch.randn(4, 5, 8)
    modulated, residual_scale = model._apply_recurrence_conditioning(
        conditioned, depth=2, stage=1
    )
    updated = model._apply_conditioned_update(
        previous, candidate + modulated, depth=2,
        residual_scale=residual_scale,
    )
    updated.square().mean().backward()
    assert model.depth_modulation[-1].weight.grad is not None
    assert model.depth_modulation[-1].weight.grad.abs().sum() > 0


def test_zero_initialized_behavior_adapter_preserves_input_and_trains_output():
    model = recurrence_stub(behavior_adapter_dim=3)
    conditioned = torch.randn(4, 5, 8)
    adapted = conditioned + model.behavior_adapters[1](conditioned)
    torch.testing.assert_close(adapted, conditioned)
    adapted.square().mean().backward()
    assert model.behavior_adapters[1][-1].weight.grad is not None
    assert model.behavior_adapters[1][-1].weight.grad.abs().sum() > 0

