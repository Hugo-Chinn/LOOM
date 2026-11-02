import torch
from torch import nn

from model_zoo.RankMixer import RankMixer


def test_residual_bypass_zero_init_is_identity():
    model = object.__new__(RankMixer)
    nn.Module.__init__(model)
    model.task_token_mode = "none"
    model.base_bypass_mode = "residual"
    model.base_bypass_logits = nn.Parameter(torch.zeros(2))
    model.base_bypass_mask = torch.ones(2)
    state = torch.randn(4, 3, 8)
    base = torch.randn(4, 3, 8)
    actual = model._head_representation(state, 1, base)
    assert torch.equal(actual, state.mean(dim=1))


def test_dynamic_branch_zero_init_matches_static_half_gates():
    torch.manual_seed(7)
    h_click = torch.randn(4, 3, 8)
    h_cart = torch.randn(4, 3, 8)
    h_fav = torch.randn(4, 3, 8)
    router = nn.Linear(24, 2)
    nn.init.zeros_(router.weight)
    nn.init.zeros_(router.bias)
    dynamic = torch.sigmoid(router(torch.cat([
        h_click.mean(1), h_cart.mean(1), h_fav.mean(1)
    ], dim=-1)))
    dynamic_fused = (
        h_click
        + dynamic[:, 0].view(-1, 1, 1) * (h_cart - h_click)
        + dynamic[:, 1].view(-1, 1, 1) * (h_fav - h_click)
    )
    static_fused = h_click + 0.5 * (h_cart - h_click) + 0.5 * (h_fav - h_click)
    assert torch.equal(dynamic_fused, static_fused)


def test_dynamic_softmax_zero_init_matches_parent_mean():
    torch.manual_seed(11)
    parent_a = torch.randn(4, 3, 8)
    parent_b = torch.randn(4, 3, 8)
    base = torch.randn(4, 3, 8)
    router = nn.Linear(24, 2)
    nn.init.zeros_(router.weight)
    nn.init.zeros_(router.bias)
    logits = router(torch.cat([
        parent_a.mean(1), parent_b.mean(1), base.mean(1)
    ], dim=-1))
    weights = torch.softmax(logits, dim=-1)
    routed = (
        torch.stack([parent_a, parent_b], dim=0)
        * weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
    ).sum(dim=0)
    assert torch.equal(routed, 0.5 * (parent_a + parent_b))


class _StubFeatureMap:
    def __init__(self, labels):
        self.labels = labels


def _stub_model(base_bypass_tasks, labels):
    model = object.__new__(RankMixer)
    nn.Module.__init__(model)
    model.task_token_mode = "none"
    model.base_bypass_mode = "residual"
    model.base_bypass_tasks = base_bypass_tasks
    model.feature_map = _StubFeatureMap(labels)
    return model


def test_bypass_mask_resolution_by_label_and_index():
    model = _stub_model(["is_click"], ["is_click", "is_conversion"])
    assert torch.equal(model._resolve_base_bypass_mask(2), torch.tensor([1.0, 0.0]))
    model = _stub_model([1], ["is_click", "is_conversion"])
    assert torch.equal(model._resolve_base_bypass_mask(2), torch.tensor([0.0, 1.0]))
    model = _stub_model(None, ["is_click", "is_conversion"])
    assert torch.equal(model._resolve_base_bypass_mask(2), torch.ones(2))


def test_bypass_mask_freezes_unselected_tasks():
    model = _stub_model(["b"], ["a", "b"])
    model.base_bypass_logits = nn.Parameter(torch.tensor([0.6, 0.6]))
    model.register_buffer("base_bypass_mask", torch.tensor([0.0, 1.0]))
    state = torch.randn(4, 3, 8)
    base = torch.randn(4, 3, 8)
    out_a = model._head_representation(state, 0, base)
    out_b = model._head_representation(state, 1, base)
    assert torch.equal(out_a, state.mean(dim=1))
    expected_b = state.mean(dim=1) + torch.tanh(torch.tensor(0.6)) * (
        base.mean(dim=1) - state.mean(dim=1)
    )
    assert torch.allclose(out_b, expected_b)
    out_b.sum().backward()
    assert model.base_bypass_logits.grad[0] == 0.0
    assert model.base_bypass_logits.grad[1].abs() > 0.0


def test_task_loss_weights_scale_primary_losses():
    model = object.__new__(RankMixer)
    nn.Module.__init__(model)
    model.num_tasks = 2
    model.feature_map = _StubFeatureMap(["a", "b"])
    model.model_type = "parallel"
    model.supervision_mode = "progressive"
    model.auxiliary_loss_weight = 0.5
    model.loop_step_reg_weight = 0.0
    model.loop_variance_reg_weight = 0.0
    model.sample_trust_reg_weight = 0.0
    model.preference_score_head = None
    model.loss_fn = [
        lambda pred, y_true, reduction="mean": pred,
        lambda pred, y_true, reduction="mean": pred,
    ]
    model.register_buffer("task_loss_weight_tensor", torch.tensor([1.0, 4.0]))
    return_dict = {"a_pred": torch.tensor(2.0), "b_pred": torch.tensor(3.0)}
    assert float(model.add_loss(return_dict, [torch.tensor(0.0), torch.tensor(1.0)])) == 14.0
    model.task_loss_weight_tensor = None
    assert float(model.add_loss(return_dict, [torch.tensor(0.0), torch.tensor(1.0)])) == 5.0


def test_loop_block_layers_sets_refinement_depth():
    from model_zoo.RankMixer import RankMixerBlock
    # 1 layer by default under loop_scope=refinement; explicit knob overrides it
    assert RankMixerBlock(input_dim=32, num_ns_token=4, num_layers=1).num_layers == 1
    assert RankMixerBlock(input_dim=32, num_ns_token=4, num_layers=2).num_layers == 2
    model = object.__new__(RankMixer)
    model.loop_block_layers = None
    assert model._resolve_loop_block_layers(3, 1) == 1
    assert model._resolve_loop_block_layers(3, 3) == 3
    model.loop_block_layers = 2
    assert model._resolve_loop_block_layers(3, 1) == 2
    assert model._resolve_loop_block_layers(3, 3) == 2


def test_child_adapter_node_coverage():
    model = object.__new__(RankMixer)
    model.child_adapter_nodes = "multi_parent"
    assert model._node_has_adapter(1) is False
    assert model._node_has_adapter(2) is True
    model.child_adapter_nodes = "all"
    assert model._node_has_adapter(1) is True
    assert model._node_has_adapter(0) is True
