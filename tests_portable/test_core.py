import pytest
import torch
from loom import LOOM, LOOMConfig


def config(**overrides):
    values = dict(tasks=["click", "cart", "fav", "buy"], token_dim=8, num_tokens=3,
                  graph={"click": [], "cart": ["click"], "fav": ["click"],
                         "buy": ["cart", "fav"]})
    values.update(overrides)
    return LOOMConfig(**values)


def test_forward_backward_and_auxiliary_semantics():
    model = LOOM(config(num_preloops=2))
    output = model(torch.randn(5, 3, 8))
    assert output["logits"].shape == (5, 4)
    assert set(output["auxiliary"]) == {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)}
    y = torch.randint(0, 2, (5, 4)).float()
    y[0, 0] = float("nan")
    loss = model.loss(output, y)
    loss.backward()
    assert model.behavior_embedding.weight.grad is not None
    assert torch.isfinite(loss)
    model.eval()
    assert not model(torch.randn(2, 3, 8))["auxiliary"]


def test_no_aux_preserves_predictions_and_main_loss():
    full = LOOM(config())
    no_aux = LOOM(config(supervision="progressive"))
    no_aux.load_state_dict(full.state_dict())
    x = torch.randn(5, 3, 8)
    a, b = full(x), no_aux(x)
    torch.testing.assert_close(a["logits"], b["logits"])
    assert not b["auxiliary"]
    y = torch.randint(0, 2, (5, 4)).float()
    expected = sum(torch.nn.functional.binary_cross_entropy_with_logits(
        b["logits"][:, i], y[:, i]) for i in range(4))
    torch.testing.assert_close(no_aux.loss(b, y), expected)


def test_no_behavior_embedding_really_removes_parameters():
    model = LOOM(config(behavior_conditioning=False))
    assert model.behavior_embedding is None
    assert not any("behavior_embedding" in name for name, _ in model.named_parameters())


def test_untied_blocks_include_preloops_and_all_receive_gradients():
    model = LOOM(config(shared_update=False, num_preloops=2))
    assert model.shared_block is None
    assert len(model.blocks) == 6
    ids = [{id(p) for p in block.parameters()} for block in model.blocks]
    assert all(not ids[i].intersection(ids[j]) for i in range(6) for j in range(i))
    output = model(torch.randn(4, 3, 8))
    model.loss(output, torch.randint(0, 2, (4, 4)).float()).backward()
    assert all(block.w1[0].grad is not None for block in model.blocks)


def test_mean_softmax_first_and_branch_merge():
    states = [torch.zeros(2, 3, 8), torch.ones(2, 3, 8),
              torch.full((2, 3, 8), 3.0), None]
    base = states[0]
    for fusion in ("mean", "softmax"):
        model = LOOM(config(parent_fusion=fusion))
        torch.testing.assert_close(model._merge(3, states, base), torch.full_like(base, 2.0))
    assert LOOM(config(parent_fusion="first"))._merge(3, states, base) is states[1]
    branch = LOOM(config(topology="branch", graph={}))
    torch.testing.assert_close(branch._merge(3, states, base), torch.full_like(base, 2.0))


def test_zero_residual_bypass_is_identity():
    base = LOOM(config())
    bypass = LOOM(config(base_bypass_tasks=["buy"]))
    bypass.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(4, 3, 8)
    torch.testing.assert_close(base(x)["logits"], bypass(x)["logits"])


def test_task_order_need_not_be_topological():
    model = LOOM(config(tasks=["buy", "fav", "click", "cart"]))
    assert model.order.index(2) < model.order.index(0)
    assert model(torch.randn(2, 3, 8))["logits"].shape == (2, 4)


@pytest.mark.parametrize("overrides", [
    {"graph": {"click": ["buy"], "cart": ["click"], "fav": ["click"], "buy": ["cart"]}},
    {"tasks": ["click", "click"]}, {"token_dim": 0}, {"parent_fusion": "invalid"},
    {"task_loss_weights": [1]}, {"behavior_conditioning": "false"},
    {"base_bypass_tasks": ["unknown"]}, {"loop_gate_init": 1.0},
    {"graph": {}}, {"topology": "linear"},
])
def test_invalid_configuration_fails(overrides):
    with pytest.raises(ValueError):
        LOOM(config(**overrides))


def test_mixer_matches_historical_operator():
    import importlib.util
    from pathlib import Path
    from loom.core import TokenMixerBlock
    root = Path(__file__).resolve().parents[1]
    path = root / "unirank/pytorch/layers/interactions/token_mix.py"
    spec = importlib.util.spec_from_file_location("historical_mix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    portable = TokenMixerBlock(config())
    x = torch.randn(4, 3, 8)
    torch.testing.assert_close(portable.mix(x), module.MultiHeadTokenMixing(8, 3)(x))


def test_invalid_labels_and_input():
    model = LOOM(config())
    with pytest.raises(ValueError):
        model(torch.randn(2, 8))
    output = model(torch.randn(2, 3, 8))
    with pytest.raises(ValueError):
        model.loss(output, torch.full((2, 4), float("nan")))
    with pytest.raises(ValueError):
        model.loss(output, torch.full((2, 4), 2.0))


def test_entire_mixer_block_matches_historical_with_identical_weights():
    import ast
    import importlib.util
    from pathlib import Path
    from loom.core import TokenMixerBlock
    root = Path(__file__).resolve().parents[1]
    operators = {}
    for filename, name in [("token_mix.py", "MultiHeadTokenMixing"),
                           ("per_token_ffn.py", "PerTokenFeedForward")]:
        path = root / "unirank/pytorch/layers/interactions" / filename
        spec = importlib.util.spec_from_file_location(filename[:-3], path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        operators[name] = getattr(module, name)
    tree = ast.parse((root / "model_zoo/RankMixer.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "RankMixerBlock")
    namespace = {"torch": torch, "nn": torch.nn, **operators}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "historical_block", "exec"),
         namespace)
    c = config(update_layers=2)
    historical = namespace["RankMixerBlock"](8, 3, 2, expand=2)
    portable = TokenMixerBlock(c)
    portable.mixer_norms.load_state_dict(historical.mixer_norms.state_dict())
    portable.ffn_norms.load_state_dict(historical.pffn_norms.state_dict())
    with torch.no_grad():
        for i in range(2):
            for target, source in ((portable.w1, "W1"), (portable.w2, "W2"),
                                   (portable.b1, "b1"), (portable.b2, "b2")):
                target[i].copy_(getattr(historical.pffn_layers[i], source))
    x = torch.randn(5, 3, 8)
    torch.testing.assert_close(portable(x), historical(x))
