"""All built-in candidates must satisfy the same recurrent update contract."""
import pytest
import torch
from torch import nn

from loom import LOOM, LOOMConfig, available_updates, register_update


def configuration(name, shared=True):
    return LOOMConfig(tasks=["click", "buy"], token_dim=8, num_tokens=4,
                      topology="linear", num_preloops=2, update_type=name,
                      shared_update=shared)


@pytest.mark.parametrize("name", available_updates())
@pytest.mark.parametrize("shared", [True, False])
def test_update_shape_gradients_sharing_and_checkpoint(name, shared, tmp_path):
    config = configuration(name, shared)
    model = LOOM(config)
    x = torch.randn(5, 4, 8, requires_grad=True)
    out = model(x)
    assert out["logits"].shape == (5, 2)
    model.loss(out, torch.randint(0, 2, (5, 2)).float()).backward()
    blocks = [model.shared_block] if shared else model.blocks
    assert all(any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in block.parameters()) for block in blocks)
    assert torch.isfinite(x.grad).all()
    path = tmp_path / "state.pt"
    torch.save({"config": config.to_dict(), "model": model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = LOOM(LOOMConfig(**saved["config"]))
    restored.load_state_dict(saved["model"])
    model.eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(x)["logits"], restored(x)["logits"])
        assert not restored(x)["auxiliary"]


def test_unknown_and_invalid_update_options_fail():
    with pytest.raises(ValueError, match="unknown update"):
        LOOM(configuration("does_not_exist"))
    config = configuration("attention")
    config.update_options = {"num_heads": 3}
    with pytest.raises(ValueError, match="divisible"):
        LOOM(config)
    config = configuration("mixer")
    config.update_options = {"typo": 1}
    with pytest.raises(ValueError, match="no update_options"):
        LOOM(config)


def test_custom_factory_cannot_reuse_untied_parameters():
    block = nn.Linear(8, 8)
    register_update("test_reused", lambda config: block)
    with pytest.raises(ValueError, match="reused parameters"):
        LOOM(configuration("test_reused", shared=False))
    with pytest.raises(ValueError, match="already registered"):
        register_update("test_reused", lambda config: block)


def test_bad_shape_factory_fails_at_forward():
    class BadUpdate(nn.Module):
        def forward(self, x):
            return x[:, 0, :]
    register_update("test_bad_shape", lambda config: BadUpdate())
    with pytest.raises(ValueError, match="preserve"):
        LOOM(configuration("test_bad_shape"))(torch.randn(2, 4, 8))
