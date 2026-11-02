"""Regression checks for historical supervision and deterministic CPU training."""
import torch
from loom import LOOM, LOOMConfig


def test_legacy_branch_supervision_includes_sibling_stage():
    config = LOOMConfig(tasks=["click", "cart", "fav", "buy"], token_dim=8,
                        num_tokens=4, topology="branch")
    model = LOOM(config)
    output = model(torch.randn(3, 4, 8))
    assert set(output["auxiliary"]) == {
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    config = LOOMConfig(tasks=["click", "cart", "fav", "buy"], token_dim=8,
                        num_tokens=4,
                        graph={"click": [], "cart": ["click"], "fav": ["click"],
                               "buy": ["cart", "fav"]})
    assert (1, 2) not in LOOM(config)(torch.randn(3, 4, 8))["auxiliary"]


def test_same_seed_cpu_optimizer_steps_are_identical():
    def fit():
        torch.manual_seed(20262027)
        config = LOOMConfig(tasks=["click", "buy"], token_dim=8, num_tokens=4,
                            topology="linear", num_preloops=2)
        model = LOOM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        x = torch.randn(8, 4, 8)
        y = torch.randint(0, 2, (8, 2)).float()
        for _ in range(3):
            optimizer.zero_grad()
            model.loss(model(x), y).backward()
            optimizer.step()
        return model.state_dict()
    a, b = fit(), fit()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
