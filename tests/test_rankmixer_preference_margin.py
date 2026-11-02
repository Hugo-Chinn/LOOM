import types

import pytest
import torch

from model_zoo.RankMixer import RankMixer


def margin_stub(mode="progressive", negative_mode="click_only"):
    model = RankMixer.__new__(RankMixer)
    torch.nn.Module.__init__(model)
    model.preference_margin_mode = mode
    model.preference_margin_start = 0.0
    model.preference_margin_target = 0.4
    model.preference_positive_label = "buy"
    model.preference_negative_mode = negative_mode
    model.feature_map = types.SimpleNamespace(
        labels=["is_click", "cart", "fav", "buy"]
    )
    return model


def labels(click, cart, fav, buy):
    return [torch.tensor(values, dtype=torch.float32).view(-1, 1)
            for values in (click, cart, fav, buy)]


def test_click_only_pair_masks_exclude_middle_feedback():
    model = margin_stub()
    y_true = labels(
        [0, 1, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    )
    positive, negative = model._preference_pair_masks(y_true)
    assert positive.tolist() == [True, False, False, False]
    assert negative.tolist() == [False, True, False, True]


def test_progressive_margin_rewards_larger_deep_gap():
    model = margin_stub("progressive")
    y_true = labels([0, 1], [0, 0], [0, 0], [1, 0])
    groups = torch.tensor([7, 7])
    good = torch.tensor([[0.2, 0.5, 1.0], [0.0, 0.0, 0.0]])
    bad = torch.tensor([[0.2, 0.5, 0.1], [0.0, 0.0, 0.0]])
    assert model._preference_margin_loss(
        good, y_true, groups
    ) < model._preference_margin_loss(
        bad, y_true, groups
    )


def test_reverse_progressive_reverses_depth_schedule():
    model = margin_stub("reverse_progressive")
    reference = torch.zeros(2, 3)
    assert model._preference_margins(3, reference).tolist() == pytest.approx(
        [0.4, 0.2, 0.0]
    )


def test_fixed_margin_uses_only_final_depth():
    model = margin_stub("fixed")
    y_true = labels([0, 1], [0, 0], [0, 0], [1, 0])
    groups = torch.tensor([7, 7])
    first = torch.tensor([[10.0, -10.0, 0.7], [-10.0, 10.0, 0.0]])
    second = torch.tensor([[-5.0, 5.0, 0.7], [5.0, -5.0, 0.0]])
    assert model._preference_margin_loss(first, y_true, groups) == pytest.approx(
        model._preference_margin_loss(second, y_true, groups)
    )


def test_pairs_are_restricted_to_the_same_user():
    model = margin_stub("progressive")
    y_true = labels([0, 1], [0, 0], [0, 0], [1, 0])
    logits = torch.randn(2, 3, requires_grad=True)
    loss = model._preference_margin_loss(logits, y_true, torch.tensor([1, 2]))
    assert loss.item() == 0.0


def test_sparse_batch_returns_differentiable_zero():
    model = margin_stub()
    y_true = labels([1, 1], [0, 0], [0, 0], [0, 0])
    logits = torch.randn(2, 3, requires_grad=True)
    loss = model._preference_margin_loss(logits, y_true, torch.tensor([1, 1]))
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None
