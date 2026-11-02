import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_zoo.OneTrans import OneTrans


def test_exact_pattern_pair_and_no_regret_losses():
    # Four samples: for cart, samples 0/1 differ only in cart; samples 2/3
    # differ in both click and cart and therefore form a different exact group.
    y_true = [
        torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
        torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        torch.zeros(4, 1),
        torch.zeros(4, 1),
    ]
    stage = torch.zeros(4, 4, 1, requires_grad=True)
    parent = torch.zeros(4, 4, 1)
    # Positive cart examples get larger local corrections than their exact
    # pattern-matched negatives.
    stage.data[0, 1, 0] = 1.0
    stage.data[1, 1, 0] = 0.0
    stage.data[2, 1, 0] = 0.5
    stage.data[3, 1, 0] = -0.5

    dummy = object.__new__(OneTrans)
    dummy.num_tasks = 4
    pair_loss, regret_loss = OneTrans._pattern_pair_losses(
        dummy, stage, parent, y_true)
    assert torch.isfinite(pair_loss)
    assert torch.isfinite(regret_loss)
    assert pair_loss.item() > 0
    assert regret_loss.item() == 0
    pair_loss.backward()
    assert stage.grad is not None
    assert torch.isfinite(stage.grad).all()
    assert dummy.last_pattern_pair_stats["paired_positive"][1] == 2


def test_pattern_pair_loss_handles_no_pairs():
    y_true = [torch.zeros(3, 1) for _ in range(4)]
    stage = torch.zeros(3, 4, 1, requires_grad=True)
    parent = torch.zeros(3, 4, 1)
    dummy = object.__new__(OneTrans)
    dummy.num_tasks = 4
    pair_loss, regret_loss = OneTrans._pattern_pair_losses(
        dummy, stage, parent, y_true)
    assert pair_loss.item() == 0
    assert regret_loss.item() == 0
    (pair_loss + regret_loss).backward()
    assert stage.grad is not None


def test_explicit_pair_matches_one_to_one_with_score_bins():
    y_true = [
        torch.tensor([[1.0], [1.0], [1.0], [1.0]]),
        torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        torch.zeros(4, 1),
        torch.zeros(4, 1),
    ]
    stage = torch.zeros(4, 4, 1, requires_grad=True)
    parent = torch.zeros(4, 4, 1)
    base = torch.zeros(4, 4, 1)
    # Two exact-pattern cart pairs with clearly positive pair margins.
    stage.data[0, 1, 0] = 1.0
    stage.data[2, 1, 0] = 0.5

    dummy = object.__new__(OneTrans)
    dummy.num_tasks = 4
    dummy.explicit_pair_score_bins = 1
    dummy.explicit_pair_stage_weights = [0.0, 1.0, 0.0, 0.0]
    loss = OneTrans._explicit_pattern_pair_loss(
        dummy, stage, parent, base, y_true)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    loss.backward()
    assert stage.grad is not None
    assert dummy.last_explicit_pair_stats["paired_positive"][1] == 2
