import torch

from model_zoo.OneTransCapacity import OneTransCapacity
from test_mtl_baselines import _build, _inputs, _run


def build(**kwargs):
    return _build(OneTransCapacity, num_heads=2, expansion_factor=4,
                  task="binary_classification",
                  model_type="base_refiner", peft_type="film",
                  num_loops=4, supervision_mode="deep_progressive", **kwargs)


def test_capacity_backward_and_optimizer_coverage():
    model = build(refiner_expansion_factor=8, capacity_bypass_tasks=["buy"])
    _run(model)
    covered = {id(p) for group in model.optimizer.param_groups for p in group["params"]}
    assert all(id(p) in covered for p in model.ns_refiner.parameters())
    assert id(model.capacity_bypass_gates["buy"]) in covered


def test_zero_bypass_equivalence():
    plain = build()
    bypass = build(capacity_bypass_tasks=["buy"])
    bypass.load_state_dict(plain.state_dict(), strict=False)
    plain.eval()
    bypass.eval()
    inputs = _inputs()
    with torch.no_grad():
        a, b = plain(inputs), bypass(inputs)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_widening_leaves_backbone_dimensions_unchanged():
    normal, wide = build(), build(refiner_expansion_factor=8)
    assert {k: v.shape for k, v in normal.base_encoder.state_dict().items()} == {
        k: v.shape for k, v in wide.base_encoder.state_dict().items()
    }
    assert sum(p.numel() for p in wide.ns_refiner.parameters()) > sum(
        p.numel() for p in normal.ns_refiner.parameters())
