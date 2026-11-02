"""Smoke + invariant tests for the multi-task routing baselines.

The tests use a synthetic feature map so they can run without the preprocessed
datasets. Beyond "it runs", each test pins one structural property that is easy
to break silently: zero-initialization, mask binarity, the mutual-invisibility
mask, deep-supervision keys and the score-fusion wiring.
"""

import torch
from unittest.mock import patch

from model_zoo import (ParallelIndependentTower, SharedBottom, MMoE, PLE, STEM,
                       MoME, ResFlow, OneRank, LoopMT)

LABELS = ["click", "cart", "fav", "buy"]
BATCH = 6
HISTORY = 4
EMB = 8
TOKEN_DIM = 16


class _StubFeatureMap:
    """Minimal FeatureMap replacement: only what the models actually read."""

    def __init__(self, labels=LABELS):
        self.dataset_id = "stub"
        self.labels = list(labels)
        self.group_id = "user_id"
        self.default_emb_dim = EMB
        self.features = {
            "user_index": {"type": "meta"},
            "item_index": {"type": "meta"},
            "user_id": {"type": "categorical", "vocab_size": 50, "embedding_dim": EMB},
            "gender": {"type": "categorical", "vocab_size": 3, "embedding_dim": EMB},
            "item_id": {"type": "categorical", "vocab_size": 40, "embedding_dim": EMB,
                        "source": "item"},
            "video_category": {"type": "categorical", "vocab_size": 5,
                               "embedding_dim": EMB, "source": "item"},
        }

    def sum_emb_out_dim(self, feature_source=[]):
        total = 0
        for spec in self.features.values():
            if spec["type"] == "meta":
                continue
            total += spec.get("embedding_dim", self.default_emb_dim)
        return total


def _inputs():
    torch.manual_seed(0)
    batch_dict = {
        "user_id": torch.randint(1, 50, (BATCH,)),
        "gender": torch.randint(1, 3, (BATCH,)),
    }
    item_dict = {
        "item_id": torch.randint(1, 40, (BATCH, HISTORY + 1)),
        "video_category": torch.randint(1, 5, (BATCH, HISTORY + 1)),
    }
    mask = torch.ones(BATCH, HISTORY, dtype=torch.bool)
    return (batch_dict, item_dict, mask)


def _labels():
    return [torch.randint(0, 2, (BATCH, 1)).float() for _ in LABELS]


def _build(model_cls, **overrides):
    params = dict(
        num_tasks=len(LABELS),
        embedding_dim=EMB,
        token_dim=TOKEN_DIM,
        num_layers=1,
        num_ns_token=3,
        tokenizer_type="Chunk",
        tower_hidden_units=[16, 8],
        verbose=0,
        model_root="/tmp/unirank_mtl_tests/",
        metrics=["AUC"],
        loss=["binary_crossentropy"] * len(LABELS),
        enable_torch_compile=False,
        gradient_checkpointing=False,
    )
    params.update(overrides)
    torch.manual_seed(1234)
    return model_cls(_StubFeatureMap(), **params)


ALL_MODELS = [SharedBottom, ParallelIndependentTower, MMoE, PLE, STEM, MoME,
              ResFlow, OneRank, LoopMT]


def _run(model):
    inputs = _inputs()
    y_true = _labels()
    out = model(inputs)
    assert sorted(k for k in out if not k.startswith("__")) == sorted(
        "{}_pred".format(label) for label in LABELS)
    for label in LABELS:
        pred = out["{}_pred".format(label)]
        assert pred.shape == (BATCH, 1), pred.shape
        assert torch.all((pred >= 0) & (pred <= 1))
    loss = model.add_loss(out, y_true)
    loss.backward()
    grads = [p.grad.abs().sum() for p in model.parameters()
             if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g) for g in grads)
    return out


def test_all_models_forward_backward():
    for model_cls in ALL_MODELS:
        model = _build(model_cls)
        _run(model)


def test_shared_bottom_arms_share_the_backbone():
    """Both shared-bottom arms must sit on the same RankMixer backbone."""
    shared = _build(SharedBottom, head_sharing="shared_first")
    parallel = _build(ParallelIndependentTower)
    for model in (shared, parallel):
        assert model.bottom_type == "mixer"
        assert model.bottom_blocks is not None
        assert model.repr_dim == TOKEN_DIM
    # The only difference is how much the towers share.
    assert shared.head_sharing == "shared_first"
    assert shared.shared_first_layer is not None
    assert len(shared.task_head_tails) == len(LABELS)
    assert len(parallel.tower) == len(LABELS)
    assert getattr(parallel, "task_head_tails", None) is None


def test_shared_bottom_shares_the_first_tower_layer():
    model = _build(SharedBottom, head_sharing="shared_first")
    shared_repr = model.shared_first_layer(torch.randn(BATCH, model.repr_dim))
    tails = [tail(shared_repr) for tail in model.task_head_tails]
    assert all(tail.shape == (BATCH, 1) for tail in tails)
    assert not torch.allclose(tails[0], tails[1])


def test_mmoe_gates_mix_the_same_expert_pool():
    model = _build(MMoE, num_experts=4)
    shared = torch.randn(BATCH, model.repr_dim)
    reprs = model._task_representations(shared)
    assert len(reprs) == 4
    assert not torch.allclose(reprs[0], reprs[1])


def test_ple_stacks_levels():
    model = _build(PLE, num_levels=2, num_shared_experts=2, num_task_experts=2)
    assert len(model.task_experts) == 2 and len(model.shared_experts) == 2
    shared = torch.randn(BATCH, model.repr_dim)
    reprs = model._task_representations(shared)
    assert not torch.allclose(reprs[0], reprs[1])


def test_stem_starts_at_the_shared_bottom_function():
    """Zero-initialized private residuals are an identity at initialization."""
    model = _build(STEM, task_embedding="residual", private_rank=4)
    model.eval()
    for residual in model.private_residuals:
        assert torch.allclose(residual.up.weight,
                              torch.zeros_like(residual.up.weight))
    shared = torch.randn(BATCH, model.repr_dim)
    for residual in model.private_residuals:
        assert torch.allclose(residual(shared), shared, atol=1e-6)
    # Without the embedding axis there is no private direction at all.
    plain = _build(STEM, task_embedding="none")
    assert plain.private_residuals is None


def test_mome_masks_are_binary_at_inference():
    model = _build(MoME, num_experts=4)
    model.eval()
    shared = torch.randn(BATCH, model.repr_dim)
    mask = model._model_mask(shared, 0)
    assert torch.all((mask == 0) | (mask == 1))
    expert_masks = model._expert_masks()
    assert torch.all((expert_masks == 0) | (expert_masks == 1))


def test_resflow_starts_without_residual_flow():
    model = _build(ResFlow)
    for scale in model.residual_scale:
        assert torch.allclose(scale, torch.zeros_like(scale))
    shared = torch.randn(BATCH, model.repr_dim)
    before = model._task_representations(shared)[1]
    with torch.no_grad():
        model.residual_scale[1].fill_(1.0)
    after = model._task_representations(shared)[1]
    assert not torch.allclose(before, after)


def test_resflow_score_fusion_is_additive():
    """Pin the explicit logit-sum variant, not probability multiplication."""
    plain = _build(ResFlow, fusion_mode="none")
    fused = _build(ResFlow, fusion_mode="additive")
    out_plain = plain(_inputs())
    out_fused = fused(_inputs())
    own = [torch.logit(out_plain["{}_pred".format(label)]) for label in LABELS]
    got = [torch.logit(out_fused["{}_pred".format(label)]) for label in LABELS]
    order = fused.task_order
    assert torch.allclose(got[order[0]], own[order[0]], atol=1e-3)
    running = own[order[0]]
    for position in range(1, len(order)):
        task_idx = order[position]
        expected = own[task_idx] + running
        assert torch.allclose(got[task_idx], expected, atol=1e-3)
        running = expected


def test_onerank_task_queries_are_mutually_invisible():
    model = _build(OneRank, head_num_layers=2, num_heads=4)
    assert model.bottom_type == "mixer"
    mask = model.task_mask
    assert mask.shape == (len(LABELS), len(LABELS))
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            if column != row:
                assert bool(mask[row, column]) is True
    for module in model.cross_proj:
        assert torch.allclose(module.weight, torch.zeros_like(module.weight))


def test_loopmt_deep_supervision_and_loop_states():
    model = _build(LoopMT, num_loops=4, stage_embedding_init="zero")
    assert torch.allclose(model.stage_embedding.weight,
                          torch.zeros_like(model.stage_embedding.weight))
    out = model(_inputs())
    aux_keys = sorted(k for k in out if k.startswith("__aux_"))
    assert "__aux_0_1" in aux_keys and "__aux_2_3" in aux_keys
    assert "__aux_3_4" not in aux_keys  # last stage has no deeper stage


def test_loopmt_readout_modes():
    """Progressive readout supervises deeper stages; final readout does not."""
    progressive = _build(LoopMT, num_loops=4, readout="progressive")
    final = _build(LoopMT, num_loops=4, readout="final")
    assert [k for k in progressive(_inputs()) if k.startswith("__aux_")]
    assert not [k for k in final(_inputs()) if k.startswith("__aux_")]


def test_textbook_shared_bottom_equals_parallel():
    shared = _build(SharedBottom)
    parallel = _build(ParallelIndependentTower)
    shared.eval()
    parallel.eval()
    for key, value in shared(_inputs()).items():
        assert torch.equal(value, parallel(_inputs())[key])


def test_loop_auxiliary_loss_uses_mean():
    model = _build(LoopMT)
    out, targets = model(_inputs()), _labels()
    primary = sum(model.loss_fn[i](out[f"{label}_pred"], targets[i], reduction="mean")
                  for i, label in enumerate(LABELS))
    auxiliary = [model.loss_fn[int(key.split("_")[3])](
        value, targets[int(key.split("_")[3])], reduction="mean")
        for key, value in out.items() if key.startswith("__aux_")]
    expected = primary + 0.5 * torch.stack(auxiliary).mean()
    assert torch.allclose(model.add_loss(out, targets), expected)


def test_loop_width_gates_preloops_and_untied_control():
    model = _build(LoopMT, expansion_factor=4, use_loop_gate=True, num_preloops=2)
    assert model.bottom_blocks.pffn_layers[0].W1.shape[-1] == 4 * TOKEN_DIM
    assert torch.allclose(torch.sigmoid(model.loop_gate_logits),
                          torch.full((4,), 0.5))
    assert torch.allclose(torch.sigmoid(model.preloop_gate_logits),
                          torch.full((2,), 0.5))
    untied = _build(LoopMT, share_loop_parameters=False, num_preloops=2)
    ids = [{id(p) for p in block.parameters()} for block in untied.refinement_blocks]
    assert len(ids) == 6
    assert all(not a.intersection(b) for i, a in enumerate(ids) for b in ids[i + 1:])
    _run(untied)


def test_stem_all_forward_task_specific_backward():
    model = _build(STEM)
    shared = torch.randn(BATCH, model.repr_dim)
    baseline = model._task_representations(shared)[0]
    with torch.no_grad():
        next(model.task_experts[1][0].parameters()).add_(0.5)
    assert not torch.allclose(baseline, model._task_representations(shared)[0])
    model.zero_grad()
    model._task_representations(shared)[0].sum().backward()
    assert any(p.grad is not None for p in model.task_experts[0].parameters())
    assert all(p.grad is None for p in model.task_experts[1].parameters())


def test_onerank_default_is_sample_local():
    model = _build(OneRank)
    model.eval()
    inputs = _inputs()
    changed = tuple({k: v.clone() for k, v in x.items()} if isinstance(x, dict)
                    else x.clone() for x in inputs)
    changed[0]["user_id"][1:] = 1
    changed[1]["item_id"][1:] = 1
    with torch.no_grad():
        before, after = model(inputs), model(changed)
    for label in LABELS:
        assert torch.allclose(before[f"{label}_pred"][0], after[f"{label}_pred"][0])


def test_all_parameters_are_registered_with_optimizers():
    for cls in ALL_MODELS:
        model = _build(cls)
        registered = {id(p) for opt in model.optimizers
                      for group in opt.param_groups for p in group["params"]}
        assert registered == {id(p) for p in model.parameters() if p.requires_grad}


def test_compile_never_wraps_structural_containers():
    class CallableWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__("module"), name)

    def compile_callable(module, **kwargs):
        assert not isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict))
        return CallableWrapper(module)

    with patch.object(torch, "compile", side_effect=compile_callable):
        for cls in ALL_MODELS:
            model = _build(cls, enable_torch_compile=True)
            _run(model)


def test_onetrans_all_heads_forward_backward():
    for cls in ALL_MODELS:
        model = _build(cls, bottom_type="onetrans", num_heads=2)
        assert model.repr_dim == TOKEN_DIM * 3
        _run(model)


def test_onetrans_parallel_matches_native_backbone():
    from model_zoo.OneTrans import OneTrans
    native = _build(OneTrans, num_heads=2, expansion_factor=4,
                    task="binary_classification", reduction_ratio=0.5)
    head = _build(ParallelIndependentTower, bottom_type="onetrans", num_heads=2,
                  expansion_factor=4, reduction_ratio=0.5)
    state = {key.replace("unified_interaction_layers.", "bottom_blocks."): value
             for key, value in native.state_dict().items()}
    head.load_state_dict(state, strict=True)
    native.eval()
    head.eval()
    with torch.no_grad():
        expected, actual = native(_inputs()), head(_inputs())
    for label in LABELS:
        assert torch.allclose(expected[f"{label}_pred"], actual[f"{label}_pred"], atol=1e-7)
