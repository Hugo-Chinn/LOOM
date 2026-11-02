import torch
from torch import nn
from model_zoo.RankMixer import RankMixer


def test_merge_ablation_routes_first_parent_without_changing_default():
    model = object.__new__(RankMixer)
    model.graph_parents = [[], [], [0, 1]]
    model.dag_parent_fusion = "mean"
    states = [torch.ones(2, 4, 3), torch.full((2, 4, 3), 3.0), None]
    model.disable_branch_merge = False
    assert torch.equal(model._fuse_dag_parents(2, states, states[0]),
                       torch.full((2, 4, 3), 2.0))
    model.disable_branch_merge = True
    assert model._fuse_dag_parents(2, states, states[0]) is states[0]


def test_single_parent_unchanged():
    model = object.__new__(RankMixer)
    model.graph_parents = [[], [0]]
    model.dag_parent_fusion = "mean"
    model.disable_branch_merge = True
    states = [torch.ones(2, 4, 3), None]
    assert torch.equal(model._fuse_dag_parents(1, states, states[0]), states[0])


def test_untied_preloops_use_separate_blocks_and_receive_gradients():
    model = object.__new__(RankMixer)
    nn.Module.__init__(model)
    model.num_loops = 2
    model.num_preloops = 2
    model.loop_block = None
    model.loop_blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(4)])
    model.depth_gate_base_logit = None
    model.preloop_gate_logits = None
    model._condition_on_iteration_depth = lambda h, depth: h
    model._apply_recurrence_conditioning = lambda h, depth, stage: (h, None)
    model._apply_conditioned_update = lambda h, candidate, depth, scale: candidate
    model.activation_checkpoint = lambda block, h: block(h)
    model._run_preloops(torch.ones(2, 4, 3)).sum().backward()
    assert model.loop_blocks[0].weight.grad is None
    assert model.loop_blocks[1].weight.grad is None
    assert model.loop_blocks[2].weight.grad is not None
    assert model.loop_blocks[3].weight.grad is not None
