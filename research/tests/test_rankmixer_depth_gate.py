import torch
from torch import nn

from model_zoo.RankMixer import RankMixer


def depth_gate_stub(topology="linear", num_loops=4, num_preloops=0, graph_depths=None):
    model = object.__new__(RankMixer)
    nn.Module.__init__(model)
    model.loop_topology = topology
    model.num_loops = num_loops
    model.num_preloops = num_preloops
    model.graph_depths = graph_depths
    model.depth_gate_base_logit = nn.Parameter(torch.tensor(0.0))
    model.depth_gate_slope_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(0.1))))
    return model


def test_depth_gate_is_monotonic_and_starts_at_half():
    model = depth_gate_stub(num_preloops=2)
    gates = torch.stack([model._depth_loop_gate(depth, torch.float32) for depth in range(6)])
    assert torch.isclose(gates[0], torch.tensor(0.5))
    assert bool(torch.all(gates[1:] <= gates[:-1]))


def test_depth_gate_parameters_receive_gradients():
    model = depth_gate_stub(topology="dag", num_loops=6, num_preloops=2,
                            graph_depths=[0, 2, 1, 2, 2, 1])
    loss = sum(model._depth_loop_gate(depth, torch.float32) for depth in range(5))
    loss.backward()
    assert model.depth_gate_base_logit.grad is not None
    assert model.depth_gate_slope_raw.grad is not None
    assert torch.isfinite(model.depth_gate_base_logit.grad)
    assert torch.isfinite(model.depth_gate_slope_raw.grad)
