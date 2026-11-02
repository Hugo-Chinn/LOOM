import torch
from torch import nn

from model_zoo.RankMixer import RankMixer


def head_stub(mode, num_tasks=3, input_dim=4, hidden_dim=2):
    model = object.__new__(RankMixer)
    nn.Module.__init__(model)
    model.head_sharing = mode
    model.task_head_input_taps = nn.ModuleList([nn.Identity() for _ in range(num_tasks)])
    model.shared_first_layer = None
    model.task_head_tails = None
    model.grouped_towers = None
    model.grouped_first_layers = None
    model.grouped_task_tails = None
    model.grouped_trunks = None
    model.grouped_task_outputs = None
    model.grouped_task_adapters = None
    model.task_to_head_group = None
    if mode == "independent":
        model.tower = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_tasks)])
        model.shared_head_trunk = None
        model.task_output_layers = None
        model.shared_tower = None
    elif mode == "shared_first":
        model.tower = nn.ModuleList()
        model.shared_first_layer = nn.Linear(input_dim, hidden_dim)
        model.task_head_tails = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])
        model.shared_head_trunk = None
        model.task_output_layers = None
        model.shared_tower = None
    elif mode == "grouped":
        model.tower = nn.ModuleList()
        model.grouped_towers = nn.ModuleList([
            nn.Linear(input_dim, 1), nn.Linear(input_dim, 1)
        ])
        model.task_to_head_group = [0, 1, 1]
        model.shared_head_trunk = None
        model.task_output_layers = None
        model.shared_tower = None
    elif mode == "grouped_first":
        model.tower = nn.ModuleList()
        model.grouped_first_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim), nn.Linear(input_dim, hidden_dim)
        ])
        model.grouped_task_tails = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])
        model.task_to_head_group = [0, 1, 1]
        model.shared_head_trunk = None
        model.task_output_layers = None
        model.shared_tower = None
    elif mode in {"grouped_trunk", "grouped_trunk_adapter"}:
        model.tower = nn.ModuleList()
        model.grouped_trunks = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim), nn.Linear(input_dim, hidden_dim)
        ])
        model.grouped_task_outputs = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])
        model.task_to_head_group = [0, 1, 1]
        if mode == "grouped_trunk_adapter":
            model.grouped_task_adapters = nn.ModuleList([
                nn.Identity(), nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.Linear(hidden_dim, hidden_dim, bias=False)
            ])
        model.shared_head_trunk = None
        model.task_output_layers = None
        model.shared_tower = None
    elif mode == "shared_trunk":
        model.tower = nn.ModuleList()
        model.shared_head_trunk = nn.Linear(input_dim, hidden_dim)
        model.task_output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])
        model.shared_tower = None
    else:
        model.tower = nn.ModuleList()
        model.shared_head_trunk = None
        model.task_output_layers = None
        model.shared_tower = nn.Linear(input_dim, 1)
    return model


def test_shared_head_uses_one_readout_for_every_task():
    model = head_stub("shared")
    x = torch.randn(5, 4)
    outputs = [model._task_logit(x, task) for task in range(3)]
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


def test_shared_trunk_keeps_task_specific_output_layers():
    model = head_stub("shared_trunk")
    x = torch.randn(5, 4)
    loss = sum(model._task_logit(x, task).sum() for task in range(3))
    loss.backward()
    assert model.shared_head_trunk.weight.grad is not None
    assert all(layer.weight.grad is not None for layer in model.task_output_layers)
    assert len({id(layer.weight) for layer in model.task_output_layers}) == 3


def test_shared_first_keeps_task_specific_tails():
    model = head_stub("shared_first")
    x = torch.randn(5, 4)
    loss = sum(model._task_logit(x, task).sum() for task in range(3))
    loss.backward()
    assert model.shared_first_layer.weight.grad is not None
    assert all(tail.weight.grad is not None for tail in model.task_head_tails)
    assert len({id(tail.weight) for tail in model.task_head_tails}) == 3


def test_grouped_heads_share_only_declared_task_group():
    model = head_stub("grouped")
    x = torch.randn(5, 4)
    outputs = [model._task_logit(x, task) for task in range(3)]
    assert torch.equal(outputs[1], outputs[2])
    assert not torch.equal(outputs[0], outputs[1])


def test_grouped_first_shares_first_layer_but_keeps_task_tails():
    model = head_stub("grouped_first")
    x = torch.randn(5, 4)
    loss = sum(model._task_logit(x, task).sum() for task in range(3))
    loss.backward()
    assert model.grouped_first_layers[1].weight.grad is not None
    assert model.grouped_task_tails[1].weight.grad is not None
    assert model.grouped_task_tails[2].weight.grad is not None
    assert id(model.grouped_task_tails[1].weight) != id(model.grouped_task_tails[2].weight)


def test_grouped_trunk_keeps_task_outputs():
    model = head_stub("grouped_trunk")
    x = torch.randn(5, 4)
    loss = sum(model._task_logit(x, task).sum() for task in range(3))
    loss.backward()
    assert model.grouped_trunks[1].weight.grad is not None
    assert model.grouped_task_outputs[1].weight.grad is not None
    assert model.grouped_task_outputs[2].weight.grad is not None


def test_grouped_trunk_adapter_routes_task_specific_residuals():
    model = head_stub("grouped_trunk_adapter")
    x = torch.randn(5, 4)
    outputs = [model._task_logit(x, task) for task in range(3)]
    assert outputs[1].shape == outputs[2].shape == (5, 1)
    assert model.grouped_task_adapters[1] is not model.grouped_task_adapters[2]


def test_independent_heads_do_not_share_parameters():
    model = head_stub("independent")
    parameter_ids = [
        {id(parameter) for parameter in tower.parameters()}
        for tower in model.tower
    ]
    assert parameter_ids[0].isdisjoint(parameter_ids[1])
    assert parameter_ids[1].isdisjoint(parameter_ids[2])
