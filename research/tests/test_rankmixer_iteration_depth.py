from model_zoo.RankMixer import RankMixer


def stub(topology, num_loops, num_preloops=0, graph_depths=None):
    model = object.__new__(RankMixer)
    model.loop_topology = topology
    model.num_loops = num_loops
    model.num_preloops = num_preloops
    model.graph_depths = graph_depths
    return model


def test_linear_iteration_depths_follow_call_order():
    model = stub("linear", num_loops=4, num_preloops=2)
    assert [model._behavior_iteration_depth(i) for i in range(4)] == [2, 3, 4, 5]
    assert model._max_behavior_depth() == 3


def test_branch_iteration_depths_share_parallel_middle_level():
    model = stub("branch", num_loops=4, num_preloops=1)
    assert [model._behavior_iteration_depth(i) for i in range(4)] == [1, 2, 2, 3]
    assert model._max_behavior_depth() == 2


def test_dag_iteration_depths_use_graph_longest_path():
    model = stub("dag", num_loops=6, num_preloops=2, graph_depths=[0, 2, 1, 2, 2, 1])
    assert [model._behavior_iteration_depth(i) for i in range(6)] == [2, 4, 3, 4, 4, 3]
    assert model._max_behavior_depth() == 2
