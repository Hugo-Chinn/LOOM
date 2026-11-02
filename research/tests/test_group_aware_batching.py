import numpy as np


def group_shuffle_order(groups, seed=2026):
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(groups, return_inverse=True)
    rank = np.empty(int(inverse.max()) + 1, dtype=np.int64)
    rank[rng.permutation(len(rank))] = np.arange(len(rank), dtype=np.int64)
    return np.argsort(rank[inverse], kind="stable")


def test_group_shuffle_keeps_each_group_contiguous():
    groups = np.array([1, 2, 1, 3, 2, 1, 4, 3])
    ordered = groups[group_shuffle_order(groups)]
    for group in np.unique(groups):
        positions = np.flatnonzero(ordered == group)
        assert positions[-1] - positions[0] + 1 == len(positions)


def test_group_shuffle_changes_group_order_across_epochs():
    groups = np.repeat(np.arange(20), 2)
    first = groups[group_shuffle_order(groups, seed=1)]
    second = groups[group_shuffle_order(groups, seed=2)]
    assert not np.array_equal(first, second)
