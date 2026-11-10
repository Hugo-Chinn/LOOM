"""Graph relations used by relational supervision. No torch dependency."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Relations:
    edges: tuple      # (child, parent)
    siblings: tuple   # (a, b, shared_parent) with a < b


def topology_parents(topology, num_tasks, graph_parents=None):
    """Parent lists in task order for linear / branch / dag topologies."""
    if topology == "linear":
        return [[]] + [[i - 1] for i in range(1, num_tasks)]
    if topology == "branch":
        if num_tasks != 4:
            raise ValueError("branch topology requires four ordered tasks")
        return [[], [0], [0], [1, 2]]
    if topology == "dag":
        if graph_parents is None:
            raise ValueError("dag topology needs graph parents")
        return [list(p) for p in graph_parents]
    raise ValueError(f"unknown topology {topology!r}")


def relations(parents):
    """Edges and sibling pairs of a behavior graph.

    Siblings share at least one parent and neither is an ancestor of the
    other. The first shared parent (in the first node's declared order) is the
    reference state for increment-based terms.
    """
    n = len(parents)
    ancestors = [None] * n

    def resolve(node, trail=()):
        if node in trail:
            raise ValueError("behavior graph contains a cycle")
        if ancestors[node] is None:
            found = set()
            for parent in parents[node]:
                found.add(parent)
                found |= resolve(parent, trail + (node,))
            ancestors[node] = found
        return ancestors[node]

    for node in range(n):
        resolve(node)
    edges = tuple((child, parent) for child in range(n) for parent in parents[child])
    siblings = []
    for a in range(n):
        for b in range(a + 1, n):
            shared = [p for p in parents[a] if p in parents[b]]
            if shared and a not in ancestors[b] and b not in ancestors[a]:
                siblings.append((a, b, shared[0]))
    return Relations(edges, tuple(siblings))
