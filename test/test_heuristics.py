import numpy as np
import pytest
from src.target_recall_hnsw.heuristics import select_neighbors_heuristic
from src.target_recall_hnsw.distance import l2_squared


class Node:
    def __init__(self, vector):
        self.vector = np.array(vector, dtype=float)


class Graph:
    def __init__(self, nodes_dict):
        self.nodes = nodes_dict


def test_select_neighbors_heuristic_domination():
    """
    Two candidates on the same side of the query.
    C1 is closer to the query. C2 is farther from the query and very close to C1,
    so C2 should be dominated by C1 and discarded.
    """
    query = np.array([0.0, 0.0])

    # Node 1 vector: [1.0, 0.0] -> dist_to_q squared = 1.0
    # Node 2 vector: [1.1, 0.0] -> dist_to_q squared = 1.21
    # Distance between Node 1 and Node 2 squared = 0.1^2 = 0.01
    # Since 0.01 < 1.21 (dist of node 2 to query), Node 2 is dominated by Node 1.
    nodes = {
        1: Node([1.0, 0.0]),
        2: Node([1.1, 0.0])
    }
    graph = Graph(nodes)

    # Candidates must be sorted nearest-first: (dist_to_q, cand_id)
    candidates = [
        (1.0, 1),
        (1.21, 2)
    ]

    selected = select_neighbors_heuristic(query, candidates, m=2, graph=graph)

    # Only node 1 should be selected; node 2 is discarded due to domination logic
    assert selected == [1]


def test_select_neighbors_heuristic_opposite_sides():
    """
    Two candidates on opposite sides of the query.
    Neither should dominate the other, so both must be kept to preserve diversity.
    """
    query = np.array([0.0, 0.0])

    # Node 1 at [1.0, 0.0] -> dist_to_q = 1.0
    # Node 2 at [-1.0, 0.0] -> dist_to_q = 1.0
    # Distance between Node 1 and Node 2 = 2.0 -> squared distance = 4.0
    # Since 4.0 < 1.0 is False, Node 2 is NOT dominated by Node 1.
    nodes = {
        1: Node([1.0, 0.0]),
        2: Node([-1.0, 0.0])
    }
    graph = Graph(nodes)

    candidates = [
        (1.0, 1),
        (1.0, 2)
    ]

    selected = select_neighbors_heuristic(query, candidates, m=2, graph=graph)

    # Both candidates should be kept because they are on opposite sides
    assert set(selected) == {1, 2}
