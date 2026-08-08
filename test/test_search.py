import numpy as np
import pytest
from src.target_recall_hnsw.search import search_layer


class Node:
    def __init__(self, vector, neighbors=None):
        self.vector = np.array(vector, dtype=float)
        self.neighbors = neighbors or {}  # Словарь вида {layer: [neighbor_ids]}


class Graph:
    def __init__(self):
        self.nodes = {}


def test_search_layer_correctness():
    graph = Graph()
    graph.nodes = {
        0: Node([0.0, 0.0], neighbors={0: [1, 2]}),
        1: Node([1.0, 0.0], neighbors={0: [0, 3]}),
        2: Node([0.0, 1.0], neighbors={0: [0, 3]}),
        3: Node([2.0, 2.0], neighbors={0: [1, 2]}),
    }

    query = np.array([1.9, 1.9])
    entry_points = {0}
    ef = 2
    layer = 0

    results = search_layer(graph, query, entry_points, ef=ef, layer=layer)

    assert len(results) <= ef
    assert len(results) > 0

    closest_dist, closest_id = results[0]
    assert closest_id == 3

    expected_d3 = np.sum((query - graph.nodes[3].vector) ** 2)
    assert np.isclose(closest_dist, expected_d3, atol=1e-5)
