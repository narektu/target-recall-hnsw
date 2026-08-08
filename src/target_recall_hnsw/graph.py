from dataclasses import dataclass, field
import numpy as np

@dataclass
class Node:
    id: int
    vector: np.ndarray
    level: int
    neighbors: dict[int, set[int]] = field(default_factory=dict)

class Graph:
    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.entry_point: int | None = None
        self.top_layer: int = -1

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        for layer in range(node.level + 1):
            node.neighbors.setdefault(layer, set())

    def connect(self, a: int, b: int, layer: int) -> None:
        self.nodes[a].neighbors[layer].discard(b)
        self.nodes[b].neighbors[layer].discard(a)

    def disconnect(self, a: int, b: int, layer: int) -> None:
        self.nodes[a].neighbors[layer].discard(b)
        self.nodes[b].neighbors[layer].discard(a)
