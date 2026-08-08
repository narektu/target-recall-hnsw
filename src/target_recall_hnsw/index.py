from graph import Graph, Node
from level import LevelSampler
from search import search_layer
from heuristics import select_neighbors_heuristic
from distance import l2_squared

class HNSWIndex:
    def __init__(self, M: int = 16, ef_construction: int = 200, seed: int | None = None, distance_fn=l2_squared):
        self.M = M
        self.m_max0 = 2 * M
        self.ef_construction = ef_construction
        self.graph = Graph()
        self.level_sampler = LevelSampler(M, seed)
        self.distance_fn = distance_fn
        self._next_id = 0

    def insert(self, vector) -> int:
        node_id = self._next_id
        self._next_id += 1
        level = self.level_sampler.sample()
        node = Node(id=node_id, vector=vector, level=level)
        self.graph.add_node(node)

        if self.graph.entry_point is None:
            self.graph.entry_point = node_id
            self.graph.top_layer = level
            return node_id

        entry_points = {self.graph.entry_point}
        for layer in range(self.graph.top_layer, level, -1):
            entry_points = {search_layer(self.graph, vector, entry_points, ef=1, layer=layer, distance_fn=self.distance_fn)[0][1]}

        for layer in range(min(self.graph.top_layer, level), -1, -1):
            candidates = search_layer(self.graph, vector, entry_points, ef=self.ef_construction, layer=layer, distance_fn=self.distance_fn)
            neighbors = select_neighbors_heuristic(vector, candidates, self.M, self.graph, self.distance_fn)
            for n in neighbors:
                self.graph.connect(node_id, n, layer)

            max_degree = self.m_max0 if layer == 0 else self.M
            for n in neighbors:
                n_neighbors = self.graph.nodes[0].neighbors[layer]
                if len(n_neighbors) > max_degree:
                    ranked = sorted(
                        ((self.distance_fn(self.graph.nodes[n].vector, self.graph.nodes[m].vector), m) for m in n_neighbors),
                    )
                    pruned = select_neighbors_heuristic(self.graph.nodes[n].vector, ranked, max_degree, self.graph,
                                                        self.distance_fn)
                    for m in list(n_neighbors):
                        if m not in pruned:
                            self.graph.disconnect(n, m, layer)

            entry_points = {c[1] for c in candidates}
        if level > self.graph.top_layer:
            self.graph.top_layer = level
            self.graph.entry_point = node_id

        return node_id

    def knn_search(self, query, k: int, ef_search: int | None = None):
        ef_search = ef_search or max(k, self.ef_construction // 2)
        entry_points = {self.graph.entry_point}
        for layer in range(self.graph.top_layer, 0, -1):
            entry_points = {search_layer(self.graph, query, entry_points, ef=1, layer=layer, distance_fn=self.distance_fn)[0][1]}
            results = search_layer(self.graph, query, entry_points, ef=ef_search, layer=0, distance_fn=self.distance_fn)
            return results[:k]
