import heapq
from graph import Graph
from distance import l2_squared

def search_layer(
    graph: Graph, query: "np.ndarray", entry_points: set[int],
    ef: int, layer: int, distance_fn=l2_squared,
) -> list[tuple[float, int]]:
    visited = set(entry_points)
    candidates = []
    results = []

    for ep in entry_points:
        d = distance_fn(query, graph.nodes[ep].vector)
        heapq.heappush(candidates, (d, ep))
        heapq.heappush(results, (-d, ep))

    while candidates:
        cur_dist, cur_id = heapq.heappop(candidates)
        worst_result_dist = -results[0][0]

        if cur_dist > worst_result_dist and len(results) >= ef:
            break

        for neighbor_id in graph.nodes[cur_id].neighbors.get(layer, ()):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            d = distance_fn(query, graph.nodes[neighbor_id].vector)
            worst_result_dist = -results[0][0]

            if len(results) < ef or d < worst_result_dist:
                heapq.heappush(candidates, (d, neighbor_id))
                heapq.heappush(results, (-d, neighbor_id))
                if len(results) > ef:
                    heapq.heappop(results)

    return sorted([(-d, i) for d, i in results])
