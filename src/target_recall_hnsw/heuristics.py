from distance import l2_squared

def select_neighbors_heuristic(query, candidates: list[tuple[float, int]], m: int, graph, distance_fn=l2_squared) -> list[int]:
    selected: list[int] = []
    for dist_to_q, cand_id in candidates:
        if len(selected) >= m:
            break
        cand_vec = graph.nodes[cand_id].vector
        dominated = any(
            distance_fn(cand_vec, graph.nodes[s].vector < dist_to_q)
            for s in selected
        )
        if not dominated:
            selected.append(cand_id)
    return selected


