"""Directed arc graph of ONE trip's road network (each trip has its own network).

Two-way roads already exist as twin arcs (reverse direction), so the graph is used
exactly as given: no reverse arcs are added here.
"""
import numpy as np

from features.candidates import twin_ids


class RoadGraph:
    def __init__(self, arcs):
        self.from_node = arcs["from_node"].to_numpy()
        self.to_node = arcs["to_node"].to_numpy()
        self.n_arcs = len(self.from_node)
        self.twin = twin_ids(self.from_node, self.to_node)  # reverse arc or -1
        # arcs sorted by from_node, so "arcs leaving node n" is one slice
        self._order = np.argsort(self.from_node, kind="stable")
        sorted_from = self.from_node[self._order]
        n_nodes = int(max(self.from_node.max(), self.to_node.max())) + 1
        self._start = np.searchsorted(sorted_from, np.arange(n_nodes), side="left")
        self._end = np.searchsorted(sorted_from, np.arange(n_nodes), side="right")
        self._reach_cache: dict[tuple[int, int], dict[int, int]] = {}

    def successors(self, arc_id: int) -> np.ndarray:
        """Arcs whose from_node equals this arc's to_node (includes the twin = a U-turn)."""
        n = self.to_node[arc_id]
        return self._order[self._start[n]:self._end[n]]

    def reachable(self, arc_id: int, max_hops: int = 3) -> dict[int, int]:
        """{arc: fewest arc steps from arc_id} for all arcs within max_hops (0 = itself).
        BFS result is cached per (arc, max_hops): a trip revisits the same arcs constantly."""
        key = (int(arc_id), max_hops)
        hit = self._reach_cache.get(key)
        if hit is not None:
            return hit
        dist = {int(arc_id): 0}
        frontier = [int(arc_id)]
        for h in range(1, max_hops + 1):
            nxt = []
            for a in frontier:
                for b in self.successors(a):
                    b = int(b)
                    if b not in dist:
                        dist[b] = h
                        nxt.append(b)
            frontier = nxt
        self._reach_cache[key] = dist
        return dist

    def hops(self, a: int, b: int, max_hops: int = 3):
        """Fewest arc steps from a to b (0 if equal), or None if more than max_hops."""
        return self.reachable(a, max_hops).get(int(b))
