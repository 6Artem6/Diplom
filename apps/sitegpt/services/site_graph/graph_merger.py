import networkx as nx
from typing import Dict, Any, List


class GraphMerger:
    def __init__(self, rules: Dict[str, Any] = None):
        self.rules = rules or {}

    def merge(
        self, page_graph: nx.DiGraph, visual_blocks: List[Dict] = None
    ) -> nx.DiGraph:
        """Схлопывает граф на основе правил и опционально визуальных блоков"""
        merged_graph = page_graph.copy()

        # 1. Схлопывание по правилам
        for node in list(merged_graph.nodes):
            attrs = merged_graph.nodes[node]
            if attrs.get("tag") in self.rules.get("merge_tags", []):
                preds = list(merged_graph.predecessors(node))
                succs = list(merged_graph.successors(node))
                merged_graph.remove_node(node)
                for p in preds:
                    for s in succs:
                        merged_graph.add_edge(p, s)

        # 2. Схлопывание по визуальным блокам
        if visual_blocks:
            for vb in visual_blocks:
                vb_nodes = [
                    n
                    for n, data in merged_graph.nodes(data=True)
                    if self._is_inside(vb["bbox"], data.get("bbox"))
                ]
                if vb_nodes:
                    main_node = vb_nodes[0]
                    for n in vb_nodes[1:]:
                        nx.contracted_nodes(
                            merged_graph, main_node, n, self_loops=False
                        )

        return merged_graph

    def _is_inside(self, outer, inner):
        """Проверка, что bbox inner находится внутри bbox outer"""
        if not outer or not inner:
            return False
        x1, y1, x2, y2 = outer
        ix1, iy1, ix2, iy2 = inner
        return x1 <= ix1 and y1 <= iy1 and x2 >= ix2 and y2 >= iy2

    # ---------- Удобный экспорт ----------
    def to_json(self) -> Dict[str, Any]:
        data = {
            "nodes": {n: attrs for n, attrs in self.graph.nodes(data=True)},
            "edges": list(self.graph.edges()),
        }
        return data
