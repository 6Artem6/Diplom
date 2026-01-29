import networkx as nx


class PageGraphStats:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.stats = {}

    def compute(self) -> dict:
        nodes = list(self.graph.nodes(data=True))
        total = len(nodes)
        if total == 0:
            return {}

        depths = [attrs.get("depth", 0) for _, attrs in nodes]
        wrappers = [
            n
            for n, attrs in nodes
            if attrs.get("tag") in {"div", "span"} and not attrs.get("role")
        ]
        interactives = [
            n
            for n, attrs in nodes
            if attrs.get("tag") in {"button", "input", "a", "select", "textarea"}
        ]
        hidden = [
            n
            for n, attrs in nodes
            if attrs.get("style", "").find("display:none") >= 0
            or attrs.get("style", "").find("visibility:hidden") >= 0
            or not attrs.get("text")
        ]

        self.stats = {
            "total_nodes": total,
            "max_depth": max(depths) if depths else 0,
            "wrapper_ratio": len(wrappers) / total,
            "interactive_ratio": len(interactives) / total,
            "hidden_ratio": len(hidden) / total,
        }
        return self.stats


def need_vision_analyzer(stats: dict) -> bool:
    return (
        stats.get("max_depth", 0) > 10
        or stats.get("wrapper_ratio", 0) > 0.6
        or stats.get("hidden_ratio", 0) > 0.4
        or stats.get("interactive_ratio", 1) < 0.05
    )
