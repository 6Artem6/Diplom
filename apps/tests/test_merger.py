import networkx as nx
from sitegpt.services.site_graph.graph_merger import GraphMerger


def test_merge_simple():
    g = nx.DiGraph()
    g.add_node("A", tag="div")
    g.add_node("B", tag="section")
    g.add_node("C", tag="p")
    g.add_edges_from([("A", "B"), ("B", "C")])

    merger = GraphMerger(rules={"merge_tags": ["section"]})
    mg = merger.merge(g)

    assert "B" not in mg.nodes
    assert ("A", "C") in mg.edges
