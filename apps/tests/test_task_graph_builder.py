import pytest
import networkx as nx
from sitegpt.services.task_graph.task_graph_builder import TaskGraphBuilder


@pytest.fixture
def builder():
    return TaskGraphBuilder()


def test_build_login_graph(builder):
    nlu_result = {"intent": "login", "steps": ["login"], "entities": {}}
    graph = builder.build(nlu_result)
    assert isinstance(graph, nx.DiGraph)
    assert "login_form" in graph.nodes


def test_build_navigation_graph(builder):
    nlu_result = {"intent": "navigate", "steps": ["navigate"], "entities": {}}
    graph = builder.build(nlu_result)
    assert "navigation" in graph.nodes


def test_build_combined_flow(builder):
    nlu_result = {
        "intent": "login_and_navigate",
        "steps": ["login", "navigate"],
        "entities": {},
    }
    graph = builder.build(nlu_result)
    assert "login_form" in graph.nodes
    assert "navigation" in graph.nodes
    assert ("login_form", "navigation") in graph.edges
