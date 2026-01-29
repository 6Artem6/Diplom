import pytest
from sitegpt.services.action_graph.action_graph_builder import ActionGraphBuilder


def test_build_simple_html():
    html = """
    <html><body>
        <a href="/page">Link</a>
        <button>Click Me</button>
        <input type="text" name="username"/>
    </body></html>
    """
    agb = ActionGraphBuilder()
    graph = agb.build(html)

    assert len(graph.nodes) >= 3
    actions = [data["action"] for _, data in graph.nodes(data=True)]
    assert "click:navigate" in actions
    assert "click:button" in actions
    assert "input:text" in actions


def test_get_actions_by_element():
    html = "<button>Test</button>"
    agb = ActionGraphBuilder()
    graph = agb.build(html)

    node_id = list(graph.nodes)[0]
    action = graph.nodes[node_id]["action"]

    assert action == "click:button"
    assert agb.get_actions_by_element(node_id) == graph.get(node_id, [])
