import pytest
from sitegpt.services.query_analyzer import QueryAnalyzer
from sitegpt.services.task_graph.task_graph_builder import TaskGraphBuilder
from services.executor.execution_engine import ExecutionEngine
from sitegpt.services.action_graph.action_graph_builder import ActionGraphBuilder


@pytest.fixture
def analyzer():
    return QueryAnalyzer()


@pytest.fixture
def task_builder():
    return TaskGraphBuilder()


@pytest.fixture
def execution_engine():
    return ExecutionEngine()


@pytest.fixture
def action_graph_builder():
    return ActionGraphBuilder()


def test_full_pipeline_login_navigation(
    analyzer, task_builder, execution_engine, action_graph_builder
):
    query = "Залогинься как admin и перейди на следующую страницу"
    nlu_result = analyzer.analyze(query)
    task_graph = task_builder.build(nlu_result)

    # эмулируем HTML для ActionGraph
    html = """
    <html>
        <body>
            <form><input type="text" /><input type="password" /><button type="submit">Login</button></form>
            <a href="/next">Next</a>
        </body>
    </html>
    """
    action_graph = action_graph_builder.build(html)

    trace = execution_engine.run(task_graph, action_graph)

    assert any("login" in step for step in trace)
    assert any("navigate" in step for step in trace)
    assert trace[0].startswith("Executing")
