import pytest
from sitegpt.services.query_analyzer import QueryAnalyzer
from sitegpt.services.task_graph.task_graph_builder import TaskGraphBuilder
from services.executor.execution_engine import ExecutionEngine


@pytest.fixture
def nlu_pipeline():
    return QueryAnalyzer(), TaskGraphBuilder(), ExecutionEngine()


def test_pipeline_navigation(nlu_pipeline):
    qa, tgb, ee = nlu_pipeline
    query = "Перейди на страницу входа"

    nlu_result = qa.analyze(query)
    task_graph = tgb.build_from_nlu(nlu_result)

    assert "navigate" in [t["action"] for t in task_graph["tasks"]]

    # эмулируем выполнение
    execution_result = ee.execute_task_graph(task_graph)

    assert execution_result["status"] in ("success", "simulated")


def test_pipeline_form_fill(nlu_pipeline):
    qa, tgb, ee = nlu_pipeline
    query = "Введи логин и пароль и нажми Войти"

    nlu_result = qa.analyze(query)
    task_graph = tgb.build_from_nlu(nlu_result)

    actions = [t["action"] for t in task_graph["tasks"]]
    assert "input:text" in actions or "submit:form" in actions
