import pytest
from bs4 import BeautifulSoup
from services.executor.execution_engine import ExecutionEngine
from services.executor.browser_adapter import BrowserAdapter
from services.executor.trace_logger import TraceLogger
from sitegpt.services.action_graph.action_graph_builder import ActionGraphBuilder


@pytest.fixture
def sample_html():
    return """
    <html>
        <body>
            <form>
                <input type="text" name="username" />
                <input type="password" name="password" />
                <button type="submit">Login</button>
            </form>
            <a href="/next">Next Page</a>
        </body>
    </html>
    """


def test_execution_engine_with_action_graph(sample_html):
    # 1. Строим граф действий
    agb = ActionGraphBuilder()
    graph = agb.build(sample_html)
    actions = [
        graph.nodes[n]["action"] for n in graph.nodes if "action" in graph.nodes[n]
    ]

    assert "input:text" in actions
    assert "input:password" in actions
    assert "click:button" in actions
    assert "click:navigate" in actions

    # 2. Создаем движок с мокнутыми адаптером и логгером
    browser = BrowserAdapter()
    logger = TraceLogger()
    engine = ExecutionEngine(browser, logger)

    # 3. Выполняем действия (эмулируем workflow)
    workflow = [
        {
            "action": "input:text",
            "selector": "input[name=username]",
            "value": "test_user",
        },
        {
            "action": "input:password",
            "selector": "input[name=password]",
            "value": "secret",
        },
        {"action": "click:button", "selector": "button[type=submit]"},
        {"action": "click:navigate", "selector": "a[href='/next']"},
    ]
    engine.execute_sequence(workflow)

    # 4. Проверяем логи
    logs = logger.get_logs()
    assert any("input:text" in entry for entry in logs)
    assert any("input:password" in entry for entry in logs)
    assert any("click:button" in entry for entry in logs)
    assert any("click:navigate" in entry for entry in logs)
