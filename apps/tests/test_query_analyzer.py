import pytest
from sitegpt.services.query_analyzer import QueryAnalyzer


@pytest.fixture
def analyzer():
    return QueryAnalyzer()


def test_analyze_simple_navigation(analyzer):
    query = "Открой страницу входа"
    result = analyzer.analyze(query)

    assert result["intent"] == "navigate"
    assert any("login" in step["target"].lower() for step in result["steps"])


def test_analyze_form_fill(analyzer):
    query = "Заполни форму логина"
    result = analyzer.analyze(query)

    assert result["intent"] == "fill_form"
    assert result["entities"]  # должны быть выделены поля
