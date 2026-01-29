import pytest
from sitegpt.services.query_analyzer import QueryAnalyzer


@pytest.fixture
def analyzer():
    return QueryAnalyzer()


def test_login_detection(analyzer):
    query = "Войти в систему"
    result = analyzer.analyze(query)
    assert result["intent"] == "login"
    assert "login" in result["steps"]
    assert "username" in result["entities"]


def test_navigation_detection(analyzer):
    query = "Перейди на следующую страницу"
    result = analyzer.analyze(query)
    assert result["intent"] == "navigate"
    assert "navigate" in result["steps"]


def test_login_and_navigation(analyzer):
    query = "Залогинься как admin и перейди дальше"
    result = analyzer.analyze(query)
    assert result["intent"] == "login_and_navigate"
    assert result["entities"]["username"] == "admin"
