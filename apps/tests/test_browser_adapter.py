import pytest
from unittest.mock import MagicMock
from services.executor.browser_adapter import BrowserAdapter


def test_open_url(monkeypatch):
    mock_playwright = MagicMock()
    mock_page = MagicMock()
    mock_playwright.open_url.return_value = mock_page

    adapter = BrowserAdapter(mock_playwright)
    result = adapter.navigate("http://example.com")

    assert result == mock_page
    mock_playwright.open_url.assert_called_once()


def test_click(monkeypatch):
    mock_playwright = MagicMock()
    adapter = BrowserAdapter(mock_playwright)

    adapter.click("#login")
    mock_playwright.click.assert_called_with("#login")
