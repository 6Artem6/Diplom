import pytest
from sitegpt.services.analyzers.pattern_analyzer import PatternAnalyzer

HTML_CASES = {
    "static_html": ("<html><body><h1>Hello</h1></body></html>", 0),
    "simple_js": ("<form onsubmit='return false;'></form>", 1),
    "jquery": ("<script src='jquery.js'></script><script>$(function(){})</script>", 2),
    "bootstrap": ("<div class='modal fade'></div>", 3),
    "vue_app": ("<div id='app'><router-view></router-view></div>", 4),
    "react_app": ("<div data-reactroot></div>", 4),
    "angular_app": ("<div ng-version='13.0.0'></div>", 4),
    "wordpress": ("<link rel='stylesheet' href='/wp-content/style.css'>", 5),
    "yii2": ("<div id='w0' class='form-group' data-confirm='Are you sure?'></div>", 5),
    "laravel": ("<meta name='csrf-token' content='XSRF-TOKEN'>", 5),
    "magento": ("<body class='cms-index-index cms-home magento'>", 5),
    "graphql": ("<script>fetch('/graphql',{method:'POST'})</script>", 6),
    "websocket": ("<script>var ws = new WebSocket('ws://localhost');</script>", 6),
    "pwa": ("<link rel='manifest' href='/manifest.json'>", 6),
    "custom_logic": ("<iframe src='remote.html'></iframe>", 6),
    "realtime": ("<script>new RTCDataChannel()</script>", 7),
}


@pytest.mark.parametrize(
    "case,expected_level", HTML_CASES.values(), ids=HTML_CASES.keys()
)
def test_pattern_analyzer_levels(case, expected_level):
    analyzer = PatternAnalyzer()
    analyzer.analyze_html(case)
    level = analyzer.estimate_complexity()
    assert (
        level["id"] == expected_level
    ), f"Expected {expected_level}, got {level['id']}"
