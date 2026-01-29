from sitegpt.services.extractors.event_extractor import EventExtractor
from bs4 import BeautifulSoup


def test_extract_inline_and_add_event_listener():
    html = """
    <button onclick="alert('clicked')">Click</button>
    <script>document.getElementById('btn').addEventListener('click', ()=>{})</script>
    """
    soup = BeautifulSoup(html, "html.parser")
    extractor = EventExtractor()

    events = extractor.extract(soup)
    assert any("onclick" in e["event"] for e in events)
    assert any("click" in e["event"] for e in events)
