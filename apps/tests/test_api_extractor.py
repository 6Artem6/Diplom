from sitegpt.services.extractors.api_extractor import ApiExtractor
from bs4 import BeautifulSoup


def test_extracts_api_routes():
    html = """
    <script>
        fetch("/api/data");
        axios.post("/api/save");
    </script>
    """
    soup = BeautifulSoup(html, "html.parser")
    extractor = ApiExtractor("http://localhost")

    routes = extractor.extract(soup)
    assert "/api/data" in routes
    assert "/api/save" in routes
