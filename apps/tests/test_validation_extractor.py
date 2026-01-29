from sitegpt.services.extractors.validation_extractor import ValidationExtractor
from bs4 import BeautifulSoup


def test_extract_validations():
    html = """
    <form>
        <input name="email" type="email" required pattern=".+@.+">
    </form>
    """
    soup = BeautifulSoup(html, "html.parser")
    extractor = ValidationExtractor()

    validations = extractor.extract(soup)
    assert any("email" in v["field"] for v in validations)
    assert any("required" in v["rules"] for v in validations)
    assert any("pattern" in v["rules"] for v in validations)
