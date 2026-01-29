from sitegpt.services.extractors.framework_extractor import FrameworkExtractor


def test_framework_extractor_wraps_pattern_analyzer():
    html = "<script src='/static/jquery.min.js'></script>"
    extractor = FrameworkExtractor(html)
    frameworks, complexity = extractor.extract()

    assert "jQuery" in frameworks
    assert isinstance(complexity, dict)
    assert "id" in complexity
