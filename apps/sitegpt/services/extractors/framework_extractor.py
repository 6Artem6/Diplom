from services.analyzers.pattern_analyzer import PatternAnalyzer


class FrameworkExtractor:
    def __init__(self, patterns_path="../config/site_patterns.yml"):
        self.analyzer = PatternAnalyzer(patterns_path=patterns_path)

    def extract(self, html: str):
        frameworks = self.analyzer.analyze_html(html)
        complexity = self.analyzer.estimate_complexity()
        return frameworks, complexity
