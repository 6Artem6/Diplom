import re
import yaml
from pathlib import Path
from .patterns import FRAMEWORK_PATTERNS, JS_COMPLEXITY_PATTERNS
from .complexity_levels import COMPLEXITY_LEVELS


class PatternAnalyzer:
    def __init__(self, patterns_path: str | None = None):
        self.detected = set()
        self.matches = {}  # {pattern: [list of regexes/keywords]}
        self.site_patterns = {}

        if patterns_path and Path(patterns_path).exists():
            with open(patterns_path, "r", encoding="utf-8") as f:
                self.site_patterns = yaml.safe_load(f).get("patterns", {})

    def analyze_html(self, html: str):
        """Ищем паттерны во всей HTML-странице"""
        self.detected.clear()
        self.matches.clear()

        # --- встроенные паттерны ---
        # Фреймворки
        for framework, regex_list in FRAMEWORK_PATTERNS.items():
            for regex in regex_list:
                if re.search(regex, html, re.IGNORECASE):
                    self.detected.add(framework)
                    self.matches.setdefault(framework, []).append(regex)

        # JS-сложность
        for cat, regex_list in JS_COMPLEXITY_PATTERNS.items():
            for regex in regex_list:
                if re.search(regex, html, re.IGNORECASE):
                    self.detected.add(cat)
                    self.matches.setdefault(cat, []).append(regex)

        # эвристики
        if "iframe" in html.lower():
            self.detected.add("custom_logic")
            self.matches.setdefault("custom_logic", []).append("iframe")

        if re.search(r"WebSocket|RTCDataChannel", html, re.IGNORECASE):
            self.detected.add("websocket_heavy")
            self.matches.setdefault("websocket_heavy", []).append("WebSocket/RTC")

        # --- кастомные из site_patterns.yml ---
        for name, cfg in self.site_patterns.items():
            for indicator in cfg.get("indicators", []):
                if re.search(indicator, html, re.IGNORECASE):
                    self.detected.add(name)
                    self.matches.setdefault(name, []).append(indicator)

        return list(self.detected)

    def estimate_complexity(self):
        """Определяем уровень сложности сайта + причины"""
        for level in reversed(COMPLEXITY_LEVELS):  # начинаем с самого сложного
            criteria = set(level["criteria"])
            if self.detected & criteria:
                return {
                    "id": level["id"],
                    "title": level["title"],
                    "matched": {
                        crit: self.matches.get(crit, [])
                        for crit in self.detected & criteria
                    },
                    "all_detected": list(self.detected),
                }

        # по умолчанию level 0
        return {
            "id": 0,
            "title": COMPLEXITY_LEVELS[0]["title"],
            "matched": {},
            "all_detected": list(self.detected),
        }
