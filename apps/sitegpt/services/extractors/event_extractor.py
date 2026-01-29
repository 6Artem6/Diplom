import re
from typing import List, Dict
from bs4 import BeautifulSoup


class EventExtractor:
    EVENT_ATTRS = ["onclick", "onchange", "oninput", "onsubmit"]

    def extract(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        events = []
        scripts = soup.find_all("script")

        qs_pattern = re.compile(
            r"""document\.querySelector\(['"](?P<selector>[^'"]+)['"]\)\.addEventListener\(['"](?P<event>\w+)['"]"""
        )
        jq_on_pattern = re.compile(
            r"""\$\(['"](?P<selector>[^'"]+)['"]\)\.on\(['"](?P<event>\w+)['"]"""
        )
        jq_click_pattern = re.compile(
            r"""\$\(['"](?P<selector>[^'"]+)['"]\)\.(?P<event>click|change|submit)\("""
        )

        for script in scripts:
            code = script.string or ""
            for m in qs_pattern.finditer(code):
                events.append(
                    {"selector": m.group("selector"), "event": m.group("event")}
                )
            for m in jq_on_pattern.finditer(code):
                events.append(
                    {"selector": m.group("selector"), "event": m.group("event")}
                )
            for m in jq_click_pattern.finditer(code):
                events.append(
                    {"selector": m.group("selector"), "event": m.group("event")}
                )

        # fallback: inline on* attrs
        for el in soup.find_all(True, attrs=self.EVENT_ATTRS):
            for attr, val in el.attrs.items():
                if attr in self.EVENT_ATTRS:
                    events.append({"selector": self._build_selector(el), "event": attr})

        return events

    @staticmethod
    def _build_selector(el):
        if el.get("id"):
            return f"#{el['id']}"
        if el.get("class"):
            return "." + ".".join(el.get("class"))
        return el.name
