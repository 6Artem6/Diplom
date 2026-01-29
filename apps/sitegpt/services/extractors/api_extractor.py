import re
from urllib.parse import urljoin
from typing import List, Set
from bs4 import BeautifulSoup


class ApiExtractor:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def extract(self, soup: BeautifulSoup) -> List[str]:
        scripts = soup.find_all("script")
        api_candidates: Set[str] = set()

        fetch_re = re.compile(r"""fetch\(\s*['"](?P<url>[^'"]+)['"]""")
        xhr_re = re.compile(
            r"""open\(\s*['"](?P<method>GET|POST|PUT|DELETE)['"]\s*,\s*['"](?P<url>[^'"]+)['"]"""
        )
        axios_re = re.compile(
            r"""axios\.(get|post|put|delete)\(\s*['"](?P<url>[^'"]+)['"]"""
        )
        jq_ajax_re = re.compile(
            r"""\$\.ajax\(\s*{[^}]*url\s*:\s*['"](?P<url>[^'"]+)['"]""", re.S
        )

        for script in scripts:
            code = script.string or ""
            for m in fetch_re.finditer(code):
                api_candidates.add(urljoin(self.base_url, m.group("url")))
            for m in xhr_re.finditer(code):
                api_candidates.add(urljoin(self.base_url, m.group("url")))
            for m in axios_re.finditer(code):
                api_candidates.add(urljoin(self.base_url, m.group("url")))
            for m in jq_ajax_re.finditer(code):
                api_candidates.add(urljoin(self.base_url, m.group("url")))

        # формы
        for f in soup.find_all("form"):
            action = f.get("action")
            if action:
                api_candidates.add(urljoin(self.base_url, action))

        return sorted(api_candidates)
