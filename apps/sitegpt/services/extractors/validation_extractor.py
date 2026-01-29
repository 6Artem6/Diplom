import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup


class ValidationExtractor:
    def extract(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        validators: List[Dict[str, Any]] = []
        func_pattern = re.compile(
            r"""function\s+(?P<name>validate\w*)\s*\((?P<arg>[^)]*)\)\s*{(?P<body>[^}]+)}""",
            re.MULTILINE | re.DOTALL,
        )
        selector_pattern = re.compile(
            r"""querySelector\(['"](?P<sel>[^'"]+)['"]\)|
               \$\(['"](?P<jqsel>[^'"]+)['"]\)""",
            re.VERBOSE,
        )

        for script in soup.find_all("script"):
            code = script.string or ""
            for m in func_pattern.finditer(code):
                selectors = set()
                for ms in selector_pattern.finditer(m.group("body")):
                    sel = ms.group("sel") or ms.group("jqsel")
                    if sel:
                        selectors.add(sel)
                validators.append(
                    {
                        "function": m.group("name"),
                        "arg": m.group("arg"),
                        "body": m.group("body"),
                        "selectors": sorted(selectors),
                    }
                )
        return validators
