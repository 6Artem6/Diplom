from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

import networkx as nx
import yaml
from bs4 import BeautifulSoup

# Попытка импортировать guess_action из проекта; если не найдено — используем fallback.
try:
    from app.utils.actions import guess_action  # type: ignore
except Exception:  # pragma: no cover - fallback

    def guess_action(el) -> Optional[str]:
        tag = getattr(el, "name", "")
        if tag == "a":
            return "navigate"
        if tag == "button":
            return "click"
        if tag == "input":
            t = el.get("type", "text")
            if t in ("submit", "button"):
                return "click"
            return "input_value"
        return None


class PageGraphBuilder:
    """
    Базовый сборщик графа страницы: области и элементы.
    Хранит graph (networkx.DiGraph) с узлами и ребрами.
    """

    LOGICAL_TAGS: Set[str] = {
        "header",
        "footer",
        "nav",
        "main",
        "section",
        "article",
        "aside",
        "form",
        "table",
        "ul",
        "ol",
        "dialog",
    }

    INTERACTIVE_TAGS: Set[str] = {"button", "a", "input", "select", "textarea"}

    def __init__(
        self,
        html: str,
        url: str,
        session: Optional[Any] = None,
        visited: Optional[Set[str]] = None,
        patterns_path: Optional[str] = None,
    ) -> None:
        self.html = html
        self.url = url
        self.soup = BeautifulSoup(html, "lxml")
        self.graph: nx.DiGraph = nx.DiGraph()
        self.session = session
        self.visited: Set[str] = visited or set()
        self.seen_hashes: Set[str] = set()

        # Загружаем site_patterns.yml
        self.patterns: Dict[str, Any] = {}
        if patterns_path:
            path = Path(patterns_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.patterns = yaml.safe_load(f) or {}
            else:
                raise FileNotFoundError(
                    f"site_patterns.yml not found at {patterns_path}"
                )

    def build(self) -> nx.DiGraph:
        """
        Построить базовый граф страницы: page -> логические области/элементы.
        """
        page_node = f"page:{self.url}"
        self.graph.add_node(
            page_node,
            type="page",
            url=self.url,
        )

        root = self.soup.find("body") or self.soup.find("html")
        if root:
            self._traverse(root, parent_id=page_node)
        return self.graph

    # --- Рекурсивный обход DOM ---
    def _traverse(self, element, parent_id: str) -> None:
        if not getattr(element, "name", None):
            return

        tag = element.name.lower()
        node_type = self._classify_node(element)

        # Пропускаем незначимые узлы, но продолжаем рекурсию
        if tag not in self.LOGICAL_TAGS and tag not in self.INTERACTIVE_TAGS:
            for child in element.children:
                self._traverse(child, parent_id=parent_id)
            return

        subtree_hash = self._hash_subtree(element)
        node_id = self._node_id(element)

        node_data: Dict[str, Any] = {
            "type": node_type,
            "tag": tag,
            "attrs": self._collect_attrs(element),
            "action": None,
            "duplicate": subtree_hash in self.seen_hashes,
        }

        # Эвристика: предполагаем действие для интерактивных узлов
        if node_type == "element" and tag in self.INTERACTIVE_TAGS:
            try:
                node_data["action"] = guess_action(element)
            except Exception:
                node_data["action"] = None

        self.graph.add_node(node_id, **node_data)
        self.graph.add_edge(parent_id, node_id)

        if subtree_hash not in self.seen_hashes:
            self.seen_hashes.add(subtree_hash)
            for child in element.children:
                self._traverse(child, parent_id=node_id)

    def _classify_node(self, el) -> str:
        tag = (el.name or "").lower()
        classes = el.get("class", []) or []
        role = el.get("role", "") or ""

        if tag == "form":
            return "form"
        if tag == "table":
            return "table"
        if tag in ("ul", "ol"):
            return "list"
        if tag == "dialog" or "modal" in classes or role == "dialog":
            return "modal"
        if tag in self.LOGICAL_TAGS:
            return "section"
        if any(c in ("panel", "card", "sidebar", "container") for c in classes):
            return "panel"
        return "element"

    def _node_id(self, el) -> str:
        ident = el.get("id") or el.get("name") or el.get("class")
        if isinstance(ident, list):
            ident = "-".join(ident)
        ident = ident or ""
        base = f"{el.name}:{ident}" if el.name else "text"
        digest = hashlib.sha1(str(el).encode()).hexdigest()[:12]
        return f"{base}:{digest}"

    def _hash_subtree(self, el) -> str:
        if not getattr(el, "name", None):
            return ""
        # Хэш структуры: тег + дочерние подписи
        children = [self._hash_subtree(c) for c in el.find_all(recursive=False)]
        signature = el.name + "|" + "|".join(sorted(children))
        return hashlib.md5(signature.encode()).hexdigest()

    def _collect_attrs(self, el) -> Dict[str, Any]:
        allowed = {
            "id",
            "name",
            "type",
            "href",
            "src",
            "value",
            "class",
            "placeholder",
            "title",
            "aria-label",
        }
        attrs = {k: v for k, v in el.attrs.items() if k in allowed}
        # добавим tag для удобства сопоставления селекторов
        attrs["tag"] = el.name
        return attrs

    # --- Экспорт графа в JSON (иерархия) ---
    def to_json(self) -> str:
        def serialize(node_id: str) -> Dict[str, Any]:
            data = dict(self.graph.nodes[node_id])
            children = [serialize(c) for c in self.graph.successors(node_id)]
            data["children"] = children
            return data

        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        return json.dumps([serialize(r) for r in roots], indent=2, ensure_ascii=False)

    def to_dict(self) -> List[Dict[str, Any]]:
        return json.loads(self.to_json())

    # --- Вспомогательные методы: ссылки и формы ---
    def extract_links(self) -> List[str]:
        return [a.get("href") for a in self.soup.find_all("a", href=True)]

    def extract_form_actions(self) -> List[str]:
        return [f.get("action") for f in self.soup.find_all("form", action=True)]
