from bs4 import BeautifulSoup
import networkx as nx


class ActionGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        self.graph.clear()

        for el in soup.find_all(True):
            action = self._infer_action(el)
            if action:
                node_id = self._node_id(el)
                selector = self._build_selector(el)

                self.graph.add_node(
                    node_id,
                    action=action,
                    tag=el.name,
                    selector=selector,
                    attrs=el.attrs,
                    text=el.get_text(strip=True),
                )

                if el.parent and el.parent.name not in ["[document]", "html", "body"]:
                    parent_id = self._node_id(el.parent)
                    self.graph.add_edge(parent_id, node_id)

        return self.graph

    def _infer_action(self, el):
        if el.name == "a" and el.get("href"):
            return "click:navigate"
        if el.name == "button":
            return "click:button"
        if el.name == "input":
            t = el.get("type", "text")
            if t in ["text", "email"]:
                return "input:text"
            if t == "password":
                return "input:password"
            if t == "submit":
                return "submit:form"
        if el.name == "form":
            return "form"
        if el.name == "select":
            return "choose:option"
        if el.name == "iframe":
            return "custom_logic"
        return None

    def _node_id(self, el):
        return f"{el.name}:{hash(str(el)) % 100000}"

    def _build_selector(self, el):
        """Упрощённое правило: id > class > tag"""
        if el.get("id"):
            return f"#{el['id']}"
        if el.get("class"):
            return f".{'.'.join(el['class'])}"
        return el.name
