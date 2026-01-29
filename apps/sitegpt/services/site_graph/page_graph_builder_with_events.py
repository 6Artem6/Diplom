from services.analyzer.pattern_analyzer import PatternAnalyzer
from services.extractors.events_extractor import EventsExtractor
from services.extractors.framework_extractor import FrameworkExtractor
from services.extractors.api_extractor import ApiExtractor
from services.extractors.validations_extractor import ValidationsExtractor


class PageGraphBuilderWithEvents(PageGraphBuilder):
    """
    Наследник PageGraphBuilder: добавляет анализ JS-событий, детекцию фреймворка,
    извлечение API-маршрутов и извлечение простых JS-валидаторов.
    """

    def __init__(self, html, url, session=None, visited=None):
        super().__init__(html=html, url=url, session=session, visited=visited)
        self.script_events = []
        self.frameworks = []
        self.api_routes = []
        self.js_validations = []
        self.complexity = None
        # подключаем YAML-паттерны
        self.analyzer = PatternAnalyzer(patterns_path="../config/site_patterns.yml")

    def build(self):
        page_node = f"page:{self.url}"
        super().build()

        # Фреймворк + сложность
        self.frameworks = self.analyzer.analyze_html(self.html)
        self.complexity = self.analyzer.estimate_complexity()

        # События
        self.script_events = EventExtractor().extract(self.soup)
        self.attach_script_events()

        # API-маршруты
        self.api_routes = ApiExtractor(self.url).extract(self.soup)

        # Валидации
        self.js_validations = ValidationsExtractor().extract(self.soup)

        if page_node in self.graph.nodes:
            self.graph.nodes[page_node].update(
                {
                    "frameworks": self.frameworks,
                    "complexity": self.complexity,
                    "api_routes": self.api_routes,
                    "js_validations": self.js_validations,
                }
            )
        return self.graph

    async def build_with_visual_merge(self, screenshot_path: str = None):
        g = self.build()

        if screenshot_path:
            async with aiohttp.ClientSession() as session:
                with open(screenshot_path, "rb") as f:
                    form = aiohttp.FormData()
                    form.add_field(
                        "file", f, filename="screenshot.png", content_type="image/png"
                    )
                    async with session.post(VISION_ANALYZER_URL, data=form) as resp:
                        detections = await resp.json()

            merger = GraphMerger(rules={"merge_tags": ["section", "article"]})
            g = merger.merge(g, visual_blocks=detections["detections"])

        return g

    # ---------- Удобный экспорт в JSON (включает meta данные) ----------
    def to_json(self) -> str:
        """
        Переопределяем, чтобы гарантированно включить framework/api/js_validations в корень.
        """
        # убедимся, что анализ выполнен
        if not self.graph.nodes:
            self.build()

        def serialize(node_id: str) -> Dict[str, Any]:
            data = dict(self.graph.nodes[node_id])
            children = [serialize(c) for c in self.graph.successors(node_id)]
            data["children"] = children
            return data

        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        return json.dumps([serialize(r) for r in roots], indent=2, ensure_ascii=False)

    # --- Опциональные: follow_links и submit_forms (с использованием self.session, если есть) ---
    def follow_links(
        self, url_patterns: Optional[List[str]] = None, max_depth: int = 2
    ) -> None:
        """
        Рекурсивно переходит по ссылкам и добавляет их графы (если передан session).
        Осторожно: network IO — вызывается при наличии self.session.
        """
        if not self.session:
            return
        # простой стек для обхода
        stack: List[tuple[str, int]] = [(self.url, 0)]
        while stack:
            cur_url, depth = stack.pop()
            if depth >= max_depth or cur_url in self.visited:
                continue
            try:
                resp = self.session.get(cur_url, timeout=10)
                if resp.status_code != 200:
                    continue
                self.visited.add(cur_url)
                builder = PageGraphBuilderWithEvents(
                    resp.text, cur_url, session=self.session, visited=self.visited
                )
                builder.build()
                # объединяем графы: переносим узлы и ребра
                self._merge_graph(builder.graph)
                # добавляем линковку дальше
                for link in builder.extract_links():
                    full = urljoin(cur_url, link)
                    if url_patterns and not any(p in full for p in url_patterns):
                        continue
                    if full not in self.visited:
                        stack.append((full, depth + 1))
            except Exception:
                continue

    def _merge_graph(self, other: nx.DiGraph) -> None:
        """
        Простое слияние графов (без устранения дублей по содержимому).
        """
        for n, d in other.nodes(data=True):
            if n not in self.graph:
                self.graph.add_node(n, **d)
        for u, v in other.edges():
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v)

    def submit_forms(self) -> None:
        """
        Отправляет формы с автозаполнением обязательных полей (только для сбора данных, без изменения важных данных).
        Требует self.session.
        """
        if not self.session:
            return
        for form in self.soup.find_all("form"):
            action = form.get("action") or self.url
            method = form.get("method", "get").lower()
            form_url = urljoin(self.url, action)
            data: Dict[str, Any] = {}
            for inp in form.find_all(["input", "select", "textarea"]):
                name = inp.get("name")
                if not name:
                    continue
                if (
                    inp.get("required") is not None
                    or inp.get("aria-required") == "true"
                ):
                    itype = inp.get("type", "text")
                    if itype == "email":
                        data[name] = "test@example.com"
                    elif itype == "number":
                        data[name] = "1"
                    else:
                        data[name] = inp.get("value") or "test"
            try:
                if method == "post":
                    resp = self.session.post(form_url, data=data, timeout=10)
                else:
                    resp = self.session.get(form_url, params=data, timeout=10)
                if resp.status_code == 200:
                    # можно добавить новый граф по результату
                    builder = PageGraphBuilderWithEvents(
                        resp.text, form_url, session=self.session, visited=self.visited
                    )
                    builder.build()
                    self._merge_graph(builder.graph)
            except Exception:
                continue
