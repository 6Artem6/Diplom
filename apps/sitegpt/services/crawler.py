from collections import deque
from typing import Dict, Any
from urllib.parse import urlparse, urljoin

from app.services.fetcher import fetch_html
from app.services.graph_builder_with_events import PageGraphBuilderWithEvents
from app.services.url_normalizer import URLNormalizer


class Crawler:
    def __init__(self, base_url: str, max_depth: int = 2):
        self.base_url = base_url.rstrip("/")
        self.normalizer = URLNormalizer(self.base_url)
        self.max_depth = max_depth
        self.visited: set[str] = set()
        self.graphs: Dict[str, Any] = {}

    async def crawl_site(self) -> Dict[str, Any]:
        """Обход сайта через очередь, BFS, с нормализацией URL"""
        queue = deque([(self.base_url, 0)])
        parsed_base = urlparse(self.base_url)
        base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

        while queue:
            url, depth = queue.popleft()
            norm_url = self.normalizer.normalize(url, self.base_url)
            if not norm_url or norm_url in self.visited or depth > self.max_depth:
                continue

            self.visited.add(norm_url)

            try:
                html = await fetch_html(url)
            except Exception as e:
                print(f"[Crawler] Ошибка загрузки {url}: {e}")
                continue

            builder = PageGraphBuilderWithEvents(html, url)
            builder.build()
            # автозаполнение форм
            builder.submit_forms()
            self.graphs[norm_url] = builder.to_dict()

            # собираем ссылки и формы
            next_urls = set(builder.extract_links() + builder.extract_form_actions())
            for link in next_urls:
                full_url = urljoin(base_domain, link)
                parsed = urlparse(full_url)
                if parsed.netloc != parsed_base.netloc:
                    continue  # только внутри домена
                normalized = self.normalizer.normalize(full_url, self.base_url)
                if normalized and normalized not in self.visited:
                    queue.append((full_url, depth + 1))

        return self.graphs
