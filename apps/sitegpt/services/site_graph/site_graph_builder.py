import asyncio
import aiohttp
import json
from collections import defaultdict
from bs4 import BeautifulSoup

from services.crawler import Crawler
from services.page_graph.page_graph_builder_with_events import (
    PageGraphBuilderWithEvents,
)
from services.url_normalizer import URLNormalizer
from services.page_graph.graph_merger import GraphMerger


class SiteGraphBuilder:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.crawler = Crawler(base_url)
        self.url_normalizer = URLNormalizer(base_url)
        self.site_graph: dict = defaultdict(dict)
        self.seen_page_types: set = set()

    async def build_site_graph(self, screenshot_merger: bool = False):
        urls = await self.crawler.crawl_site()
        async with aiohttp.ClientSession() as session:
            tasks = [self.process_url(session, url, screenshot_merger) for url in urls]
            await asyncio.gather(*tasks)
        return self.site_graph

    async def process_url(
        self, session: aiohttp.ClientSession, url: str, screenshot_merger: bool
    ):
        async with session.get(url) as resp:
            if resp.status != 200:
                return
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")

            # Определяем тип страницы
            page_type_key = self.url_normalizer.normalize(url, self.base_url)
            if page_type_key in self.seen_page_types:
                return
            self.seen_page_types.add(page_type_key)

            # Строим граф страницы
            builder = PageGraphBuilderWithEvents(html, url, session=session)
            page_graph = (
                await builder.build_with_visual_merge()
                if screenshot_merger
                else builder.build()
            )

            # Определяем модуль через первый сегмент нормализованного пути
            module_name = page_type_key.strip("/").split("/")[0] or "root"
            module_node = self.site_graph.setdefault(module_name, {})
            page_type_node = module_node.setdefault(page_type_key, {"workspaces": {}})

            # Разбиваем на workspace
            for element_id, data in page_graph.nodes(data=True):
                workspace_name = data.get("workspace", "main")
                workspace = page_type_node["workspaces"].setdefault(workspace_name, {})

                el_id = data.get("id") or element_id
                el_node = workspace.setdefault(
                    el_id,
                    {
                        "tag": data.get("tag"),
                        "classes": data.get("classes", []),
                        "attributes": data.get("attributes", {}),
                        "optional": data.get("optional", False),
                        "scripts": data.get("scripts", []),
                        "api_routes": data.get("api_routes", []),
                    },
                )

                # Объединяем скрипты/API, если элемент уже есть
                el_node["scripts"] = list(
                    set(el_node["scripts"] + data.get("scripts", []))
                )
                el_node["api_routes"] = list(
                    set(el_node["api_routes"] + data.get("api_routes", []))
                )

            print(
                f"[INFO] Processed {url} → module: {module_name}, page_type: {page_type_key}"
            )

    def to_json(self) -> str:
        """Сохраняем полный многоуровневый граф сайта в JSON."""
        return json.dumps(self.site_graph, indent=2, ensure_ascii=False)
