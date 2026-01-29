from fastapi import APIRouter, Body, Query
from app.services.graph_builder import PageGraphBuilder
from app.services.fetcher import fetch_html
from app.services.crawler import Crawler

router = APIRouter()


@router.post("/graph")
async def build_graph(html: str = Body(..., embed=True), url: str = Body(...)):
    builder = PageGraphBuilder(html, url)
    builder.build()
    return {"url": url, "graph": builder.to_dict()}


@router.get("/graph_from_url")
async def graph_from_url(url: str = Query(..., description="URL страницы")):
    html = await fetch_html(url)
    builder = PageGraphBuilder(html, url)
    builder.build()
    return {"url": url, "graph": builder.to_dict()}


@router.get("/crawl")
async def crawl(url: str = Query(...), depth: int = Query(2)):
    crawler = Crawler(url, max_depth=depth)
    graphs = await crawler.crawl_site()
    return graphs
