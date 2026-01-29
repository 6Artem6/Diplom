import asyncio
from playwright.async_api import async_playwright


async def fetch_html(url: str, timeout: int = 10000) -> str:
    """
    Загружает страницу через Chromium (Playwright) и возвращает HTML.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto(url, timeout=timeout, wait_until="networkidle")
        content = await page.content()
        await browser.close()
        return content
