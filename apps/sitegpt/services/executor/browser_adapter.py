from playwright.async_api import Page


class BrowserAdapter:
    def __init__(self, page: Page):
        self.page = page

    async def navigate(self, url: str):
        await self.page.goto(url)
        return {"navigated": url}

    async def click(self, selector: str):
        await self.page.click(selector)
        return {"clicked": selector}

    async def input_text(self, selector: str, text: str):
        await self.page.fill(selector, text)
        return {"input": {"selector": selector, "value": text}}

    async def submit_form(self, selector: str):
        await self.page.locator(selector).press("Enter")
        return {"submitted": selector}

    async def choose_option(self, selector: str, value: str):
        await self.page.select_option(selector, value)
        return {"selected": {"selector": selector, "value": value}}

    async def snapshot(self, prefix: str):
        """Сохраняет DOM и скриншот"""
        html = await self.page.content()
        path = f"{prefix}.png"
        screenshot = await self.page.screenshot(path=path)
        return {"html": html, "screenshot": path}
