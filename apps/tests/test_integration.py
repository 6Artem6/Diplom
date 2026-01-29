import aiohttp
import pytest


@pytest.mark.asyncio
async def test_integration_graph_from_template():
    async with aiohttp.ClientSession() as session:
        # 1. Берём HTML шаблона login.html.j2 через sitegpt
        async with session.get(
            "http://sitegpt_app:8000/graph_from_url",
            params={"url": "http://template:8080/login"},
        ) as resp:
            data = await resp.json()
            assert "graph" in data
            assert "url" in data
            assert data["url"].endswith("login")

        # 2. Проверим vision_analyzer на тестовом изображении
        with open("/tests/data/sample_page.png", "rb") as f:
            form = aiohttp.FormData()
            form.add_field(
                "file", f, filename="sample_page.png", content_type="image/png"
            )
            async with session.post(
                "http://vision_analyzer:5001/analyze", data=form
            ) as resp:
                det = await resp.json()
                assert "detections" in det
                assert isinstance(det["detections"], list)
