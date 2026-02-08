#!/usr/bin/env python3
"""
Пайплайн захвата скриншотов и bbox для датасета UI-детекции.

Использует Playwright:
1. Открывает test-ui.html?variant=<name>&theme=<theme> для каждой пары (variant, theme).
2. Делает скриншот viewport (1440×900).
3. Извлекает bbox из DOM (getBoundingClientRect) для всех [data-ui-type].
4. Сохраняет PNG и JSON в output_dir. Имя файла: {variant}_{theme}.png / .json.
5. JSON включает variant, theme (style_variant) и elements.

bbox — ground truth из DOM, без OCR и эвристик.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Viewport для воспроизводимости (desktop)
VIEWPORT = {"width": 1440, "height": 900}
OUTPUT_DIR_DEFAULT = "output"
HTML_FILE = "test-ui.html"


def sanitize_filename(name: str) -> str:
    """Имя варианта → безопасное имя файла."""
    return re.sub(r"[^\w\-]", "_", name).strip("_") or "unnamed"


def get_variants_from_page(page) -> list[str]:
    """Читает список вариантов из window.UI_DATASET_VARIANTS после загрузки страницы."""
    try:
        return page.evaluate("() => window.UI_DATASET_VARIANTS || []")
    except Exception:
        return []


def get_themes_from_page(page) -> list[str]:
    """Читает список тем из window.UI_DATASET_THEMES после загрузки страницы."""
    try:
        return page.evaluate("() => window.UI_DATASET_THEMES || []")
    except Exception:
        return []


def extract_bbox_from_dom(page) -> list[dict]:
    """
    Находит все элементы с data-ui-type, читает data-ui-id, data-ui-type и bbox из DOM.
    bbox = [x, y, width, height] в пикселях viewport (getBoundingClientRect).
    """
    script = """
    () => {
        const nodes = document.querySelectorAll('[data-ui-type][data-ui-id]');
        return Array.from(nodes).map(el => {
            const r = el.getBoundingClientRect();
            return {
                id: el.getAttribute('data-ui-id'),
                type: el.getAttribute('data-ui-type'),
                bbox: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]
            };
        });
    }
    """
    return page.evaluate(script)


def _url_append_param(url: str, param: str, value: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{param}={value}"


def capture_variant_theme(
    page,
    url: str,
    variant: str,
    theme: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Открывает url?variant=...&theme=..., делает скриншот, извлекает bbox, сохраняет PNG и JSON.
    Имя файла: {variant}_{theme}.png / .json. JSON включает theme и style_variant.
    Возвращает (path_png, path_json).
    """
    full_url = _url_append_param(_url_append_param(url, "variant", variant), "theme", theme)
    page.goto(full_url, wait_until="networkidle", timeout=15000)
    page.wait_for_selector("[data-ui-type]", state="attached", timeout=5000)

    safe_variant = sanitize_filename(variant)
    safe_theme = sanitize_filename(theme)
    file_base = f"{safe_variant}_{safe_theme}"
    path_png = output_dir / f"{file_base}.png"
    path_json = output_dir / f"{file_base}.json"

    page.screenshot(path=str(path_png))
    elements = extract_bbox_from_dom(page)

    payload = {
        "image": path_png.name,
        "variant": variant,
        "theme": theme,
        "style_variant": theme,
        "viewport": VIEWPORT,
        "elements": elements,
    }
    path_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Captured variant=%s theme=%s: %d elements -> %s, %s",
        variant,
        theme,
        len(elements),
        path_png.name,
        path_json.name,
    )
    return path_png, path_json


DEFAULT_THEMES = [
    "default",
    "dark",
    "high_contrast",
    "low_contrast",
    "outline",
    "ghost",
    "rounded",
    "square",
]


def main(
    output_dir: str | Path = OUTPUT_DIR_DEFAULT,
    variants_override: list[str] | None = None,
    themes_override: list[str] | None = None,
    base_url: str | None = None,
) -> int:
    """
    Запускает захват для всех пар (variant, theme).
    base_url: URL страницы test-ui.html (если None — file:// из текущей директории).
    variants_override / themes_override: если заданы, использовать эти списки; иначе читать из страницы.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return 1

    root = Path(__file__).resolve().parent
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if base_url is None:
        html_path = root / HTML_FILE
        if not html_path.exists():
            logger.error("Not found: %s", html_path)
            return 1
        base_url = html_path.as_uri()

    default_variants = [
        "buttons_small",
        "buttons_wide",
        "buttons_mixed_styles",
        "links_standalone",
        "links_mixed",
        "inputs_placeholders",
        "inputs_dense",
        "inputs_with_labels",
        "textareas_small",
        "textareas_large",
        "checkboxes_radios",
        "forms_dense",
        "full_mixed",
        "cards_with_controls",
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()

        try:
            page.goto(base_url, wait_until="networkidle", timeout=15000)
            variants = variants_override or get_variants_from_page(page)
            if not variants:
                logger.warning("No variants from page; using default list")
                variants = default_variants
            themes = themes_override or get_themes_from_page(page)
            if not themes:
                logger.warning("No themes from page; using default list")
                themes = DEFAULT_THEMES

            for v in variants:
                for t in themes:
                    capture_variant_theme(page, base_url, v, t, out)
        finally:
            context.close()
            browser.close()

    logger.info("Done. Output: %s", out)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Capture UI dataset screenshots and bbox from test-ui.html (variant × theme)"
    )
    parser.add_argument("-o", "--output", default=OUTPUT_DIR_DEFAULT, help="Output directory for PNG and JSON")
    parser.add_argument("-v", "--variants", nargs="*", help="Variant names (default: from page)")
    parser.add_argument("-t", "--themes", nargs="*", help="Theme names (default: from page)")
    parser.add_argument("--url", default=None, help="Base URL of test-ui.html (default: file://)")
    args = parser.parse_args()
    sys.exit(
        main(
            output_dir=args.output,
            variants_override=args.variants or None,
            themes_override=args.themes or None,
            base_url=args.url,
        )
    )
