#!/usr/bin/env python3
"""
Генерация скриншотов тестовых CRM/ERP HTML-страниц и метаданных для input_candidate пайплайна.

Цель: набор страниц с формами и input-полями для проверки OCR-seeds и fallback visual seeds.
Для каждой страницы: скриншот (≥1200px ширина) + JSON с координатами и типами input по формам.

Зависимость: pip install playwright && playwright install chromium

Использование:
  python scripts/crm_forms_screenshots.py [--html-dir DIR] [--out-dir DIR] [--width W] [--dry-run]
  python scripts/crm_forms_screenshots.py [--themes default,borders,dark,green]  # вариативность по темам
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_HTML_DIR = "data/crm_forms/html"
DEFAULT_OUT_DIR = "data/crm_forms"
MIN_VIEWPORT_WIDTH = 1200
DEFAULT_VIEWPORT = {"width": 1280, "height": 800}
WAIT_AFTER_LOAD_MS = 500
AVAILABLE_THEMES = ("default", "borders", "dark", "green")


def collect_input_metadata_js() -> str:
    """JS для сбора форм и input-полей с bbox и типами (выполняется в браузере)."""
    return """
    (() => {
      const forms = [];
      const formSections = document.querySelectorAll('[data-form], .card, .panel');
      formSections.forEach((section, idx) => {
        const formId = section.getAttribute('data-form') || section.className || 'form_' + idx;
        const inputs = [];
        section.querySelectorAll('input:not([type="hidden"]):not([type="submit"]):not([type="button"]), textarea')
          .forEach(el => {
            const r = el.getBoundingClientRect();
            if (r.width < 1 || r.height < 1) return;
            const bbox = [r.left + window.scrollX, r.top + window.scrollY, r.left + window.scrollX + r.width, r.top + window.scrollY + r.height];
            inputs.push({
              type: el.type || (el.tagName.toLowerCase() === 'textarea' ? 'textarea' : 'text'),
              tagName: el.tagName.toLowerCase(),
              bbox: bbox,
              placeholder: el.placeholder || null,
              name: el.name || null,
              id: el.id || null
            });
          });
        if (inputs.length > 0) {
          forms.push({ id: formId, inputs });
        }
      });
      return { forms, viewport: { width: window.innerWidth, height: window.innerHeight } };
    })();
    """


def capture_page(
    browser: Any,
    html_path: Path,
    screenshot_dir: Path,
    metadata_dir: Path,
    viewport: Dict[str, int],
    theme: Optional[str] = None,
    timeout_ms: int = 15000,
) -> bool:
    """Открывает HTML-страницу (опционально ?theme=X), делает скриншот и сохраняет метаданные форм/input."""
    file_url = html_path.as_uri()
    if theme:
        file_url = file_url + ("&" if "?" in file_url else "?") + "theme=" + theme
    stem = html_path.stem
    suffix = "_" + theme if theme else ""
    screenshot_path = screenshot_dir / f"{stem}{suffix}.png"
    metadata_path = metadata_dir / f"{stem}{suffix}.json"

    ctx = browser.new_context(viewport=viewport)
    page = ctx.new_page()
    page.set_default_timeout(timeout_ms)
    try:
        page.goto(file_url, wait_until="domcontentloaded")
        page.wait_for_timeout(WAIT_AFTER_LOAD_MS)
        page.screenshot(path=str(screenshot_path))
        meta = page.evaluate(collect_input_metadata_js())
        meta["page"] = html_path.name
        meta["screenshot"] = screenshot_path.name
        meta["viewport_capture"] = viewport
        meta["theme"] = theme
        metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("  %s theme=%s -> %s (forms: %d, inputs: %d)",
                    html_path.name, theme or "default", screenshot_path.name,
                    len(meta.get("forms", [])), sum(len(f["inputs"]) for f in meta.get("forms", [])))
        return True
    except Exception as e:
        logger.warning("  %s theme=%s failed: %s", html_path.name, theme or "default", e)
        return False
    finally:
        ctx.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Screenshot CRM/ERP test pages and collect input metadata")
    parser.add_argument("--html-dir", type=str, default=DEFAULT_HTML_DIR, help="Directory with HTML files")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output base directory")
    parser.add_argument("--width", type=int, default=DEFAULT_VIEWPORT["width"], help="Viewport width (≥%d)" % MIN_VIEWPORT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_VIEWPORT["height"], help="Viewport height")
    parser.add_argument("--theme", type=str, default=None, help="Single theme for all pages (default, borders, dark, green)")
    parser.add_argument("--themes", type=str, default=None, help="Comma-separated themes; each page shot once per theme")
    parser.add_argument("--dry-run", action="store_true", help="List HTML files only")
    args = parser.parse_args()

    base = Path.cwd()
    html_dir = base / args.html_dir
    out_dir = base / args.out_dir
    screenshot_dir = out_dir / "screenshots"
    metadata_dir = out_dir / "metadata"

    if not html_dir.is_dir():
        logger.error("HTML dir not found: %s", html_dir)
        return 1

    html_files = sorted(html_dir.glob("*.html"))
    if not html_files:
        logger.warning("No .html files in %s", html_dir)
        return 0

    width = max(args.width, MIN_VIEWPORT_WIDTH)
    viewport = {"width": width, "height": args.height}

    themes: List[Optional[str]] = [None]
    if args.themes:
        raw = [t.strip() for t in args.themes.split(",") if t.strip()]
        themes = [None if t.lower() == "default" else t for t in raw] if raw else [None]
    elif args.theme:
        t = args.theme.strip()
        themes = [None] if t.lower() == "default" else [t]

    if args.dry_run:
        for f in html_files:
            for th in themes:
                logger.info("Would process: %s theme=%s", f.name, th or "default")
        logger.info("Total: %d files × %d themes, viewport %dx%d", len(html_files), len(themes), viewport["width"], viewport["height"])
        return 0

    screenshot_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright not installed: pip install playwright && playwright install chromium")
        return 1

    ok = 0
    total = len(html_files) * len(themes)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for html_path in html_files:
            for theme in themes:
                logger.info("Processing %s theme=%s", html_path.name, theme or "default")
                if capture_page(browser, html_path, screenshot_dir, metadata_dir, viewport, theme=theme):
                    ok += 1
        browser.close()

    logger.info("Done: %d/%d screenshots and metadata in %s", ok, total, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
