#!/usr/bin/env python3
"""
Генератор демо-форм для demo_mode пайплайна.

Рендер через HTML + Playwright: идеальное выравнивание текста и placeholder
по центру, чёткие границы полей, высокое разрешение без обрезки.
"""

from __future__ import annotations

import os
import sys

# Размер viewport (пропорции как 800x1200, увеличены для качества)
PAGE_W = 1600
PAGE_H = 2400


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright не найден в текущем окружении (в котором вы запускаете этот скрипт).\n"
            "Установите библиотеку именно в нём:\n"
            "  pip install playwright\n"
            "  playwright install chromium\n"
            "Примечание: pipx ставит только CLI; для generate_demo_forms.py нужен pip install в venv проекта.",
            file=sys.stderr,
        )
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "html", "form.html")
    out_dir = os.path.join(base_dir, "images")

    if not os.path.isfile(html_path):
        print("Missing %s" % html_path, file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    url = "file://" + html_path

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context(
                viewport={"width": PAGE_W, "height": PAGE_H},
                device_scale_factor=1,
            )
            page = context.new_page()

            for i in range(1, 6):
                form_id = "%02d" % i
                page.goto(url + "?form=" + form_id, wait_until="networkidle")
                out_path = os.path.join(out_dir, "demo_form_%s.png" % form_id)
                page.screenshot(path=out_path)
                print("Written %s" % out_path)

            context.close()
        finally:
            browser.close()

    print("Done. Demo forms in %s" % out_dir)


if __name__ == "__main__":
    main()
