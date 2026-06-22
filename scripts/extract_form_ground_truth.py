#!/usr/bin/env python3
"""
Извлечение ground truth bbox из form.html для проверки пайплайна.

Использует Playwright: открывает страницу с теми же параметрами viewport,
что и при съёмке скриншотов (1600x2400), собирает getBoundingClientRect()
для всех значимых элементов внутри видимого .form-container и сохраняет в JSON.

Запуск:
  python scripts/extract_form_ground_truth.py
  # Результат: data/demo_forms/ground_truth/demo_form_01.json, ...

Формат JSON:
  - viewport: { width, height }
  - container_bbox: [x1, y1, x2, y2]  (форма)
  - elements: [ { role, bbox, ... }, ... ]
  bbox всегда [x1, y1, x2, y2] в координатах viewport (как на скриншоте).
"""

from __future__ import annotations

import json
import os
import sys

# Viewport как в generate_demo_forms.py
PAGE_W = 1600
PAGE_H = 2400

# Формы для экспорта (как в generate_demo_forms)
FORM_IDS = list(map("%02d".__mod__, list(range(1, 6)) + list(range(10, 19))))


def _collect_bboxes_js() -> str:
    """JS-код для выполнения в странице: возвращает JSON-строку с элементами и bbox."""
    return r"""
    (function() {
      var container = document.querySelector('.form-container:not([hidden])');
      if (!container) return JSON.stringify({ error: 'no visible form container' });
      var r = container.getBoundingClientRect();
      var out = {
        viewport: { width: window.innerWidth, height: window.innerHeight },
        container_bbox: [ r.left, r.top, r.right, r.bottom ],
        elements: []
      };
      function add(el, role, extra) {
        var b = el.getBoundingClientRect();
        if (b.width < 0.1 || b.height < 0.1) return;
        var o = { role: role, bbox: [ b.left, b.top, b.right, b.bottom ] };
        if (extra) for (var k in extra) o[k] = extra[k];
        out.elements.push(o);
      }
      container.querySelectorAll('input, textarea, select, button').forEach(function(el) {
        var role = el.tagName.toLowerCase();
        if (el.type) role = el.type;
        var extra = { tag: el.tagName.toLowerCase() };
        if (el.placeholder) extra.placeholder = el.placeholder;
        if (el.type) extra.type = el.type;
        if (el.name) extra.name = el.name;
        add(el, role, extra);
      });
      container.querySelectorAll('label').forEach(function(el) {
        var text = (el.textContent || '').trim().replace(/\s+/g, ' ');
        add(el, 'label', { text: text });
      });
      container.querySelectorAll('.title, .section-title, .subsection-title').forEach(function(el) {
        var text = (el.textContent || '').trim().replace(/\s+/g, ' ');
        var role = el.classList.contains('title') ? 'title' : (el.classList.contains('section-title') ? 'section_title' : 'subsection_title');
        add(el, role, { text: text });
      });
      return JSON.stringify(out);
    })();
    """


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright не найден. Установите: pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        sys.exit(1)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base, "data", "demo_forms", "html", "form.html")
    out_dir = os.path.join(base, "data", "demo_forms", "ground_truth")

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

            for form_id in FORM_IDS:
                page.goto(url + "?form=" + form_id, wait_until="networkidle")
                raw = page.evaluate(_collect_bboxes_js())
                if isinstance(raw, dict) and raw.get("error"):
                    print("Form %s: %s" % (form_id, raw["error"]), file=sys.stderr)
                    continue
                if isinstance(raw, str):
                    data = json.loads(raw)
                else:
                    data = raw
                out_path = os.path.join(out_dir, "demo_form_%s.json" % form_id)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print("Written %s (%d elements)" % (out_path, len(data.get("elements", []))))

            context.close()
        finally:
            browser.close()

    print("Done. Ground truth in %s" % out_dir)


if __name__ == "__main__":
    main()
