#!/usr/bin/env python3
"""
Оффлайн-скрипт генерации датасета скриншотов UI через Playwright.

Цель: разнообразные состояния UI для прогона UI-пайплайна и накопления датасета CatBoost (button vs input).
Не интегрируется в основной пайплайн. Вход: YAML/JSON конфиг (urls, viewports, zoom, themes, languages, form_states, button_states).
Для каждой комбинации: открыть страницу, применить состояние, дождаться стабильного layout, screenshot.
Сохраняет: PNG + metadata.json рядом. Имена файлов — детерминированный хеш от параметров.

Зависимость: pip install playwright pyyaml && playwright install chromium
(YAML опционален; для .json конфига достаточно stdlib.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    """Загружает YAML или JSON конфиг."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(raw) or {}
        except ImportError:
            raise ImportError("PyYAML required for YAML config: pip install pyyaml")
    if p.suffix.lower() == ".json":
        return json.loads(raw)
    raise ValueError("Config must be .yaml, .yml or .json")


def params_hash(params: Dict[str, Any]) -> str:
    """Детерминированный короткий хеш от параметров (для имени файла)."""
    canonical = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _resolve_generator_url(config: Dict[str, Any]) -> str:
    """Возвращает file:// URL для локальной Bootstrap-страницы генератора."""
    html_path = config.get("generator_html", "ui_dataset_generator/catboost_ui.html")
    p = Path(html_path)
    if not p.is_absolute():
        # Относительно корня проекта (откуда запускают скрипт или CWD)
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(f"Generator HTML not found: {p}")
    return p.as_uri()


def _get_generator_axes_from_page(browser: Any, file_url: str, timeout_ms: int) -> Dict[str, List[str]]:
    """Открывает генератор один раз и читает UI_DATASET_* из window."""
    ctx = browser.new_context()
    page = ctx.new_page()
    page.set_default_timeout(timeout_ms)
    try:
        page.goto(file_url + "?variant=buttons_small&theme=default", wait_until="domcontentloaded")
        page.wait_for_selector("[data-ui-type]", timeout=timeout_ms)
        axes: Dict[str, List[str]] = {}
        for key, js in [
            ("variants", "window.UI_DATASET_VARIANTS || []"),
            ("themes", "window.UI_DATASET_THEMES || []"),
            ("layouts", "window.UI_DATASET_LAYOUTS || ['normal']"),
            ("form_filled", "window.UI_DATASET_FORM_FILLED || ['0','1']"),
            ("button_hover", "window.UI_DATASET_BUTTON_HOVER || ['0','1']"),
        ]:
            val = page.evaluate(js)
            axes[key] = list(val) if val else (["normal"] if key == "layouts" else [])
        return axes
    finally:
        ctx.close()


def iter_combinations(config: Dict[str, Any], generator_axes: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
    """Генерирует комбинации.
    Режим генератора (use_generator): variant × theme × layout × viewport × zoom × form_filled × button_hover.
    Иначе: url × viewport × zoom × theme × language × form_state × button_state.
    """
    use_generator = config.get("use_generator", False)
    viewports = config.get("viewports") or [{"width": 1280, "height": 720}]
    if isinstance(viewports, dict):
        viewports = [viewports]
    zooms = config.get("zoom_levels") or [1.0]

    if use_generator:
        base_url = _resolve_generator_url(config)
        variants = config.get("variants") or (generator_axes or {}).get("variants") or ["buttons_small"]
        themes = config.get("themes") or (generator_axes or {}).get("themes") or ["default"]
        layouts = config.get("layouts") or (generator_axes or {}).get("layouts") or ["normal"]
        form_filled = config.get("form_filled") or (generator_axes or {}).get("form_filled") or ["0", "1"]
        button_hover = config.get("button_hover") or (generator_axes or {}).get("button_hover") or ["0", "1"]

        out: List[Dict[str, Any]] = []
        for variant in variants:
            for theme in themes:
                for layout in layouts:
                    for vp in viewports:
                        for zoom in zooms:
                            for ff in form_filled:
                                for bh in button_hover:
                                    q = f"variant={variant}&theme={theme}&layout={layout}&form_filled={ff}&button_hover={bh}"
                                    url = f"{base_url}?{q}"
                                    out.append({
                                        "url": url,
                                        "viewport": {"width": int(vp.get("width", 1280)), "height": int(vp.get("height", 720))},
                                        "zoom": float(zoom),
                                        "theme": theme,
                                        "language": "en-US,en;q=0.9",
                                        "form_state": "filled" if ff == "1" else "empty",
                                        "button_state": "hover" if bh == "1" else "normal",
                                        "generator": True,
                                        "variant": variant,
                                        "layout": layout,
                                        "form_filled": ff,
                                        "button_hover": bh,
                                    })
        return out

    urls = config.get("urls") or []
    themes = config.get("themes") or ["light"]
    languages = config.get("languages") or ["en-US,en;q=0.9"]
    form_states = config.get("form_states") or ["empty"]
    button_states = config.get("button_states") or ["normal"]
    out = []
    for url in urls:
        for vp in viewports:
            for zoom in zooms:
                for theme in themes:
                    for lang in languages:
                        for form in form_states:
                            for btn in button_states:
                                out.append({
                                    "url": url,
                                    "viewport": {"width": int(vp.get("width", 1280)), "height": int(vp.get("height", 720))},
                                    "zoom": float(zoom),
                                    "theme": theme,
                                    "language": lang,
                                    "form_state": form,
                                    "button_state": btn,
                                })
    return out


def apply_form_state(page: Any, form_state: str, timeout_ms: int) -> None:
    """Применяет состояние формы: empty / filled / error (generic, без привязки к сайту)."""
    if form_state == "empty":
        return
    if form_state == "filled":
        try:
            page.evaluate("""
                () => {
                    const inputs = document.querySelectorAll('input:not([type=hidden]):not([type=submit]):not([type=button]), textarea');
                    inputs.forEach((el, i) => {
                        if (el.type === 'email') el.value = 'test@example.com';
                        else if (el.type === 'password') el.value = 'password123';
                        else if (el.placeholder) el.value = el.placeholder;
                        else el.value = 'test' + (i + 1);
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                    });
                }
            """)
            time.sleep(0.3)
        except Exception as e:
            logger.debug("apply_form_state filled: %s", e)
    if form_state == "error":
        try:
            # Клик по первой кнопке сабмита без заполнения — часто даёт ошибки валидации
            page.evaluate("""
                () => {
                    const btn = document.querySelector('button[type=submit], input[type=submit], [type=submit]');
                    if (btn) btn.click();
                }
            """)
            time.sleep(1.0)
        except Exception as e:
            logger.debug("apply_form_state error: %s", e)


def apply_button_state(page: Any, button_state: str) -> None:
    """Применяет состояние кнопки: normal / hover / disabled."""
    if button_state == "normal":
        return
    if button_state == "hover":
        try:
            page.evaluate("""
                () => {
                    const btn = document.querySelector('button, input[type=submit], input[type=button], [role=button], a.btn');
                    if (btn) btn.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
                }
            """)
            time.sleep(0.2)
        except Exception as e:
            logger.debug("apply_button_state hover: %s", e)
    # disabled — без site-specific логики не задаётся универсально


def wait_stable(page: Any, wait_ms: int) -> None:
    """Ждёт стабильного layout (простая пауза + ожидание networkidle при возможности)."""
    time.sleep(wait_ms / 1000.0)
    try:
        page.wait_for_load_state("networkidle", timeout=min(3000, wait_ms))
    except Exception:
        pass


def capture_one(
    browser: Any,
    params: Dict[str, Any],
    output_dir: Path,
    timeout_ms: int,
    wait_after_load_ms: int,
    stability_wait_ms: int,
) -> Optional[Path]:
    """Открывает страницу с заданными параметрами, применяет состояние, делает скриншот. Возвращает путь к PNG или None."""
    url = params["url"]
    vp = params["viewport"]
    zoom = params["zoom"]
    theme = params["theme"]
    lang = params["language"]
    form_state = params["form_state"]
    button_state = params["button_state"]

    try:
        ctx_opts: Dict[str, Any] = {
            "viewport": {"width": vp["width"], "height": vp["height"]},
            "locale": (lang.split(",")[0].strip() if lang else "en-US"),
            "color_scheme": "dark" if theme == "dark" else "light",
        }
        if lang:
            ctx_opts["extra_http_headers"] = {"Accept-Language": lang}
        context = browser.new_context(**ctx_opts)
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_load_state("load", timeout=timeout_ms)
        time.sleep(wait_after_load_ms / 1000.0)
        if zoom != 1.0:
            page.evaluate(f"() => document.body.style.zoom = {zoom}")
        time.sleep(0.2)

        if not params.get("generator"):
            apply_form_state(page, form_state, timeout_ms)
            apply_button_state(page, button_state)
        wait_stable(page, stability_wait_ms)

        name = params_hash(params)
        out_png = output_dir / f"{name}.png"
        out_meta = output_dir / f"{name}.metadata.json"
        page.screenshot(path=str(out_png), full_page=False)
        if params.get("generator"):
            try:
                elements = page.evaluate("""() => {
                    var els = document.querySelectorAll('[data-ui-type]');
                    return Array.from(els).map(function(el) {
                        var r = el.getBoundingClientRect();
                        var id = el.getAttribute('data-ui-id') || el.id || ('el_' + Math.random().toString(36).slice(2));
                        var type = el.getAttribute('data-ui-type') || '';
                        return { id: id, type: type, bbox: [r.left, r.top, r.left + r.width, r.top + r.height] };
                    }).filter(function(x) { return x.type; });
                }""")
                out_elements = output_dir / f"{name}.elements.json"
                out_elements.write_text(json.dumps(elements, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                logger.debug("capture elements (generator): %s", e)
        metadata = {
            "url": url,
            "viewport": vp,
            "zoom": zoom,
            "theme": theme,
            "language": lang,
            "form_state": form_state,
            "button_state": button_state,
            "filename": out_png.name,
        }
        if params.get("generator"):
            metadata["variant"] = params.get("variant")
            metadata["layout"] = params.get("layout")
            metadata["form_filled"] = params.get("form_filled")
            metadata["button_hover"] = params.get("button_hover")
        out_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        context.close()
        return out_png
    except Exception as e:
        logger.warning("capture %s: %s", params_hash(params), e)
        return None


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python generate_ui_screenshots.py <config.yaml|config.json> [--dry-run]", file=sys.stderr)
        return 1
    config_path = sys.argv[1]
    dry_run = "--dry-run" in sys.argv
    config = load_config(config_path)
    output_dir = Path(config.get("output_dir", "datasets/ui_screenshots"))
    timeout_ms = int(config.get("timeout_ms", 15000))
    wait_after_load_ms = int(config.get("wait_after_load_ms", 800))
    stability_wait_ms = int(config.get("stability_wait_ms", 500))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright not installed: pip install playwright && playwright install chromium")
        return 1

    use_generator = config.get("use_generator", False)
    if use_generator and not (config.get("variants") and config.get("themes")):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                base_url = _resolve_generator_url(config)
                generator_axes = _get_generator_axes_from_page(browser, base_url.split("?")[0], timeout_ms)
                combinations = iter_combinations(config, generator_axes)
            finally:
                browser.close()
    else:
        combinations = iter_combinations(config)

    logger.info("Total combinations: %s", len(combinations))
    if dry_run:
        for i, c in enumerate(combinations[:5]):
            logger.info("  [%s] %s", i, params_hash(c))
        if len(combinations) > 5:
            logger.info("  ... and %s more", len(combinations) - 5)
        return 0

    ok = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for i, params in enumerate(combinations):
            url_preview = (params["url"][:60] + "…") if len(params.get("url", "")) > 60 else params.get("url", "")
            logger.info("[%s/%s] %s %s", i + 1, len(combinations), url_preview, params_hash(params))
            result = capture_one(browser, params, output_dir, timeout_ms, wait_after_load_ms, stability_wait_ms)
            if result:
                ok += 1
        browser.close()
    logger.info("Done: %s/%s screenshots saved to %s", ok, len(combinations), output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
