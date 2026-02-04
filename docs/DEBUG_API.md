# Debug API: layout / text detection / OCR

Тестирование пайплайна через HTTP, без переключения переменных в коде.

**Base URL:** `POST /api/v1/debug/...`

**Debug-изображения:** сохраняются в `DEBUG_OUTPUT_DIR` (по умолчанию `debug/output/`). Задать свой каталог: `export DEBUG_OUTPUT_DIR=/path/to/output`.

---

## Ручки

### `POST /api/v1/debug/layout`

**Вход:** изображение (multipart `image`).

**Выход:**
- `regions`: список `{x, y, w, h, type}` (`text_region` | `ui_region` | `background`);
- `debug_image_path`: путь к сохранённому изображению с нарисованными регионами.

**Без OCR.** Логи: `debug/layout: filename=... regions=N (text=... ui=... bg=...)`.

---

### `POST /api/v1/debug/text-detect`

**Вход:** изображение (multipart `image`).

**Выход:**
- `boxes`: список `{x, y, w, h}` (текстовые bbox);
- `debug_image_path`: путь к изображению с bbox.

**Без layout и OCR.** Используется PaddleOCR, если установлен; иначе возвращается пустой список и в лог пишется, что детектор недоступен.

---

### `POST /api/v1/debug/ocr`

**Вход:** multipart: `image` (файл) и `boxes_json` (строка JSON-массива bbox), например:
`boxes_json=[{"x":10,"y":20,"w":100,"h":24}]`.

**Выход:**
- `results`: список `{text, confidence}` по одному на каждый bbox.

**Без layout и text detection.** Логи: `debug/ocr: filename=... boxes_count=N`.

---

### `POST /api/v1/debug/full-pipeline`

**Вход:**
- изображение (multipart `image`);
- опционально: **`use_dl_only`** (form, bool, по умолчанию `false`). При `true`: layout только из DL (LayoutParser PubLayNet), **без CV second pass**; Paddle text detection **по всему изображению**; привязка text_boxes к регионам по overlap. Типы регионов: card, section, text_region (нет button/badge из контуров).

**Выход:**
- `regions`, `raw_paddle_text_boxes`, `text_boxes`, `gui_blocks`;
- `dropped_count`, `drop_reasons`, `text_box_drop_reasons`;
- `debug_image_path`: путь к debug-изображению.

**Полный пайплайн:** layout → text detect → OCR → сборка блоков. При `use_dl_only=false`: layout из DL+CV или CV, text detect по ROI. При `use_dl_only=true`: layout только из DL, text по всему кадру. Статус пайплайна и ограничения — см. **docs/PIPELINE_STATUS.md**.

---

## Логи и отладка

- Входные параметры (имя файла, число bbox и т.д.) логируются при каждом запросе.
- Количество регионов, текстовых боксов и отброшенных блоков логируется в соответствующих сервисах.
- При отбрасывании блока причина добавляется в `drop_reasons` и при необходимости в лог.
