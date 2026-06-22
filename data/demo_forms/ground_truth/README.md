# Ground truth bbox для проверки пайплайна

Артефакты получаются скриптом:

```bash
python scripts/extract_form_ground_truth.py
```

Требуется Playwright (`pip install playwright && playwright install chromium`).

Скрипт открывает `data/demo_forms/html/form.html` с теми же параметрами viewport (1600×2400), что и при съёмке скриншотов, и для каждой формы (`?form=01`, …) сохраняет в `demo_form_XX.json`:

- **viewport** — размер окна
- **container_bbox** — [x1, y1, x2, y2] контейнера формы
- **elements** — массив элементов с полями:
  - **role** — `input`, `textarea`, `select`, `button`, `label`, `title`, `section_title`, `subsection_title`
  - **bbox** — [x1, y1, x2, y2] в координатах viewport (как на PNG)
  - при необходимости: **text**, **placeholder**, **type**, **name**

Координаты совпадают со скриншотами из `data/demo_forms/images/`, их можно сравнивать с результатами OCR и CV пайплайна.

## Сравнение с пайплайном (в Docker)

1. Прогнать пайплайн по всем демо-формам с записью CV-bbox в лог:

```bash
docker exec -it bpg_construction_service bash -c 'export DEBUG_BBOX_LOG=/app/debug/debug.log; for img in data/demo_forms/images/*.png; do python -m src.infrastructure.atoms_v2.experimental_v2.run_state_machine_pipeline "$img" --output ./debug/forms/$(basename "$img"); done'
```

2. Запустить сравнение по логу и ground truth:

```bash
docker exec -it bpg_construction_service python scripts/compare_ground_truth.py /app/debug/debug.log /app/data/demo_forms/ground_truth
```

Скрипт выведет по каждому изображению: число совпадений по bbox (IoU≥0.2), ошибки типа (ожидался input/action/label — получен другой тип), пропущенные GT-элементы и лишние CV-элементы. В конце — сводные счётчики.

Лог на хосте: `./debug/debug.log` (в контейнере: `/app/debug/debug.log`). Формат NDJSON: **source** `ocr`/`cv`, **image_path**, **data.bboxes**.
