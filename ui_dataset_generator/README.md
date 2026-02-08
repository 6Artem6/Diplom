# Генератор тестового UI-датасета для CV-детекции

Контролируемый набор HTML-страниц с UI-элементами и пайплайн захвата скриншотов + bbox из DOM для обучения/отладки Detectron2 и post-processing.

## Идея

- **Один HTML** (`test-ui.html`) с параметрами `?variant=...` и `?theme=...` рендерит разные конфигурации UI и визуальные темы.
- Структура страницы не меняется; тема задаётся **подключаемым CSS** (папка `themes/`).
- Тип элемента задаётся **только** атрибутами: `data-ui-type` (button, link, input, textarea, checkbox, radio), `data-ui-id` — уникальный id.
- **Playwright** открывает каждую пару (variant, theme), делает скриншот viewport и читает bbox через `getBoundingClientRect()` — **ground truth из DOM**, без OCR и эвристик.
- Результат: PNG + JSON с bbox и полем `theme` (style_variant), пригодный для разнообразного датасета обучения Detectron2, регрессионных тестов и отладки post-processing.

## Структура

```
ui_dataset_generator/
├── test-ui.html          # Одна страница, JS рендерит вариант по ?variant=... и подключает тему по ?theme=...
├── themes/               # Отдельный CSS на каждую тему оформления
│   ├── theme-default.css
│   ├── theme-dark.css
│   ├── theme-high-contrast.css
│   ├── theme-low-contrast.css
│   ├── theme-outline.css
│   ├── theme-ghost.css
│   ├── theme-rounded.css
│   └── theme-square.css
├── capture_screenshots.py # Playwright: скриншоты + bbox для всех variant × theme
├── README.md
└── output/               # По умолчанию: PNG и JSON по каждой паре (variant, theme)
    ├── buttons_small_default.png
    ├── buttons_small_default.json
    ├── buttons_small_dark.png
    ├── buttons_small_dark.json
    ├── forms_dense_outline.png
    ...
```

## Варианты (variants)

Встроенные варианты в `test-ui.html`:

| Variant | Описание |
|---------|----------|
| `buttons_small` | Кнопки XS/S/M, filled/outline/ghost/rounded/disabled |
| `buttons_wide` | Широкие кнопки, длинный текст |
| `buttons_mixed_styles` | Filled, outline, ghost, borderless, rounded, low-contrast |
| `links_standalone` | Ссылки разных размеров, underline / no-underline |
| `links_mixed` | Ссылки XS–L, смешанные стили |
| `inputs_placeholders` | Input с placeholder и без, с label |
| `inputs_dense` | Input XS/S/M/L |
| `inputs_with_labels` | Input с label сверху и др. |
| `textareas_small` / `textareas_large` | Textarea 2–5 строк |
| `checkboxes_radios` | Checkbox и radio группы |
| `forms_dense` | Форма: input + textarea + checkbox + button |
| `full_mixed` | Все типы: navbar, центр, форма |
| `cards_with_controls` | Карточки с кнопками, input, checkbox/radio |

Новый вариант: добавить запись в объект `VARIANTS` в `test-ui.html` и при необходимости запустить захват с `-v variant_name` или без — скрипт подхватит список из `window.UI_DATASET_VARIANTS`.

## Темы оформления (themes)

Один и тот же набор страниц рендерится в разных визуальных стилях; меняется только подключаемый CSS (папка `themes/`).

| Theme | Описание |
|-------|----------|
| `default` | Стандартный Bootstrap 5.x, без доп. переопределений |
| `dark` | Тёмный фон, светлый текст, тёмные карточки и кнопки |
| `high_contrast` | Чёрно-белый, сильные границы |
| `low_contrast` | Серые тона, приглушённый текст и границы |
| `outline` | Кнопки только с обводкой, без заливки |
| `ghost` | Прозрачные кнопки и поля, минимальные границы |
| `rounded` | Увеличенный border-radius у кнопок, полей и карточек |
| `square` | Нулевой border-radius |

URL: `/test-ui.html?variant=buttons_small&theme=dark`, `/test-ui.html?variant=forms_dense&theme=outline`. Список тем задаётся в `window.UI_DATASET_THEMES` в `test-ui.html`. Новую тему: добавить файл `themes/theme-{name}.css` и имя в `UI_DATASET_THEMES`.

## Формат JSON

Для каждой пары (variant, theme) сохраняется файл `{variant}_{theme}.json`:

```json
{
  "image": "buttons_small_dark.png",
  "variant": "buttons_small",
  "theme": "dark",
  "style_variant": "dark",
  "viewport": { "width": 1440, "height": 900 },
  "elements": [
    {
      "id": "btn_1",
      "type": "button",
      "bbox": [x, y, width, height]
    }
  ]
}
```

- **bbox** — `[x, y, width, height]` в пикселях viewport (координаты из `getBoundingClientRect()`).
- **type** — один из: `button`, `link`, `input`, `textarea`, `checkbox`, `radio`.
- **theme** / **style_variant** — идентификатор темы оформления скриншота.

## Требования

- Python 3.10+
- Playwright: `pip install playwright`, затем `playwright install chromium`

## Запуск

### 1. Установка Playwright

```bash
pip install playwright
playwright install chromium
```

### 2. Захват всех пар variant × theme (из директории `ui_dataset_generator/`)

```bash
cd ui_dataset_generator
python capture_screenshots.py
```

Скриншоты и JSON появятся в каталоге `output/` (по умолчанию): для каждой пары (variant, theme) — файлы `{variant}_{theme}.png` и `{variant}_{theme}.json`.

### 3. Свой каталог вывода

```bash
python capture_screenshots.py -o ../data/ui_dataset
```

### 4. Только выбранные варианты и/или темы

```bash
python capture_screenshots.py -v buttons_small forms_dense full_mixed
python capture_screenshots.py -t default dark outline
python capture_screenshots.py -v buttons_small -t dark high_contrast
```

### 5. Страница по URL (например, раздача через HTTP)

```bash
python capture_screenshots.py --url http://localhost:8000/test-ui.html
```

## Использование результата

- **Обучение Detectron2**: PNG как изображения, JSON — разметка bbox и класс (type). Конвертация в COCO/свой формат — отдельным скриптом при необходимости.
- **Регрессионные тесты**: после изменений модели/постобработки заново прогнать пайплайн, сравнить bbox или метрики с эталонным JSON.
- **Отладка post-processing**: скриншот + эталонные bbox из JSON — вход для пайплайна atoms_v2; сверять стабилизированные атомы с ожидаемыми типами и областями.

## Сборка COCO-датасета для Detectron2

Скрипт `build_coco_dataset.py` конвертирует JSON-файлы Playwright (из `output/`) в COCO и раскладывает датасет:

```bash
# Один каталог с JSON и PNG — разбиение 80% train / 20% val
python build_coco_dataset.py --input-dir output -o dataset

# Отдельные каталоги для train и val
python build_coco_dataset.py --train-dir output_train --val-dir output_val -o dataset
```

Результат:
- `dataset/train/images/` — скриншоты с именами `000001.png`, `000002.png`, …
- `dataset/train/train_coco.json` — COCO с уникальными `image_id`, `annotation_id`, `category_id` (type в алфавите), bbox `[x_min, y_min, width, height]`, `area`, `iscrowd=0`
- `dataset/val/images/` и `dataset/val/val_coco.json` — то же для val
- `dataset/dataset.yaml` — `names` (категории) и пути к train/val

В конце выводится: `COCO dataset created for train/val with X images and Y annotations.`

## Ограничения

- Один вариант = один скриншот viewport (1440×900). Полноэкранные/скролл не используются.
- bbox всегда из DOM; OCR не используется.
- Новые UI-типы или стили добавляются в `test-ui.html` (VARIANTS + при необходимости функции создания элементов и классы CSS).
