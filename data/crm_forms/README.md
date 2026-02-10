# CRM/ERP тестовые страницы для input_candidate пайплайна

Набор HTML-страниц, имитирующих CRM/ERP интерфейсы с формами и input-полями, для проверки пайплайна поиска input_candidate (OCR-seeds и fallback visual seeds).

## Структура

- **html/** — исходные HTML-страницы (4 страницы).
- **screenshots/** — скриншоты (генерируются скриптом, ≥1200px ширина).
- **metadata/** — JSON с метаданными: для каждой формы список input-полей с координатами (bbox) и типами.

## Страницы

| Файл | Описание |
|------|----------|
| crm_01_vertical_form.html | Вертикальная форма: label слева, поля с разным фоном, textarea. |
| crm_02_grid_cards.html | Сетка карточек: 2–3 формы в карточках, разные бордеры. |
| crm_03_mixed_layouts.html | Sidebar + контент: горизонтальные ряды (label + input), textarea. |
| crm_04_admin_panel.html | Admin-панель: sidebar, несколько панелей с полями. |
| crm_05_borders_theme.html | Форма с явными границами (2px border) для теста Canny/контуров. |

Все страницы поддерживают темы через `?theme=borders|dark|green`. В каждой теме **три уровня контраста** для распознавания:
- **Фон страницы** — один цвет;
- **Форма (card/panel)** — другой оттенок, форма визуально отделена;
- **Поля ввода** — белые на светлой теме, тёмный оттенок на тёмной, того же тона что форма, но светлее/темнее, чтобы элементы формы отличались от контейнера.

Во всех страницах: header с кнопками Save/Cancel, input с min-width 150px, min-height 24px, max-height 80px, контраст бордера/фона для Canny.

## Генерация скриншотов и метаданных

Зависимость: `pip install playwright && playwright install chromium`

```bash
# из корня проекта
python3 scripts/crm_forms_screenshots.py

# вариативность по темам (default, borders, dark, green)
python3 scripts/crm_forms_screenshots.py --themes default,borders,dark,green

# одна тема для всех страниц
python3 scripts/crm_forms_screenshots.py --theme borders

# опции
python3 scripts/crm_forms_screenshots.py --html-dir data/crm_forms/html --out-dir data/crm_forms --width 1280 --height 800
python3 scripts/crm_forms_screenshots.py --dry-run   # только список файлов
```

Результат:

- `data/crm_forms/screenshots/<stem>.png` — скриншот каждой страницы.
- `data/crm_forms/metadata/<stem>.json` — метаданные: `forms[]` с полями `id`, `inputs[]` (type, bbox [x1,y1,x2,y2], placeholder, name, id).

## Формат метаданных

```json
{
  "page": "crm_01_vertical_form.html",
  "screenshot": "crm_01_vertical_form.png",
  "viewport_capture": { "width": 1280, "height": 800 },
  "forms": [
    {
      "id": "lead",
      "inputs": [
        { "type": "text", "tagName": "input", "bbox": [x1, y1, x2, y2], "placeholder": "Введите имя", "id": "f1-name" }
      ]
    }
  ]
}
```

## Цель для тестов

- Обеспечить наличие хотя бы одного input на странице, который пайплайн Phase A/B найдёт через OCR-seed или fallback visual seed.
- Разные цвета и контраст для стабильной работы Canny и visual fallback.
- Подготовить набор для отладки и обучения input_candidate_recovery.
