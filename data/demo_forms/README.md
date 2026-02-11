# Демо-формы для demo_mode

Идеальные формы для отладки пайплайна до этапа BPG.

## Требования к изображениям

- **Геометрия:** чёткий прямоугольный контейнер, контрастный фон, выравнивание, отступ между строками, между label и input, input ≥85% ширины контейнера.
- **Текст:** только русский, крупный шрифт; placeholder и текст кнопок выровнены по центру полей/кнопок.

## Генерация (HTML + Playwright)

Рендер через браузер: выравнивание и качество без обрезки.

**Локально (macOS/Homebrew Python):** используйте виртуальное окружение, чтобы не трогать системный Python:

```bash
# из корня проекта
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
python data/demo_forms/generate_demo_forms.py
```

Если venv уже есть и в нём установлены зависимости из `requirements.txt` (в т.ч. playwright), достаточно один раз установить браузер и запускать генератор:

```bash
source .venv/bin/activate
playwright install chromium
python data/demo_forms/generate_demo_forms.py
```

**Важно:** `pipx install playwright` ставит только CLI в отдельном окружении. Этот скрипт должен запускаться из venv проекта, где установлен пакет `playwright` (`pip install playwright` или `pip install -r requirements.txt`).

Создаёт PNG в `data/demo_forms/images/`: `demo_form_01.png` … `demo_form_05.png` (1600×2400 px).

Шаблоны форм: `data/demo_forms/html/form.html` (параметр `?form=01` … `?form=05`).

## Использование

В пайплайне передать `demo_mode=True` и путь к одному из изображений (и при необходимости `form_container_first_debug_dir`). Промежуточные артефакты: `demo_container.json`, `demo_rows.json`, `demo_slots.json`, `demo_slot_assignments.json`, `demo_form_graph.json`, `demo_visualization.png`.
