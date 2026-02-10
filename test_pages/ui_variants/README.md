# UI Variants — тестовые страницы для детекции input/textarea/button

Набор HTML-страниц в стиле CRM/ERP для проверки пайплайна детекции UI-элементов **без опоры на контуры**: основной источник истины — **layout + OCR + повторяемые паттерны формы**.

**Требования:** только HTML + CSS, без JavaScript.

**Философия:** «Сначала структура, потом пиксели. Всё остальное — шум.»

---

## Архитектура пайплайна

- **PRIMARY:** Layout + OCR → **CardFieldLayoutInference** (восстановление input по структуре card/form_region).
- **FALLBACK:** InputCandidateRecovery (Canny/контуры) только если layout ничего не дал.

---

## Файлы и что проверяется

| Файл | Что проверяется |
|------|-----------------|
| `01_single_field_light.html` | Одиночное поле, светлая тема, белый input 1px. ≥1 input в card. |
| `02_single_field_dark.html` | Одиночное поле, тёмная тема. Слабая зависимость от темы. |
| `03_grid_label_top.html` | Сетка, label сверху. Повторяемость и выравнивание. |
| `04_grid_label_left.html` | Сетка, label слева. Границы без захвата label. |
| `05_same_style_fields.html` | Одинаковые поля в одной карточке. Нормализация ширины, анти-дубли. |
| `06_input_variants_borders.html` | Белый, серый, синий; 1px, 2px, без рамки. Влияние стиля минимально. |
| `07_input_textarea_button.html` | Input, textarea, button, outline-button. Textarea ≠ input, кнопки не в input. |
| `08_light_colored_inputs.html` | Светлая тема: серый, синий, зелёный input. Цветные поля. |
| `09_dark_colored_inputs.html` | Тёмная тема: цветные input. Стабильность по теме. |
| `10_borderless_placeholder_only.html` | Поля без рамки, только placeholder. Input без border по layout. |
| `11_outline_buttons_next_to_fields.html` | Outline-кнопки рядом с полями. Не путать кнопки с input. |
| `12_grid_2col.html` | Сетка 2 колонки. Row detection и несколько полей в строке. |
| `13_grid_3col.html` | Сетка 3 колонки. Grid layout. |
| `14_colored_backgrounds.html` | Карточка с цветным фоном (не белый). Контраст форма/поле. |

---

## Темы и стили

- **Светлая:** светлый фон страницы и карточки, поля контрастируют.
- **Тёмная:** тёмный фон и карточка, поля с тёмным/цветным фоном.
- **Цвета input:** белый, серый, синий, зелёный, цветные фоны.
- **Рамки:** 1px, 2px, borderless.

---

## Использование

1. Открыть файл в браузере или отдать статику через HTTP-сервер.
2. Сделать скриншот.
3. Прогнать через пайплайн (FormStructureDetection → **CardFieldLayoutInference** → при 0 полей: InputCandidateRecovery).
4. Проверить логи:
   - `card_field_layout_inference: card_id=… rows=… field_rows=… recovered=… normalized=… button_skipped=… text_skipped=…`
   - `card_field_layout_inference: total inferred=…`
   - При fallback: `input_candidate_recovery summary: …`

---

## Критерии успеха

- В каждой card стабильно находится **≥1 input**.
- Input **без рамки** определяется по layout.
- Одинаковые поля **выровнены по ширине**, дубли убраны.
- **Textarea ≠ input** (по высоте ≥1.8× медианы).
- **Button rows** не дают инпутов; инпуты не поверх кнопок.
- Результат **слабо зависит от темы** (dark/light) и цвета/толщины рамки.
