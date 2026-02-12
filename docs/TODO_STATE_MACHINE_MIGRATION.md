# TODO: Миграция на State Machine архитектуру

**Целевой документ:** [FORM_PIPELINE_STATE_MACHINE.md](./FORM_PIPELINE_STATE_MACHINE.md)

---

## Этап 1: S1 — Visual Geometry Extraction

### 1.1 Убрать абсолютные размеры

**Файл:** `src/infrastructure/atoms_v2/visual_field_scanner.py`

- [ ] `CHECKBOX_RADIO_SIZE_MIN = 12` → `0.015 * container_size` или относительно median
- [ ] `CHECKBOX_RADIO_SIZE_MAX = 32` → `0.04 * container_size`
- [ ] `TEXTAREA_HEIGHT_MIN = 60` → `1.8 * median_input_height`
- [ ] `INPUT_HEIGHT_TYPICAL = 28` → вычислять median
- [ ] `INPUT_HEIGHT_MAX = 50` → `1.5 * median_input_height`

**Файл:** `src/infrastructure/atoms_v2/experimental_v2/form_inner_layout.py`

- [ ] `MIN_ROW_HEIGHT = 14` → относительно median
- [ ] `TEXTAREA_MIN_HEIGHT_PX = 60` → `1.8 * median_input_height`
- [ ] `LABEL_ABOVE_MAX_GAP_PX = 50` → `1.5 * median_input_height`
- [ ] Все `*_PX` константы → относительные

### 1.2 OCR overlap check

**Файл:** `src/infrastructure/atoms_v2/visual_field_scanner.py`

```python
def _should_reject_bbox(bbox, ocr_blocks):
    """
    Reject bbox if:
    - 70% overlaps with OCR block
    - AND no visible border contrast inside
    """
    for ocr in ocr_blocks:
        overlap = compute_overlap(bbox, ocr["bbox"])
        if overlap > 0.7:
            if not has_visible_border(bbox):
                return True
    return False
```

- [ ] Добавить `has_visible_border()` функцию
- [ ] Интегрировать в `_scan_all_visual_elements_roi()`
- [ ] Передавать `ocr_blocks` в детекцию

### 1.3 Textarea containment check

- [ ] Добавить проверку: если внутри bbox есть другие bbox → это container, не textarea
- [ ] Добавить проверку: textarea должна иметь видимую рамку

### 1.4 Checkbox symmetry recovery

- [ ] Кластеризация checkbox по размеру (±10%) и alignment (±0.3× height)
- [ ] Если в кластере 1 элемент → поиск парного checkbox рядом

---

## Этап 2: S2 — OCR Extraction

### 2.1 Language detection

**Файл:** Новый `src/infrastructure/atoms_v2/experimental_v2/language_detector.py`

```python
def detect_language(ocr_blocks):
    all_text = " ".join(b["text"] for b in ocr_blocks)
    cyrillic = sum(1 for c in all_text if '\u0400' <= c <= '\u04FF')
    latin = sum(1 for c in all_text if 'a' <= c.lower() <= 'z')
    total = cyrillic + latin
    
    if total == 0:
        return LanguageContext("unknown", 0.0)
    
    if cyrillic / total >= 0.6:
        return LanguageContext("ru", cyrillic / total)
    elif latin / total >= 0.6:
        return LanguageContext("en", latin / total)
    else:
        return LanguageContext("mixed", 0.5)
```

- [ ] Создать модуль
- [ ] Интегрировать в пайплайн после OCR

---

## Этап 3: S3 — Structural Segmentation

### 3.1 Относительный gap для row clustering

**Файл:** `src/infrastructure/atoms_v2/experimental_v2/form_inner_layout.py`

- [ ] `gap_y = 0.6 * median_input_height` вместо `40px`
- [ ] Вычислять `median_input_height` ДО кластеризации

### 3.2 Grid detection

- [ ] Проверка horizontal overlap < 15%
- [ ] `max_merge_distance = 3.0 * median_input_width`

---

## Этап 4: S4 — Slot Assignment

### 4.1 Max label distance

- [ ] `max_label_distance = 1.5 * median_input_width`
- [ ] Если дальше → не считать label

### 4.2 Checkbox/Radio label position

- [ ] checkbox label → справа
- [ ] radio label → справа
- [ ] input label → слева или сверху

---

## Этап 5: S5 — Pattern Analysis (NEW)

**Файл:** Новый `src/infrastructure/atoms_v2/experimental_v2/pattern_analysis.py`

### 5.1 Pattern detection

```python
@dataclass
class PatternGroup:
    pattern_type: str  # checkbox_label, radio_label, input_grid
    elements: List[VisualElement]
    labels: List[OCRBlock]
    recovered_elements: List[VisualElement]

def detect_patterns(row, visual_elements, ocr_blocks):
    # 1. Cluster by size similarity
    # 2. Check vertical alignment
    # 3. Identify repeating structures
    pass
```

- [ ] Создать модуль
- [ ] Реализовать `detect_patterns()`
- [ ] Реализовать `recover_missing_elements()`

---

## Этап 6: S6 — Semantic Validation

**Файл:** Новый `src/infrastructure/atoms_v2/experimental_v2/semantic_validation.py`

### 6.1 Language-aware validation

```python
RU_PATTERNS = ["Имя", "Телефон", "Пароль", "Email", "Адрес", "Фамилия"]
EN_PATTERNS = ["Name", "Phone", "Password", "Email", "Address", "First", "Last"]

def validate_labels(slots, language_context):
    patterns = RU_PATTERNS if language_context.primary == "ru" else EN_PATTERNS
    # validation logic
```

- [ ] Создать модуль
- [ ] Только flags, НЕ изменять geometry

---

## Этап 7: Интеграция

### 7.1 Обновить run_form_container_first_inference.py

- [ ] Порядок вызовов: S1 → S2 → S3 → S4 → S5 → S6 → S7
- [ ] Убрать обратные зависимости
- [ ] visual_elements immutable после S1

### 7.2 Тестирование

- [ ] Тесты на всех demo_forms
- [ ] Сравнение результатов до/после
- [ ] Проверка инвариантов

### 7.3 Документация

- [ ] Обновить FORM_CONTAINER_FIRST_PIPELINE.md
- [ ] Актуализировать примеры

---

## Порядок выполнения

1. **Срочно:** 1.2 (OCR overlap), 1.3 (textarea containment) — исправят текущие баги
2. **Важно:** 1.1 (относительные размеры), 3.1 (относительный gap)
3. **Новое:** 5.x (Pattern Analysis), 2.1 (Language detection)
4. **Рефакторинг:** 6.x (Semantic validation), 7.x (Интеграция)

---

## Чеклист инвариантов

После каждого изменения проверять:

- [ ] Visual elements не меняются после S1
- [ ] NMS вызывается только в S1
- [ ] OCR не создаёт bbox
- [ ] Semantic не меняет geometry
- [ ] Нет абсолютных размеров в новом коде
