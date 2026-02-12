# Form Pipeline State Machine

**Статус:** ✅ РЕАЛИЗОВАНО  
**Путь:** `src/infrastructure/atoms_v2/experimental_v2/`  
**Источник:** Анализ ошибок текущего пайплайна  
**Принцип:** Детерминированная state machine без обратных связей

---

## Реализованные модули

| Этап | Модуль | Статус |
|------|--------|--------|
| S1 | `visual_geometry_extractor.py` | ✅ |
| S2 | `ocr_extractor.py` | ✅ |
| S3 | `structural_segmentation.py` | ✅ |
| S4 | `slot_assignment.py` | ✅ |
| S5 | `pattern_analysis.py` | ✅ |
| S6 | `semantic_validation.py` | ✅ |
| S7 | `run_state_machine_pipeline.py` | ✅ |

### Запуск pipeline

```bash
# CLI — с автоматическим определением контейнера формы
python -m src.infrastructure.atoms_v2.experimental_v2.run_state_machine_pipeline \
    image.png -o ./output

# CLI — с явным указанием границ контейнера
python -m src.infrastructure.atoms_v2.experimental_v2.run_state_machine_pipeline \
    image.png --bbox 100 50 600 700 -o ./output

# Python API — автоматическое определение контейнера (S0)
from src.infrastructure.atoms_v2.experimental_v2 import (
    run_state_machine_pipeline, 
    PipelineConfig
)

result = run_state_machine_pipeline(
    image_path="form.png",
    # container_bbox не передан → авто-детект (S0)
    config=PipelineConfig(
        demo_mode=True, 
        output_dir="./output",
        auto_detect_container=True,
    )
)

# Python API — с явным bbox
result = run_state_machine_pipeline(
    image_path="form.png",
    container_bbox=[100, 50, 600, 700],
    config=PipelineConfig(demo_mode=True, output_dir="./output")
)
```

### S0: Автоматическое определение контейнера

Pipeline автоматически находит контейнер формы (card/panel) используя:
- OTSU бинаризацию для светлых областей на тёмном фоне
- Морфологические операции для объединения разрывов
- Скоринг по контрасту и центрированности

Результат доступен в:
- `result.container_bbox` — найденный bbox
- `result.container_source` — источник: `"auto_detected"`, `"provided"`, `"full_image"`
- `result.s0_container_confidence` — confidence автодетекта

---

## Ключевые принципы

1. **Никаких циклов назад** — каждый этап добавляет данные, но не меняет предыдущие
2. **Visual elements immutable после S1** — геометрия фиксируется один раз
3. **NMS только в S1** — один раз, в одном месте
4. **OCR не создаёт bbox** — только текстовая информация
5. **Semantic не меняет geometry** — только валидация и флаги
6. **Относительные размеры** — никаких абсолютных 40px, 80px

---

## Dependency Graph

```
Image
  ↓
[S1 Visual Geometry] ──────────────────┐
  ↓                                    │
[S2 OCR] ───────────────────────┐      │
  ↓                             │      │
[S3 Rows] ←─────────────────────┼──────┤
  ↓                             │      │
[S4 Slots] ←────────────────────┼──────┤
  ↓                             │      │
[S5 Pattern Analysis] ←─────────┼──────┤
  ↓                             │      │
[S6 Semantic Validation] ←──────┴──────┘
  ↓
[S7 Graph]
```

### Запрещённые зависимости

| От | К | Статус |
|----|---|--------|
| OCR | Visual classification | ❌ ЗАПРЕЩЕНО |
| Semantic | Geometry | ❌ ЗАПРЕЩЕНО |
| Slots | Visual mutation | ❌ ЗАПРЕЩЕНО |
| Pattern | Row resegmentation | ❌ ЗАПРЕЩЕНО |

---

## S0 — RAW INPUT

**Вход:** изображение  
**Выход:** `image_meta`

```python
@dataclass
class ImageMeta:
    path: str
    width: int
    height: int
    is_dark_theme: bool
```

**Переход:** → S1

---

## S1 — VISUAL GEOMETRY EXTRACTION

> ⚠️ **ЕДИНСТВЕННЫЙ ЭТАП** где разрешено находить bbox и классифицировать visual type

### Операции

1. **Detect elements:**
   - container
   - input
   - textarea
   - button
   - checkbox (checked + unchecked)
   - radio (selected + unselected)

2. **NMS** — ОДИН РАЗ

3. **Checkbox symmetry recovery:**
   ```
   После detection:
   cluster checkboxes by:
     - size similarity (±10%)
     - vertical alignment (±0.3× height)
   
   if cluster size == 1:
     search nearby for candidate square region
   ```

### Инварианты S1

| Правило | Описание |
|---------|----------|
| OCR overlap | ❌ НЕЛЬЗЯ детектировать bbox, если он 70% перекрывает OCR-блок и внутри нет геометрических границ |
| Textarea containment | ❌ НЕЛЬЗЯ классифицировать textarea, если внутри есть другие bbox |
| Relative sizes only | ❌ НЕЛЬЗЯ использовать абсолютные размеры |

### Допустимые метрики

```python
# ✅ ПРАВИЛЬНО
relative_to_container = element_width / container_width
relative_to_median = element_height / median_input_height
aspect_ratio = width / height

# ❌ НЕПРАВИЛЬНО
if height > 80:  # абсолютный размер
if width < 40:   # абсолютный размер
```

### Textarea validation

```python
def is_valid_textarea(bbox, median_input_height, all_elements):
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    aspect = width / max(1, height)
    
    # Относительные критерии
    if height < 1.8 * median_input_height:
        return False
    if aspect >= 4.0:
        return False
    
    # Нет других bbox внутри
    for other in all_elements:
        if bbox_contains(bbox, other["bbox"]):
            return False
    
    # Должна быть видимая рамка
    if not has_visible_border(bbox):
        return False
    
    return True
```

### Выход

```python
@dataclass
class VisualElement:
    bbox: List[float]  # [x1, y1, x2, y2]
    element_type: str  # input, textarea, button, checkbox, radio
    confidence: float
    is_checked: Optional[bool]  # для checkbox/radio
    has_border: bool
    source: str  # detection method
    
# Immutable после S1!
visual_elements: List[VisualElement]
```

**Переход:** → S2

---

## S2 — OCR EXTRACTION

> ⚠️ OCR не создаёт и не удаляет bbox

### Операции

1. **OCR detection** — текстовые блоки
2. **Language detection:**
   ```python
   cyrillic_ratio = count_cyrillic(text) / len(text)
   latin_ratio = count_latin(text) / len(text)
   
   if cyrillic_ratio >= 0.6:
       language = "ru"
   elif latin_ratio >= 0.6:
       language = "en"
   else:
       language = "mixed"
   ```

### Выход

```python
@dataclass
class OCRBlock:
    bbox: List[float]
    text: str
    confidence: float
    language: str  # ru, en, mixed

@dataclass
class LanguageContext:
    primary: str  # ru или en
    confidence: float

ocr_blocks: List[OCRBlock]
language_context: LanguageContext
```

**Переход:** → S3

---

## S3 — STRUCTURAL SEGMENTATION (ROWS)

> Строится ТОЛЬКО на visual_elements и container geometry. **НИКОГДА на OCR.**

### Row clustering

```python
# ✅ ПРАВИЛЬНО — относительный gap
gap_y = 0.6 * median_input_height

# ❌ НЕПРАВИЛЬНО — абсолютный gap
gap_y = 40  # px
```

### Grid detection

```python
def detect_grid_layout(row_elements):
    inputs = [e for e in row_elements if e.element_type == "input"]
    
    if len(inputs) >= 2:
        # Проверяем horizontal overlap
        overlaps = []
        for i, e1 in enumerate(inputs):
            for e2 in inputs[i+1:]:
                overlap = compute_x_overlap(e1.bbox, e2.bbox)
                overlaps.append(overlap)
        
        if max(overlaps) < 0.15:  # < 15% overlap
            return "row_grid"
    
    return "single"
```

### Max distance rule

```python
# Если расстояние > 3× median_input_width → запрещено объединять
max_merge_distance = 3.0 * median_input_width
```

### Выход

```python
@dataclass
class FormRow:
    row_index: int
    y_min: float
    y_max: float
    x_min: float
    x_max: float
    layout_type: str  # single, row_grid
    elements: List[VisualElement]  # references, not copies
```

**Переход:** → S4

---

## S4 — SLOT ASSIGNMENT

> Работает только с rows и visual_elements

### Slot rules

| Element | Label position |
|---------|---------------|
| input | слева или сверху |
| textarea | сверху |
| checkbox | справа |
| radio | справа |
| button | внутри или отсутствует |

### Max label distance

```python
# ⚠️ Если label расстояние > 1.5× median_input_width → это НЕ label
max_label_distance = 1.5 * median_input_width

def find_label_for_element(element, ocr_blocks, median_input_width):
    candidates = []
    for ocr in ocr_blocks:
        distance = compute_distance(element.bbox, ocr.bbox)
        if distance <= 1.5 * median_input_width:
            candidates.append(ocr)
    
    if not candidates:
        return None
    
    # Выбираем ближайший подходящий
    return select_best_label(candidates, element)
```

### Radio grouping

```python
def group_radio_buttons(radios, rows):
    """Группировка radio по строке и proximity"""
    groups = []
    for row in rows:
        row_radios = [r for r in radios if r in row.elements]
        if len(row_radios) >= 2:
            # Сортируем по X
            row_radios.sort(key=lambda r: r.bbox[0])
            groups.append(RadioGroup(row_radios))
    return groups
```

### Выход

```python
@dataclass
class Slot:
    slot_id: str
    role: str  # label, input, helper, action
    element: Optional[VisualElement]
    ocr_block: Optional[OCRBlock]
    row_index: int
```

**Переход:** → S5

---

## S5 — STRUCTURE PATTERN ANALYSIS (NEW)

> Поиск повторяющихся структур в пределах строки

### Pattern detection

```python
def detect_patterns(row):
    """
    Найти повторяющиеся структуры:
    - checkbox + label + checkbox + label
    - radio + label + radio + label
    - input + input (grid)
    """
    elements = row.elements
    
    # Cluster by size similarity
    size_clusters = cluster_by_size(elements, tolerance=0.1)
    
    # Cluster by vertical alignment
    for cluster in size_clusters:
        if len(cluster) >= 2:
            aligned = check_vertical_alignment(cluster, tolerance=0.3)
            if aligned:
                # Это pattern group
                pattern = PatternGroup(
                    elements=cluster,
                    pattern_type=infer_pattern_type(cluster)
                )
                yield pattern
```

### Symmetry recovery

```python
def recover_missing_elements(patterns, row, image):
    """
    Если найден только один checkbox в паттерне,
    ищем аналогичный bbox по размеру в пределах строки
    """
    for pattern in patterns:
        if pattern.pattern_type == "checkbox_label":
            checkboxes = [e for e in pattern.elements if e.element_type == "checkbox"]
            
            if len(checkboxes) == 1:
                # Ищем пропущенный checkbox
                cb = checkboxes[0]
                expected_positions = predict_pattern_positions(pattern)
                
                for pos in expected_positions:
                    if not has_element_at(pos, pattern.elements):
                        # Ищем квадратную область похожего размера
                        candidate = search_square_region(
                            image, pos,
                            size=cb.bbox_size(),
                            tolerance=0.1
                        )
                        if candidate:
                            yield candidate
```

### Выход

```python
@dataclass
class PatternGroup:
    pattern_type: str  # checkbox_label, radio_label, input_grid
    elements: List[VisualElement]
    labels: List[OCRBlock]
    recovered_elements: List[VisualElement]  # найденные через symmetry

pattern_groups: List[PatternGroup]
```

**Переход:** → S6

---

## S6 — SEMANTIC VALIDATION

> ⚠️ **ТОЛЬКО ВАЛИДАЦИЯ**, не изменение geometry

### Checks

1. **Language validation:**
   ```python
   if language_context.primary == "ru":
       expected_labels = ["Имя", "Телефон", "Пароль", "Email", "Адрес"]
   else:
       expected_labels = ["Name", "Phone", "Password", "Email", "Address"]
   ```

2. **Impossible combinations:**
   ```python
   # Textarea не может быть внутри другого элемента
   # Button не может быть label для input
   # Checkbox не может быть шире чем высокий
   ```

3. **Empty textarea check:**
   ```python
   if element.type == "textarea" and not has_visible_content(element):
       flag_inconsistency("empty_textarea", element)
   ```

### Выход

```python
@dataclass
class ValidationResult:
    element: VisualElement
    issues: List[str]  # inconsistency flags
    confidence_adjustment: float  # может понизить confidence

validation_results: List[ValidationResult]
```

> ⚠️ **НЕ МЕНЯЕТ** bbox, element_type, row assignment

**Переход:** → S7

---

## S7 — GRAPH ASSEMBLY

### FormGraph structure

```python
@dataclass
class FormGraph:
    container: VisualElement
    rows: List[FormRow]
    elements: List[VisualElement]
    slot_bindings: List[SlotBinding]
    pattern_groups: List[PatternGroup]
    validation_results: List[ValidationResult]
    language_context: LanguageContext
    
    # Metadata
    median_input_height: float
    median_input_width: float
```

### Финальный граф

```
FormGraph
├── container
├── rows[]
│   ├── row_index
│   ├── layout_type
│   └── elements[]
├── elements[] (references)
├── slot_bindings[]
│   ├── element → label
│   ├── element → helper
│   └── radio_group → [radios]
├── pattern_groups[]
└── validation_results[]
```

---

## Жёсткие инварианты

| # | Инвариант | Проверка |
|---|-----------|----------|
| 1 | Visual elements immutable после S1 | Freeze after S1 |
| 2 | NMS только в S1 | No NMS calls after S1 |
| 3 | OCR не может создавать bbox | OCR → text only |
| 4 | Semantic не может менять geometry | Validation → flags only |
| 5 | Абсолютные размеры запрещены | Lint check |
| 6 | Любая классификация относительная | Code review |

---

## Миграция с текущей архитектуры

### Этап 1: Рефакторинг visual_field_scanner.py

- [ ] Убрать абсолютные размеры (40px, 80px)
- [ ] Добавить OCR overlap check
- [ ] Добавить textarea containment check
- [ ] Добавить checkbox symmetry recovery

### Этап 2: Создание S5 (Pattern Analysis)

- [ ] Новый модуль `pattern_analysis.py`
- [ ] Pattern detection
- [ ] Symmetry recovery

### Этап 3: Language context

- [ ] Добавить language detection в OCR
- [ ] Использовать в semantic validation

### Этап 4: Semantic validation

- [ ] Отделить от geometry modification
- [ ] Только flags и confidence adjustment

### Этап 5: Интеграция

- [ ] Обновить `run_form_container_first_inference.py`
- [ ] Тестирование на demo_forms
- [ ] Обновить документацию

---

## Почему это устранит хаос

| Проблема | Решение |
|----------|---------|
| OCR-блоки поверх которых checkbox | S1: OCR overlap check |
| Большие блоки как textarea | S1: containment + border check |
| Только отмеченный checkbox | S1 + S5: symmetry recovery |
| Нет повторений структуры | S5: pattern analysis |
| Нет разделения ru/en | S2 + S6: language context |
| Grid labels через тысячи px | S4: max_label_distance |
| Абсолютные размеры | Относительные метрики везде |

---

## История изменений

| Дата | Изменение |
|------|-----------|
| 2026-02-03 | Создан документ целевой архитектуры |
