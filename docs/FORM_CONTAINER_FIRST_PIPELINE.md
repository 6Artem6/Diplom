# Пайплайн Form Container First — полный алгоритм

**Источник истины:** код в `src/infrastructure/atoms_v2/experimental_v2/`.  
**Точка входа:** `run_form_container_first_inference()` в `run_form_container_first_inference.py`.

> ⚠️ **ЦЕЛЕВАЯ АРХИТЕКТУРА:** См. [FORM_PIPELINE_STATE_MACHINE.md](./FORM_PIPELINE_STATE_MACHINE.md) — детерминированная state machine без обратных связей.

При изменении пайплайна форм обновлять этот документ и при необходимости секцию 5.1 в `PIPELINE_ARCHITECTURE.md`.

---

## Инвариант №0

Ни строка, ни слот, ни поле не существуют вне `FormContainer.bbox`. Все уровни работают только внутри контейнера.

---

## Общая схема уровней

| Уровень | Назначение | Модуль / функция |
|--------|------------|-------------------|
| 0 | Детекция контейнера формы | FormContainerDetector |
| 0.5 | **Расширенная визуальная детекция** | `visual_field_scanner.py` (scan_all, postprocess, checkbox_radio_priority) |
| 1 | Строки и колонки внутри контейнера | FormInnerLayout (`form_inner_layout.py`) |
| 1.1 | **ContainerLeafDetection (диагностика)** | `container_leaf_detector.py` |
| 2 | Слоты по строкам (label, input, helper, action) | SlotDetector (`slot_layout_inference.py`) |
| 3 | Привязка визуальных bbox к слотам | RoleBasedFieldLocator |
| 4 | Граф формы и преобразование в атомы | FormGraphAssembler |
| 4.1 | **Диагностика atoms=0** | `_diagnose_zero_atoms()` |

---

## Пошаговый алгоритм

### Вход

- `image_path` — путь к скриншоту
- `raw_ocr_boxes` — полностраничный OCR (список `{ "bbox": [x1,y1,x2,y2], "text": "..." }`)
- `dark_theme` — флаг темы
- `debug_output_dir` — опционально, каталог для отладочных изображений
- `detectron_regions` — опционально, регионы от детектора

### Шаг 0. FormContainerDetector

1. Вызов `detect_form_containers(image_path, detectron_regions)` → список контейнеров и диагностика.
2. Выбор одного контейнера: `get_best_container(containers, demo_mode=demo_mode)` (в demo_mode — предпочтение по площади).
3. При отсутствии контейнера — выход с пустым результатом.
4. **Данные:** `container` с полем `bbox` (x1, y1, x2, y2). OCR и CV дальше ограничены этим bbox.

### Шаг 1. Подготовка данных внутри контейнера

1. **OCR внутри контейнера:** фильтрация `raw_ocr_boxes` по условию: центр bbox лежит внутри `container.bbox`. Результат: `ocr_inside`.
2. **Валидация контейнера по OCR:** `validate_container_with_ocr(container.bbox, ocr_inside)` — требуется не менее 2 OCR в разных Y; при неудаче — выход. В `demo_mode` без OCR пропускается.
3. **Визуальные кандидаты (базовый скан):** вызов `run_visual_field_scan(image_path, [{"bbox": container.bbox}], dark_theme)` → список bbox полей. Результат: `visual_candidates`. При ошибке сканера — пустой список.

### Шаг 1.5. Расширенная визуальная детекция (NEW)

Модуль: `visual_field_scanner.py`. Этап выполняется после базового скана для детального анализа элементов.

1. **Приоритетная детекция checkbox/radio:** `detect_checkbox_radio_priority(image_path, container.bbox)`:
   - Бинаризация OTSU + поиск контуров
   - Фильтрация: размер 12-32px, aspect ≈ 1.0 (±0.35), fill_ratio < 0.7
   - Определение типа: circularity > 0.7 → radio, иначе checkbox
   - Результат: список `{bbox, element_type, confidence, source}`

2. **Основная детекция всех элементов:** `scan_all_visual_elements(image_path, container.bbox)`:
   - Цветовая сегментация (HSV saturation > 25) → кнопки, иконки
   - Edge detection (Canny) → поля ввода, textarea, секции
   - Адаптивный threshold → секции, контейнеры
   - Классификация `_classify_element_type()` по приоритету:
     1. checkbox/radio (маленькие квадратные 12-32px)
     2. button (цветные, 25-75px высотой)
     3. textarea (высота >80px, aspect <4)
     4. input (28-60px, aspect ≥3)
     5. section/label

3. **Объединение:** checkbox/radio + остальные элементы (checkbox/radio имеют приоритет)

4. **Пост-обработка:** `postprocess_visual_elements(elements, container_bbox)`:
   - **Разделение контейнеров и leaf:** элемент считается контейнером если:
     - Площадь ≥ 10000px² (100×100)
     - НЕ является input/textarea/button/checkbox/radio
     - Содержит другой элемент (>85% площади внутреннего)
     - Минимум в 1.5 раза больше содержимого
   - **Обработка вложенных label:** если label высотой > 2× median_text_height и содержит другие элементы → переклассифицируется в row_container
   - **NMS с логированием:** сортировка по (is_container, -confidence, area), IoU threshold 0.3 для checkbox/radio/button, 0.5 для остальных

5. **Результат:**
   - `visual_elements` — обработанные элементы с типами
   - `extra_candidates` — все bbox (для валидации контейнера)
   - `leaf_candidates` — только leaf-элементы (для построения строк)

6. **Объединение с visual_candidates:** дедупликация по IoU > 0.4

### Шаг 1.6. Валидация контейнера по визуалу

Только при непустых `visual_candidates` и не `demo_mode`: `validate_container_with_visual(container.bbox, visual_candidates)`.

**Смягчённые требования (для маленьких форм):**
- Минимум 1 row anchor (было 3)
- Минимум 1 non-button элемент (было 2)
- Минимальная высота контейнера 100px (было 120)
- Минимальная высота элемента 20px (было 40, для checkbox ~12-20px)

При неудаче — выход с логированием причины.

### Шаг 2. FormInnerLayout (`build_form_inner_layout`)

Вход: `image_path`, `container`, `ocr_inside`, `visual_candidates` (опционально).

#### 2.1. Нормализация OCR для layout

- Вызов `normalize_ocr_for_layout(ocr_inside, container.bbox, image_path)`.
- **Baseline:** по OCR внутри контейнера (исключая верхние 18% зоны, ACTION-слова, центрированные широкие кнопки) считается медиана высоты шрифта → `baseline["median_font_height"]`.
- **layout_ocr:** подмножество `ocr_inside`, прошедшее gating:
  - не в header-зоне (верх 18% контейнера, `HEADER_ZONE_TOP_RATIO = 0.18`);
  - высота не больше `median_font * FONT_HEADER_RATIO` (1.5);
  - высота не меньше `median_font * FONT_HINT_RATIO` (0.6);
  - не ACTION-слова, не центрированная широкая кнопка.
- **header_bboxes:** bbox крупного текста в верхней зоне (далее не участвуют в построении строк, учитываются в `form_start_y` и `skipped_bboxes`).
- OCR в layout **не задаёт** границы строк; используется только для семантики (label, helper, ACTION).

#### 2.2. Построение строк

**Визуальный путь** (если задан непустой `visual_candidates` и `len(container.bbox) >= 4`):

1. **Объединение кандидатов:** `all_visual = visual_candidates + tall_contours`. Высокие контуры (`_tall_contours_inside_container`) — контуры по изображению высотой ≥ 80px, не пересекающиеся по Y с уже имеющимися визуальными bbox.
2. **Якоря строк:** `collect_field_row_anchors(all_visual, container.bbox, image_path, layout_ocr, baseline)`:
   - фильтрация bbox строго внутри контейнера;
   - сортировка по (y_min, x_min);
   - кластеризация по Y-overlap с зазором 40px; высокие (h ≥ 80px) не объединяются с низкими;
   - **разделение кластера:** если в одном Y-кластере есть крупный OCR (font > baseline×1.35) и bbox нормальной высоты — кластер разделяется: отдельный якорь с `from_ocr_header=True` (→ строка HEADER в _rows_from_visual_anchors), остальные bbox — отдельный якорь;
   - каждый кластер → якорь с полями `bboxes`, `y_min`, `y_max`, `x_min`, `x_max` (и опционально `from_ocr_header`).
3. Если якоря непусты:
   - **Строки из якорей:** `_rows_from_visual_anchors(anchors, container.bbox, ocr_inside)`:
     - **median_input_height:** медиана высот всех bbox из якорей с h < 80px.
     - **textarea_threshold:** `max(80, 1.6 * median_input_height)`.
     - Для каждого якоря:
       - **TEXTAREA:** только если один bbox в якоре и его высота ≥ textarea_threshold (два поля в одном якоре не объединяются в textarea).
       - **ACTION:** один bbox в якоре и в зоне строки есть OCR с ACTION-словом.
       - **Границы строки:** при одном bbox — привязка к полю: `row_y_min/max = bbox[1]±6px` / `bbox[3]±6px`; при нескольких — по min/max Y кластера.
       - **GRID:** при нескольких bbox в якоре и X-overlap между соседними < 0.3 — `column_count = len(bboxes)`, `input_bboxes` (отсортированы по X), `vertical_separators` (середины между соседними bbox по X).
       - Для каждой строки задаётся `input_bbox` (один bbox или объединение по кластеру).
   - **Нормализация пересечений:** `_normalize_row_overlaps(rows, container_y1, container_y2)` — строки сортируются по y_min; при пересечении по Y: если overlap меньше порога (max(15px, 20% от меньшей высоты строки)) — раздвижка границы по середине; иначе — слияние строк. Переиндексация `row_index`.
4. Если якоря пусты (при непустых visual_candidates) или visual_candidates не передан — **fallback:** `_build_rows_inside_container(image_path, container, layout_ocr, ocr_inside)` — строки по OCR (кластеризация по Y) и/или по Canny edges; `_normalize_row_overlaps` не вызывается.

**Режим demo_mode** (при непустых `visual_candidates` и `len(container.bbox) >= 4`): строки строятся через `_build_rows_demo_mode(container, visual_candidates)` — одна строка на один визуальный bbox, тип ACTION при кнопке, иначе FIELD_VERTICAL; затем `_post_process_rows_demo(container, rows, layout_ocr)`. Дальше общая цепочка: `_apply_row_invariants` → патч инвариантов → `_remove_orphan_field_rows`.

#### 2.3. Header и form_start_y

- Все bbox из `header_bboxes` добавляются в `skipped_bboxes`.
- `form_start_y` сдвигается ниже низа header’ов при их наличии.

#### 2.4. Постобработка строк (`_post_process_rows`)

Для каждой строки:

1. **TEXTAREA:** уточнение нижней границы по изображению:
   - используются границы **поля ввода** `r.input_bbox` (если есть), иначе границы строки;
   - вызов `find_first_horizontal_line_below(image_path, container.bbox, input_top, input_bottom, input_left, input_right)` — поиск первой устойчивой горизонтальной линии (Canny) ниже поля;
   - при нахождении линии ниже текущего `r.y_max` — `r.y_max` обновляется; иначе остаётся bbox. OCR placeholder в этой логике не участвует.

2. **Label и тип строки (только для полевых типов):**
   - **Label сверху:** OCR из layout_ocr с `bbox[3] <= верх input (iy_min) + 8px` (`LABEL_ABOVE_INPUT_TOP_GAP_PX`) и пересечением по X с `input_bbox`.
   - **ocr_left_of_input:** OCR с `bbox[2] <= ix_min + 10` и центром по Y в пределах поля.
   - **ocr_right_of_input:** OCR с `bbox[0] >= ix_max - 10` и центром по Y в пределах поля.
   - Для кандидата в label проверяется `_is_valid_label_text(text)`: нормализованная строка длины ≥ 2 и наличие хотя бы одной буквы/цифры.
   - При валидном label **сверху:** `label_bbox`, `row_type = FIELD_VERTICAL`, `vertical_split_y` между низом label и верхом зоны ввода (по OCR в строке).
   - При валидном label **слева:** `label_bbox`, `row_type = FIELD_HORIZONTAL`.
   - При валидном label **справа:** `right_label_bbox`, `row_type = FIELD_HORIZONTAL`.
   - Если ни label, ни right_label не найдены — `row_type = FIELD_INPUT_ONLY`.

3. **Helper:**
   - Кандидаты: OCR в строке, центр по Y ≥ центр строки, `bbox[3] <= r.y_max`, центр по Y ≥ низ input − 5px; исключаются крупный шрифт и центрированные кнопки.
   - **Helper назначается только если у кандидата есть текст:** проверка `_is_valid_label_text(txt)`. Пустая зона не считается helper’ом; `r.helper_bbox` остаётся `None`.

4. В строках сохраняются: `label_bbox`, `right_label_bbox`, `input_bbox`, `helper_bbox`, `vertical_split_y` (для FIELD_VERTICAL).

#### 2.4a. Инварианты строк и патч (`_apply_row_invariants` → `enforce_form_row_invariants` → `_remove_orphan_field_rows`)

**Жёсткий инвариант:** CV — единственный источник геометрии строк. `row.y_min`, `row.y_max`, `row.x_min`, `row.x_max` формируются только из visual anchors или fallback CV/edges. OCR не может изменять эти значения.

После `_post_process_rows` выполняется (в том же порядке):

1. **`_apply_row_invariants(rows, container.bbox)`** — геометрия по CV/input: верх/низ строки по input/label, мин/макс высота строки, мин ширина; границы в пределах контейнера.
2. **`enforce_form_row_invariants(rows, layout_ocr, container.bbox, baseline)`** (модуль `form_invariants_patch.py`):
   - **Границы строк:** при расхождении «row не покрывает все OCR в строке» — только **диагностика** (лог), координаты строки не меняются.
   - **TEXTAREA нижняя граница:** задаётся только по `input_bbox` или по горизонтальной линии (CV) в `_post_process_rows`. При расхождении с OCR — только лог, `row.y_max` не меняется.
   - **Защита заголовка:** первая строка не считается input при условиях → `row_type = HEADER` (семантика, не геометрия).
   - **Классификация input / изоляция label / label из строки выше:** OCR влияет только на `row_type`, `label_bbox`, `vertical_split_y` и т.п., не на границы строки.
3. **`classify_rows(rows, layout_ocr, container.bbox, baseline)`** (модуль `row_semantic_classifier.py`) — **строгий семантический классификатор** (между CV-якорями и SlotDetector). Меняет только `row_type` и `input_bbox` (обнуление); границы строк не меняются. Правила:
   - **HEADER:** OCR в строке выше baseline×1.35, нет визуальной рамки поля, нет placeholder, нет label сверху → `row_type=HEADER`, `input_bbox=None`.
   - **ACTION:** ширина строки ≥ 60% контейнера, OCR ≤3 слова, центрирован, нет placeholder/label сверху → `row_type=ACTION`, не создавать FIELD.
   - **TEXT:** нет визуального прямоугольника поля, нет placeholder → `row_type=TEXT`, `input_bbox=None`.
   - **FIELD** допускается только при: визуальная рамка + высота bbox близка к median_input_height + (placeholder или label). Иначе → TEXT/HEADER.
4. **`run_leaf_element_detection(rows, image_path, ocr_inside, container.bbox)`** (модуль `leaf_element_detector.py`) — **диагностический слой** (Stage 1). Для каждой строки детектирует element-like паттерны внутри `row.bbox` (button, checkbox, radio, textarea). НЕ меняет `row_type`, geometry, `input_bbox`, slots. Только добавляет `row.metadata["leaf_candidates"]` и `row.metadata["leaf_debug"]`, логирует результаты. Используется для анализа конфликтов root vs leaf.
5. **`_remove_orphan_field_rows(rows)`** — удаление строк с полевым типом без `input_bbox`/`input_bboxes`; переиндексация `row_index`.

#### 2.5. Отладочная структура rows_debug (и ocr_considered_for_label_only)

Для каждой строки формируется запись:

- `row_index`, `row_y_from_visual`: (y_min, y_max) — границы строки из CV;
- `ocr_considered_for_label_only`: список bbox OCR с пересечением по X с input и условием `bbox[3] <= r.y_min + 25` или `bbox[2] <= ix_min + 10` (для отладки; на последующие шаги не влияет).

#### 2.6. Пустой список строк

Если после всего строк нет — создаётся несколько дефолтных строк с фиксированным шагом по Y внутри контейнера.

#### 2.7. Тип layout и колонки

- **layout_type:** `_infer_layout_type(rows, ...)` — `"grid"` только если у всех полевых строк `row_type == FIELD_HORIZONTAL` и хотя бы у одной `column_count > 1`; иначе `"vertical"`.
- При **layout_type == "vertical":** для каждой строки: если у строки уже `column_count > 1` и есть `input_bboxes` (≥2), то `column_count` и `input_bboxes` **не меняются**; иначе `r.column_count = 1`.
- При **layout_type == "grid":** колонки и границы берутся из первой строки с `input_bboxes` (если есть), иначе — `_build_columns_inside_container` (Canny по изображению, кластеризация X-центров). Для TEXTAREA у строки принудительно `column_count = 1`; для остальных — по числу колонок.

Формируются `FormSkeleton` (form_region, rows, columns, column_boundaries, layout_type) и диагностика (n_rows, layout_type, skipped_bboxes, textarea_row_indices, rows_debug).

---

### Шаг 3. SlotDetector (`build_slot_layout`)

Вход: `skeleton`, `raw_ocr_boxes`. Для каждой строки скелета вычисляется `ocr_in_row` (OCR с центром внутри row bbox) и вызывается `infer_slots_for_row(row, skeleton, ocr_in_row)`.

**Порядок веток в `infer_slots_for_row` (приоритет сверху вниз):**

1. **TEXT / HEADER:** слотов нет; возврат.
2. **ACTION:** один слот `action_slot` на всю строку; возврат.
3. **Визуальная сетка (приоритет):** если у строки есть `input_bboxes` и `len(input_bboxes) >= 2`:
   - по каждому bbox из `input_bboxes` создаётся слот `input_slot` (или `textarea_slot` при row_type == TEXTAREA) с геометрией bbox;
   - при наличии `label_bbox` в начало списка слотов вставляется `label_slot` по этому bbox;
   - при наличии `helper_bbox` добавляется `helper_slot` по этому bbox;
   - возврат (независимо от layout_type и row_type).
4. **FIELD_INPUT_ONLY:** один слот `input_slot` по `row.input_bbox` или по bbox строки; возврат.
5. **FIELD_VERTICAL** (есть `vertical_split_y`): слоты `label_slot` (верх до vertical_split_y / label_bbox[3]), `input_slot` (от vertical_split_y до низа строки), при наличии `helper_bbox` — `helper_slot`; возврат.
6. **FIELD_HORIZONTAL с right_label_bbox:** деление вертикальными границами: `input_slot` по `row.input_bbox`, `label_slot` по `row.right_label_bbox`; при наличии `helper_bbox` — `helper_slot`; возврат.
7. **column_count > 1 (grid по скелету):** слоты по `row.input_bboxes` (если есть и достаточно) или по `skeleton.column_boundaries` — по одному input/textarea слоту на колонку.
8. **Иначе (вертикальный ratio-layout):** слоты по фиксированным долям ширины: `label_slot` 25%, `input_slot` 65%; затем только при наличии:
   - при OCR ACTION в строке — `action_slot` до конца строки;
   - иначе при наличии `row.helper_bbox` — `helper_slot` по геометрии `helper_bbox`. Пустой helper-слот до конца формы не создаётся.

Итог: список `RowSlots` по строкам, каждый элемент содержит список `Slot` с ролями и `expected_bbox_hint`.

---

### Шаг 4. FieldLocator (`run_role_based_locator`) и коррекция назначений

1. **Вход:** `row_slots`, `visual_candidates`, `image_path`, `dark_theme`, `container_bbox`. Назначение визуальных bbox слотам: для каждого input/textarea слота (action_slot не назначается полю) выбирается лучший кандидат из `visual_candidates` (пересечение с `expected_bbox_hint`, скоринг), формируется список `SlotAssignment` (slot, bbox, confidence, field_type).
2. **Коррекция bbox по инвариантам слотов:** вызов `correct_slot_assignment_bboxes(assignments, skeleton, raw_ocr_boxes)` — меняется только `assignment.bbox`; `row.y_min`/`row.y_max` не изменяются. Сдвиг/расширение bbox по label и placeholder.
3. **Строгий debug:** `log_assignment_outside_row_invariant(assignments, skeleton)` — если `assignment.bbox` выходит за границы своей строки, логируется нарушение инварианта; строка не меняется.

---

### Шаг 5. FormGraph и атомы

1. **assemble_form_graph(skeleton, row_slots, assignments):** строится граф формы: слоты, назначения slot→bbox, связи label→input и input→helper по ролям слотов в каждой строке.
2. **Проверка инвариантов:** `_validate_form_invariants(container, skeleton, row_slots)` — площадь контейнера, высоты строк, ширина input_slot; результат записывается в `graph.metadata["invariant_violations"]` и `graph.metadata["invariant_violation"]`.
3. **demo_mode:** при заданном `debug_output_dir` сохраняются demo_*.json, строится `demo_visualization.png`, вызывается `validate_demo_pipeline`; при неудаче валидации пайплайн останавливается (BPG не строится).
4. **form_graph_to_atoms(graph, existing_atom_ids, recovery_source="form_container_first"):** по каждому назначению с непустым bbox создаётся атом типа `input_candidate` или `textarea_candidate` с id-префиксом `fcf_`, bbox, confidence и evidence (source, slot_id, slot_role).

---

## Роли OCR и CV (сводка)

| Данные | Использование |
|--------|----------------|
| **Границы строк (row_y_min, row_y_max)** | Только CV: якоря из visual_candidates + высокие контуры; при отсутствии якорей — fallback по OCR/edges. OCR не расширяет и не задаёт границы строк. |
| **TEXTAREA** | Только один bbox в якоре и высота ≥ max(80, 1.6×median_input_height). Нижняя граница — по горизонтальной линии (CV) от границ поля ввода. |
| **ACTION** | OCR в зоне строки с ACTION-словом при одном bbox в якоре. |
| **Label (сверху/слева/справа)** | OCR из layout_ocr с проверкой позиции и `_is_valid_label_text`. Записываются label_bbox, right_label_bbox, vertical_split_y. После постобработки патч инвариантов может задать label из строки выше (`enforce_form_row_invariants`). |
| **Helper** | OCR в строке ниже/справа от поля; только если текст валиден (`_is_valid_label_text`). Пустой helper не создаётся. |
| **Колонки (grid)** | По визуальным bbox: X-overlap < 0.3 → input_bboxes и vertical_separators. При layout_type == "vertical" у строк с input_bboxes column_count не обнуляется. |
| **Инварианты строк (патч)** | CV — единственный источник геометрии строк. Патч: только диагностика при расхождении row↔OCR (лог); изменение row_type, label_bbox, helper, label from above; границы строк не меняются. TEXTAREA bottom только по CV. |
| **Семантический классификатор** | `row_semantic_classifier.classify_rows`: жёсткие правила HEADER/ACTION/TEXT/FIELD после патча; заголовок и кнопка не становятся FIELD; FIELD только при визуальная рамка + высота ≈ median + (placeholder или label). Геометрию не меняет. |
| **Расширенная визуальная детекция (Stage 0.5)** | `visual_field_scanner`: `detect_checkbox_radio_priority` → `scan_all_visual_elements` → `postprocess_visual_elements`. Приоритет: checkbox/radio → button → textarea → input. Разделяет контейнеры и leaf. NMS с логированием. |
| **ContainerLeafDetection (Stage 1.1)** | `container_leaf_detector.run_container_leaf_detection`: диагностика ДО row segmentation. **Геометрическая фильтрация** checkbox/radio: fill_ratio < 0.45, inner_contours ≤ 2, edge_density < 0.25, aspect_ratio ∈ [0.85, 1.15], symmetry ≥ 0.8, max_size ≤ 40px. Результат: `container.metadata["container_leaf"]`. |
| **LeafElementDetection (Stage 1)** | `leaf_element_detector.run_leaf_element_detection`: диагностический слой, детектирует button/checkbox/radio/textarea внутри row.bbox. Не меняет row_type, geometry, slots. Записывает `row.metadata["leaf_candidates"]`. Визуализация: `leaf_detection.png` — строки с цветом по типу кандидата + confidence справа. |
| **Диагностика atoms=0** | `_diagnose_zero_atoms`: при неудаче demo_validation выводит детальный отчёт: контейнер, строки, слоты, assignments, visual_elements, причины отклонения. Сохраняет `atoms_zero_diagnostics.txt/json`. |
| **Коррекция bbox назначений** | `correct_slot_assignment_bboxes`: меняется только `assignment.bbox`; row не меняется. `log_assignment_outside_row_invariant`: лог при bbox вне строки. |

---

## Модели данных (ключевые поля)

- **FormRow:** row_index, y_min, y_max, x_min, x_max, column_count, row_type, label_bbox, right_label_bbox, input_bbox, input_bboxes, helper_bbox, vertical_split_y, vertical_separators, height_confidence.
- **RowType:** FIELD_HORIZONTAL, FIELD_VERTICAL, FIELD_INPUT_ONLY, TEXTAREA, ACTION, TEXT, HEADER, FIELD, SPACER.
- **Slot:** slot_id, role (label_slot, input_slot, textarea_slot, helper_slot, action_slot), row_index, column_index, x_min, x_max, y_min, y_max, width_hint, height_hint, expected_bbox_hint, metadata.

---

## Отладочные выходы (при заданном debug_output_dir)

### Основные

- `container_bbox.png` — контейнер формы
- `rows.png` — границы строк
- `rows_with_types.png` — строки с цветом по row_type (усиленная визуализация)
- `skipped_rows.png` — пропущенные bbox (header)
- `textarea_rows.png` — строки типа TEXTAREA
- `rows_debug.png` — row_y_from_visual и ocr_considered_for_label_only
- `slots.png` — слоты по строкам
- `slot_assignments.png` — назначения bbox слотам
- `form_graph.png` — граф (bbox и связи label→input)

### Диагностика визуальных элементов (NEW)

- `elements_enhanced.png` — усиленная контрастная визуализация всех элементов:
  - HEADER: толстая синяя рамка 4px
  - INPUT: зелёная заливка 40%
  - TEXTAREA: оранжевая заливка 40%, пунктирная рамка
  - ACTION: красная заливка 60%
  - CHECKBOX: рамка + крестик
  - RADIO: рамка + кружок
  - Подписи: type, confidence, source
- `detection_diagnostics.txt` — текстовая диагностика детекции
- `elements_nesting.png` — визуализация вложенности элементов (если есть)
- `elements_overlaps.png` — визуализация пересечений (если есть)

### Диагностика при ошибках (NEW)

- `atoms_zero_diagnostics.txt` — детальный отчёт при atoms=0
- `atoms_zero_diagnostics.json` — JSON для программного анализа
- `no_container_found.txt` — маркер при остановке пайплайна

### LeafDetection

- `leaf_detection.png` — строки с цветом по типу кандидата + confidence
- `container_leaf_detection.png` — элементы вне строк (Stage 1.1)

---

---

## Новые модули диагностики и визуализации

### detection_diagnostics.py

Сбор статистики и диагностика детекции:

- `DetectionDiagnostics` — dataclass со статистикой: container, visual_candidates, by_type, nested_elements, overlapping_pairs, suppression_reasons, atoms_by_type
- `compute_iou()` — расчёт IoU двух bbox
- `bbox_contains()` — проверка вложенности
- `analyze_nesting()` — анализ вложенности и пересечений элементов
- `diagnose_visual_candidates()` — сбор диагностики
- `apply_suppression_with_log()` — NMS с логированием причин удаления
- `separate_containers_and_leaves()` — разделение контейнеров и leaf
- `log_diagnostics()` / `print_diagnostics()` — вывод диагностики

### enhanced_visualization.py

Усиленная контрастная визуализация:

- Цвета по типам: HEADER (синий), INPUT (зелёный), TEXTAREA (оранжевый), ACTION (красный), CHECKBOX/RADIO (пурпурный), LABEL (голубой)
- `draw_filled_rect()` — прямоугольник с прозрачной заливкой
- `draw_dashed_rect()` — пунктирная рамка
- `draw_checkbox_marker()` / `draw_radio_marker()` — специальные маркеры
- `draw_element()` — полная визуализация элемента с подписью
- `visualize_elements_enhanced()` — создание изображения со всеми элементами и легендой
- `visualize_nesting()` — визуализация вложенности стрелками
- `visualize_overlaps()` — визуализация пересечений

---

## Как обновлять документ

При изменении пайплайна Form Container First:

1. Внести правки в код в `src/infrastructure/atoms_v2/experimental_v2/`.
2. Обновить этот файл (`docs/FORM_CONTAINER_FIRST_PIPELINE.md`): соответствующие шаги, таблицы, инварианты.
3. При необходимости обновить секцию 5.1 и ссылки в `docs/PIPELINE_ARCHITECTURE.md`.

---

## История изменений

| Дата | Изменение |
|------|-----------|
| 2026-02-03 | Добавлены этапы 0.5 (расширенная визуальная детекция) и 4.1 (диагностика atoms=0). Новые модули: detection_diagnostics.py, enhanced_visualization.py. Смягчены инварианты валидации контейнера. Добавлены новые debug-файлы. |
