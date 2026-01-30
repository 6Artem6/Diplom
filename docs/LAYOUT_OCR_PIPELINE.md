# Layout & OCR Pipeline: анализ и этапы

Цель: **видеть текст так, как его видит человек**. Сначала понимаем интерфейс (регионы), потом читаем текст (OCR). Контейнер ≠ текст.

---

## 1. Текущий пайплайн (этапы)

```
image
  → cv_detect_regions()           # Region prepass: LAB, k-means, morphology, contours
  → regions (text_region / ui_region / background)
  → OCR (full page: grayscale, CLAHE, Otsu) → Words
  → run_cv_prepass()              # Обогащение слов: has_background, text_color_class, rules
  → _ocr_fallback_light_words()   # Повтор OCR для светлого/короткого текста (dilate, upscale, invert)
  → assign_words_to_region()      # Слово → один регион (overlap ≥ 0.3) или fallback
  → по каждому региону:
       words → words_to_lines()   # Y-полосы (2×font_size), X-острова (gap ≤ max(3×font, 2×cw))
       → classify_line()          # body / header / button / label
       → lines_to_blocks_with_headers()  # Header/body/button НЕ объединяются
       → clip block to region, container_bbox = region
  → ui_region → один GUIBlock (text_bbox = слова, container_bbox = region)
  → GUIBlock: bounding_box = text_bbox, container_bbox опционально
```

**Где текст объединяется:** в `words_to_lines` (одна строка: Y в пределах 2×font_size, gap ≤ max(3×font, 2×char_width)); в `lines_to_blocks_with_headers` (только body+body при малом вертикальном зазоре и overlap_x ≥ 0.6).

**Где разделяется:** при смене роли (header/button/body), при horizontal_rule, при большом вертикальном зазоре; границы региона не пересекаем.

**Где считается регионом:** `ui_region` → один блок-кнопка (text_bbox + container_bbox); `text_region` — контекст, внутри него несколько TextBlock (заголовок, абзац, кнопка).

---

## 2. Почему возникают баги

| Проблема | Причина в пайплайне |
|----------|----------------------|
| **Мелкий/бледный текст** | OCR на полной странице: один global Otsu плохо для secondary/muted; светлый текст на светлом фоне теряется. Fallback по bbox (dilate, invert, upscale) срабатывает только для уже найденных bbox. |
| **Primary / colored background** | Grayscale + CLAHE по яркости: цветной фон (синий) и белый текст дают малый контраст в L. Нет отдельной ветки «светлый текст на тёмном» для всей страницы. |
| **Кнопки (маленькие)** | ui_region отсекались по площади (MIN_AREA_RATIO, UI_MIN_AREA_RATIO); малые CTA не попадали в регионы. Сейчас пороги снижены; кнопка = region + text_bbox внутри. |
| **Кнопка целиком как текст** | Раньше не было разделения text_bbox / container_bbox; теперь кнопка → TextRegion (container), текст внутри → text_bbox. |
| **Дробление / вложенные блоки** | Слишком жёсткие Y-толеранции или сравнение по raw word.h вместо median font_size по строке; разные высоты слов давали разные полосы. Сейчас: Y по 2×font_size, font_size = median по band. |
| **Три карточки в один блок** | Раньше merge_blocks_inside_region склеивал всё в регионе. Убрано: используется lines_to_blocks_with_headers, header/body/button не сливаются; карточки разделены регионами. |
| **Строки одной высоты не объединяются** | X-gap или height_ratio могли резать. Сейчас: gap_x ≤ max(3×font_size_px, 2×char_width), height_ratio ≤ 2. |
| **Весь скриншот = текст** | Блок ≥ 0.8× экрана отбрасывается; слова вне регионов идут в fallback, но без merge_blocks в один гигантский блок. |

---

## 3. Два уровня детекции (соответствие коду)

| Уровень | В коде | Описание |
|---------|--------|----------|
| **TextRegion** | Region (ui_region / text_region), GUIBlock с container_bbox | UI-блок: карточка, кнопка, badge. container_bbox = регион. |
| **TextBox** | Word → Line → TextBlock, bounding_box (text_bbox) | Реальный текст: слова/строки. text_bbox = tight по словам. |

Кнопка: один GUIBlock с element_types=["button"], text_bbox = по словам внутри, container_bbox = region. Кнопка **не** считается одним куском текста.

---

## 4. Предобработка (реализовано и желательно)

**Сейчас:**
- Полная страница: grayscale → CLAHE → Otsu (`preprocess_full_page`).
- Fallback по crop: CLAHE, Otsu, invert if dark, dilate, upscale (`preprocess_crop`).

**Желательно (разрешённые подходы):**
- CLAHE по **L-каналу (LAB)** для полной страницы — лучше на цветном фоне (primary).
- Adaptive threshold (по блоку) вместо только global Otsu для страницы.
- Отдельная ветка «светлый текст на тёмном»: инверсия по яркости фона, затем OCR.

Цвет не используется для отсечения слов: только для роли (button/header/body) и для fallback-ветки.

---

## 5. Поиск регионов (реализовано)

- **LAB + k-means** по цвету → маски кластеров.
- **Morphology:** closing (7×3), opening (3×3) → связные области.
- **Contours** → фильтр по площади, прямоугольности, aspect (ui_region ≥ 1.2).
- **Типы:** text_region, ui_region, background.

MSER / Connected Components можно добавить как альтернативу или дополнение к k-means.

---

## 6. Объединение текста (правила)

- **Строка:** слова в одной Y-полосе (|y_center − y_ref| ≤ 2×font_size_px), height_ratio ≤ 2, gap_x ≤ max(3×font_size_px, 2×char_width). Цвет/фон совместимы; правило (horizontal/vertical) между словами → разрыв.
- **Абзац:** строки с role=body объединяются при vertical_gap ≤ 2.5×font_size, overlap_x ≥ 0.6, font_size ratio ≤ 1.5×, нет horizontal_rule. Header и button в отдельные блоки.
- **Карточки не объединяются:** слова привязаны к регионам; блоки строятся внутри региона; блоки обрезаются по региону.

---

## 7. Инварианты (напоминание)

- Region ≠ TextBlock. Region — контекст/ограничитель.
- Текст первичен: если OCR нашёл слово — оно попадает в блок или в fallback, регион не выкидывает слова (overlap < 0.3 → fallback).
- Два bbox: text_bbox (tight) + container_bbox (регион/кнопка), когда есть регион.
- Контраст не критерий включения/исключения; цвет используется только для роли и для выбора fallback-OCR.

---

## 8. Где в коде (для отладки)

| Что | Файл | Суть |
|-----|------|------|
| Регионы (TextRegion) | `region_prepass.py` | `cv_detect_regions`, `_get_regions_from_mask` — LAB k-means, морфология, контуры. |
| Назначение слов региону | `region_prepass.py` | `assign_words_to_region` — overlap ≥ 0.3, иначе fallback. |
| Объединение в строку | `line_builder.py` | `_group_into_local_y_bands` (Y по 2×font_size), `_split_band_into_x_islands` (gap ≤ max(3×font, 2×cw)). |
| Разделение по ролям | `block_builder.py` | `lines_to_blocks_with_headers` — header/body/button не сливаются; body+body при малом зазоре. |
| Кнопка = регион + текст | `flow_layout_service.py` | Ветка `region.region_type == "ui_region"`: text_bbox от слов, container_bbox = region. |
| Предобработка страницы | `ocr_preprocess.py` | `preprocess_full_page`: LAB L + CLAHE + Otsu. |
| Fallback светлый текст | `flow_layout_service.py` | `_ocr_fallback_light_words`: dilate, upscale, invert по crop. |
| Debug: регион vs текст | `layout_debug.py` | Регион — пунктир, text_bbox — сплошной, container_bbox — пунктир. |
