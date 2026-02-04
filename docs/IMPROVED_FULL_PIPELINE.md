# Improved full-pipeline: Layout → OCR → GUI blocks

Трёхшаговый пайплайн для BPG: детекция layout → OCR внутри блоков → объединение в GUI-блоки.

## Шаги

### 1. Layout detection
- **Источник:** `run_ui_regions()` — DL (LayoutParser + Detectron2) с fallback на CV (контуры, иерархия).
- Обнаруживаются: кнопки, карточки, текстовые блоки, navbar, section, input-like, pill, badge.
- Мелкие элементы — за счёт второго прохода по ROI (CV) и масштабирования.
- Для карточек/секций определяется bbox, внутри которого затем ищутся элементы.
- **Лог:** `Layout: regions=N types={button: X, card: Y, ...}`.

### 2. OCR внутри блоков
- Для каждого региона: вызов `run_text_detect_roi(region)` (Paddle при включённом OCR, иначе пусто).
- По полученным bbox: `run_ocr_boxes_with_adaptive()` — Tesseract с Otsu, при пустом/коротком тексте — адаптивный порог и инвертированный вариант (мелкий/бледный текст).
- Для button/badge/pill/input без bbox: `run_ocr_roi()` (OCR по всему ROI).
- **Лог:** по каждому region_id — тип, число боксов.

### 3. Объединение в GUI-блоки
- Текстовые регионы (text_region, card, section, navbar): группировка в линии и параграфы (text_grouping); один блок на параграф/логическую группу.
- Кнопки/бейджи/инпуты: один блок на регион (bbox региона + объединённый текст).
- Блоки без текста сохраняются с `text=""`, `confidence=0` (иконки, неразмеченные кнопки).
- Типы на выходе: `button`, `card`, `text`, `input`.

## Вход / Выход

- **Вход:** изображение (файл).
- **Выход:** JSON с полем `gui_blocks` — список объектов:
  - `x`, `y`, `w`, `h` — координаты
  - `type` — `button` | `card` | `text` | `input`
  - `text` — распознанный текст (пустая строка, если OCR отключён или не нашёл)
  - `confidence` — уверенность OCR (0 при пустом тексте)

Дополнительно: `layout_log`, `ocr_log`, `merge_log`, `ocr_skipped`, `message`.

## Mac / Linux

- **Mac (DISABLE_PADDLEOCR=1):** текстовые bbox могут быть пустыми; блоки всё равно возвращаются с `text=""`.
- **Linux amd64 (PaddleOCR включён):** полная детекция текста и OCR внутри блоков.

## API

`POST /api/v1/debug/improved-full-pipeline`  
Тело: multipart, поле `image` (файл).  
Ответ: `ImprovedFullPipelineResponse` (gui_blocks, layout_log, ocr_log, merge_log, ocr_skipped, message).

## Кэш и офлайн

Используются те же переменные окружения, что и для основного пайплайна: `TORCH_HOME`, `TRANSFORMERS_CACHE`, `YOLO_CACHE_DIR`, `HF_HUB_OFFLINE`, `PADDLEOCR_OFFLINE` и т.д. Модели кэшируются; офлайн-режим поддерживается после первой загрузки.
