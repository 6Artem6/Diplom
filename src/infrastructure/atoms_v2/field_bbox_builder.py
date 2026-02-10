"""
FieldBBoxBuilder — детерминированное построение bbox полей из схемы формы и геометрии card.

Инварианты (нарушать нельзя):
  - 1 SlotSchema с has_input=True → ровно 1 bbox. Итерация только по schema.rows[].slots[].
  - Высота поля только по типу слота (input/textarea) и типу формы. Не по OCR, не по контенту.
  - Ни один bbox для row index > flow_end_row_index.
  - Нет пост-проходов, уточнений, bbox «вокруг текста». Builder не использует atoms/OCR.

«Builder — тупой, но честный».
"""

from __future__ import annotations

import logging
from typing import List, Literal, Tuple

from src.infrastructure.atoms_v2.form_schema_models import (
    FormSchema,
    RowSchema,
    SlotSchema,
)

logger = logging.getLogger(__name__)

# Отступы от границ card (доли). Не из OCR.
CARD_PADDING_X_RATIO = 0.06
CARD_PADDING_Y_RATIO = 0.04
# Фиксированная высота полосы строки — только по типу. Не из схемы, не из OCR.
INPUT_HEIGHT_RATIO = 0.065   # доля высоты card на один input
TEXTAREA_HEIGHT_RATIO = 0.12 # доля высоты card на один textarea
# Зазор между полосами (доля высоты card)
ROW_GAP_RATIO = 0.02


def _allowed_field_rows(
    schema: FormSchema,
) -> List[Tuple[int, RowSchema]]:
    """
    Строки с role=field_row и индексом <= flow_end_row_index.
    Инвариант №3: после flow_end_row_index полей не создаём.
    """
    result: List[Tuple[int, RowSchema]] = []
    for i, row in enumerate(schema.rows):
        if row.role != "field_row":
            continue
        if schema.flow_end_row_index is not None and i > schema.flow_end_row_index:
            continue
        result.append((i, row))
    return result


def _slots_with_input(
    schema: FormSchema,
) -> List[Tuple[int, RowSchema, SlotSchema]]:
    """
    Плоский список (row_index, row, slot) для всех слотов с has_input=True
    в разрешённых строках (index <= flow_end_row_index).
    Инвариант №1: один такой слот → ровно один bbox.
    """
    out: List[Tuple[int, RowSchema, SlotSchema]] = []
    for i, row in _allowed_field_rows(schema):
        for slot in row.slots:
            if slot.has_input:
                out.append((i, row, slot))
    return out


def _build_bboxes_vertical(
    schema: FormSchema,
    card_bbox: List[float],
) -> List[Tuple[List[float], Literal["input", "textarea"]]]:
    """
    Вертикальная форма: фиксированная сетка полос.
    Один bbox на слот с has_input. Высота полосы только INPUT_HEIGHT_RATIO / TEXTAREA_HEIGHT_RATIO.
    Y не зависит от OCR/atoms — только порядок слотов в schema.
    """
    if len(card_bbox) < 4:
        return []
    x1_card, y1_card, x2_card, y2_card = card_bbox[0], card_bbox[1], card_bbox[2], card_bbox[3]
    card_w = x2_card - x1_card
    card_h = y2_card - y1_card
    if card_w <= 0 or card_h <= 0:
        return []

    pad_x = card_w * CARD_PADDING_X_RATIO
    pad_y = card_h * CARD_PADDING_Y_RATIO
    content_left = x1_card + pad_x
    content_right = x2_card - pad_x
    content_top = y1_card + pad_y
    content_w = content_right - content_left

    slots_list = _slots_with_input(schema)
    if not slots_list:
        return []

    # Фиксированная высота полосы по типу слота. Инвариант №2.
    gap_px = card_h * ROW_GAP_RATIO
    result: List[Tuple[List[float], Literal["input", "textarea"]]] = []
    y_cur = content_top

    for _, _row, slot in slots_list:
        if slot.expected_input_type == "textarea":
            h = card_h * TEXTAREA_HEIGHT_RATIO
        else:
            h = card_h * INPUT_HEIGHT_RATIO
        y1 = y_cur
        y2 = y_cur + h
        y_cur = y2 + gap_px
        expected: Literal["input", "textarea"] = slot.expected_input_type
        result.append(([content_left, y1, content_right, y2], expected))
    return result


def _build_bboxes_grid(
    schema: FormSchema,
    card_bbox: List[float],
) -> List[Tuple[List[float], Literal["input", "textarea"]]]:
    """
    Grid: только cell(row_i, col_j) → один bbox на слот с has_input.
    Фиксированная высота строки по типу. Никакой row-based fallback, никакого OCR.
    """
    if len(card_bbox) < 4:
        return []
    if not schema.columns:
        return []

    x1_card, y1_card, x2_card, y2_card = card_bbox[0], card_bbox[1], card_bbox[2], card_bbox[3]
    card_w = x2_card - x1_card
    card_h = y2_card - y1_card
    if card_w <= 0 or card_h <= 0:
        return []

    pad_x = card_w * CARD_PADDING_X_RATIO
    pad_y = card_h * CARD_PADDING_Y_RATIO
    content_left = x1_card + pad_x
    content_right = x2_card - pad_x
    content_top = y1_card + pad_y
    content_w = content_right - content_left

    field_row_list = _allowed_field_rows(schema)
    if not field_row_list:
        return []

    n_cols = len(schema.columns)
    col_x1: List[float] = []
    col_x2: List[float] = []
    for j, col in enumerate(schema.columns):
        half = (col.width_hint_ratio * card_w) / 2
        cx = x1_card + col.x_center_ratio * card_w
        col_x1.append(max(content_left, cx - half))
        col_x2.append(min(content_right, cx + half))
    col_width = content_w / n_cols
    for j in range(n_cols):
        if col_x2[j] <= col_x1[j]:
            col_x2[j] = col_x1[j] + col_width

    # Фиксированная высота строки по типу первой ячейки в строке. Инвариант №2.
    row_heights: List[float] = []
    for _i, row in field_row_list:
        is_textarea = any(s.expected_input_type == "textarea" for s in row.slots)
        row_heights.append(card_h * TEXTAREA_HEIGHT_RATIO if is_textarea else card_h * INPUT_HEIGHT_RATIO)

    result: List[Tuple[List[float], Literal["input", "textarea"]]] = []
    y_cur = content_top
    for row_idx, (_, row) in enumerate(field_row_list):
        h = row_heights[row_idx]
        y1 = y_cur
        y2 = y_cur + h
        for slot in row.slots:
            if not slot.has_input:
                continue
            j = slot.column_index
            if j < 0 or j >= n_cols:
                continue
            x1 = col_x1[j]
            x2 = col_x2[j]
            expected: Literal["input", "textarea"] = slot.expected_input_type
            result.append(([x1, y1, x2, y2], expected))
        y_cur = y2 + card_h * ROW_GAP_RATIO
    return result


def build_field_bboxes(
    schema: FormSchema,
    card_bbox: List[float],
) -> List[Tuple[List[float], Literal["input", "textarea"]]]:
    """
    Строит ровно один bbox на каждый слот с has_input=True в разрешённых строках.
    Не использует OCR, atoms, y_center_ratio, height_hint_ratio, default_row_height_ratio.
    """
    if not schema.rows:
        return []
    if schema.form_type == "grid" and schema.columns:
        return _build_bboxes_grid(schema, card_bbox)
    return _build_bboxes_vertical(schema, card_bbox)


# --- Интеграция: schema → bbox → atoms (для пайплайна) ---
CONFIDENCE_FORM_SCHEMA = 0.85


def run_form_schema_field_inference(
    form_regions: List[dict],
    atoms: List[dict],
    raw_ocr_boxes: List[dict],
) -> Tuple[List[dict], List[str]]:
    """
    Form → Schema → Fields: схема (FormSchemaInference), затем builder.
    Builder не получает atoms/OCR — только schema и card_bbox.
    """
    import hashlib

    from src.infrastructure.atoms_v2.card_field_layout_inference import _elements_inside_card
    from src.infrastructure.atoms_v2.form_schema_inference import infer_form_schema

    log_lines: List[str] = []
    new_atoms: List[dict] = []
    existing_ids = {a.get("id", "") for a in atoms if a.get("id")}

    for card in form_regions:
        card_id = card.get("id", "") or ("form_region_%s" % id(card))
        card_bbox = card.get("bbox", [0, 0, 0, 0])
        if len(card_bbox) < 4:
            continue
        atoms_inside, ocr_inside = _elements_inside_card(card_bbox, atoms, raw_ocr_boxes)
        schema = infer_form_schema(card_bbox, atoms_inside, ocr_inside)
        if not schema or not schema.rows:
            log_lines.append("form_schema_field: card_id=%s no schema or rows, skip" % card_id)
            continue
        slots_count = len(_slots_with_input(schema))
        if slots_count == 0:
            log_lines.append("form_schema_field: card_id=%s no slots with has_input, skip" % card_id)
            continue
        bboxes_with_type = build_field_bboxes(schema, card_bbox)
        # Инвариант: len(bboxes_with_type) == slots_count
        if len(bboxes_with_type) != slots_count:
            log_lines.append(
                "form_schema_field: card_id=%s builder count mismatch slots=%d bboxes=%d"
                % (card_id, slots_count, len(bboxes_with_type))
            )
        for bbox, expected_type in bboxes_with_type:
            if len(bbox) < 4:
                continue
            atom_type = "textarea_candidate" if expected_type == "textarea" else "input_candidate"
            aid = "form_schema_%s" % hashlib.sha256(str([round(x, 1) for x in bbox]).encode()).hexdigest()[:12]
            if aid in existing_ids:
                continue
            existing_ids.add(aid)
            new_atoms.append({
                "id": aid,
                "type": atom_type,
                "bbox": list(bbox),
                "confidence": CONFIDENCE_FORM_SCHEMA,
                "source": "input_candidate_recovery",
                "recovery_source": "form_schema_builder",
                "evidence": {"source": "form_schema_builder"},
            })
        log_lines.append(
            "form_schema_field: card_id=%s form_type=%s slots=%d bboxes=%d"
            % (card_id, schema.form_type, slots_count, len(bboxes_with_type))
        )
    log_lines.append("form_schema_field: total inferred=%d (source=form_schema_builder)" % len(new_atoms))
    return new_atoms, log_lines
