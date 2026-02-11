"""
Патч инвариантов для пайплайна rows → slots → slot_assignments.

Архитектурный инвариант: CV — единственный источник геометрии строк (row.y_min, row.y_max,
row.x_min, row.x_max). OCR не может изменять границы строк; только row_type, label_bbox,
helper_bbox, роли слотов. Изменения только внутри логики enforce_* и проверок.
Не меняет контракты стадий и структуру данных.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.infrastructure.atoms_v2.experimental_v2.models import FormRow

logger = logging.getLogger(__name__)

# Порог вертикального зазора: текст сразу над input считается label
LABEL_ABOVE_MAX_GAP_PX = 50
# Первая строка: если высота меньше этой доли от медианы input → не input
HEADER_MAX_HEIGHT_RATIO = 0.85
# Минимальная высота input-подобной строки (ниже — не input)
MIN_INPUT_HEIGHT_PX = 28
# Центрирование: доля ширины контейнера
CENTER_TOLERANCE_RATIO = 0.35
# Textarea: не добавлять padding вниз
TEXTAREA_BOTTOM_PADDING_PX = 0


def _ocr_in_row(ob: Dict[str, Any], row: FormRow) -> bool:
    b = ob.get("bbox") or []
    if len(b) < 4:
        return False
    cy = (b[1] + b[3]) / 2
    return row.y_min <= cy <= row.y_max and row.x_min <= (b[0] + b[2]) / 2 <= row.x_max


def _row_has_box(row: FormRow) -> bool:
    """Строка имеет визуальный box (input_bbox от детекции, не весь row)."""
    ib = getattr(row, "input_bbox", None)
    if not ib or len(ib) < 4:
        return False
    # input_bbox не равен всей строке (допуск по высоте)
    row_h = row.y_max - row.y_min
    ib_h = ib[3] - ib[1]
    if row_h < 1:
        return True
    # Если input занимает почти всю строку по Y и это не textarea — считаем что есть box
    return ib_h >= min(MIN_INPUT_HEIGHT_PX, row_h * 0.5)


def _row_has_placeholder_style_ocr(ocr_in_row: List[Dict[str, Any]], input_bbox: List[float]) -> bool:
    """Есть ли OCR внутри/рядом с input (placeholder-style)."""
    for ob in ocr_in_row:
        b = ob.get("bbox") or []
        if len(b) < 4 or len(input_bbox) < 4:
            continue
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2
        ix1, iy1, ix2, iy2 = input_bbox[0], input_bbox[1], input_bbox[2], input_bbox[3]
        if ix1 <= cx <= ix2 and iy1 <= cy <= iy2:
            return True
    return False


def _is_centered(b: List[float], container_bbox: List[float]) -> bool:
    if len(b) < 4 or len(container_bbox) < 4:
        return False
    cx = (b[0] + b[2]) / 2
    c_w = container_bbox[2] - container_bbox[0]
    cont_cx = (container_bbox[0] + container_bbox[2]) / 2
    return c_w > 0 and abs(cx - cont_cx) / c_w <= CENTER_TOLERANCE_RATIO


def _is_large_text(ob: Dict[str, Any], median_font: float) -> bool:
    b = ob.get("bbox") or []
    if len(b) < 4:
        return False
    return (b[3] - b[1]) >= median_font * 1.2


def enforce_row_boundary_invariant(
    rows: List[FormRow],
    layout_ocr: List[Dict[str, Any]],
    container_bbox: List[float],
) -> None:
    """
    CV — единственный источник границ строк. OCR не меняет row.y_min/row.y_max.
    При нарушении «row покрывает все OCR в строке» только диагностика (лог).
    """
    if len(container_bbox) < 4 or not rows or not layout_ocr:
        return
    for r in rows:
        ocr_here = [ob for ob in layout_ocr if _ocr_in_row(ob, r)]
        if not ocr_here:
            continue
        tops = [ob["bbox"][1] for ob in ocr_here if len(ob.get("bbox", [])) >= 4]
        bottoms = [ob["bbox"][3] for ob in ocr_here if len(ob.get("bbox", [])) >= 4]
        if not tops or not bottoms:
            continue
        min_top = min(tops)
        max_bottom = max(bottoms)
        if r.y_min > min_top:
            logger.warning(
                "row_boundary_invariant: row %d y_min=%.0f > min(ocr.top)=%.0f (CV geometry not changed)",
                r.row_index, r.y_min, min_top,
            )
        if r.y_max < max_bottom:
            logger.warning(
                "row_boundary_invariant: row %d y_max=%.0f < max(ocr.bottom)=%.0f (CV geometry not changed)",
                r.row_index, r.y_max, max_bottom,
            )


def enforce_textarea_bottom(
    rows: List[FormRow],
    layout_ocr: List[Dict[str, Any]],
) -> None:
    """
    TEXTAREA.bottom задаётся только по input_bbox или по горизонтальной линии (CV).
    OCR не меняет row.y_max. При расхождении — только диагностика (лог).
    """
    for r in rows:
        if r.row_type != "TEXTAREA":
            continue
        ocr_here = [ob for ob in layout_ocr if _ocr_in_row(ob, r)]
        if not ocr_here:
            continue
        bottoms = [ob["bbox"][3] for ob in ocr_here if len(ob.get("bbox", [])) >= 4]
        if not bottoms:
            continue
        ocr_bottom = max(bottoms)
        if r.y_max < ocr_bottom:
            logger.warning(
                "textarea_bottom_invariant: row %d (TEXTAREA) y_max=%.0f < max(ocr.bottom)=%.0f (CV geometry not changed)",
                r.row_index, r.y_max, ocr_bottom,
            )


def enforce_header_protection(
    rows: List[FormRow],
    layout_ocr: List[Dict[str, Any]],
    container_bbox: List[float],
    baseline: Dict[str, Any],
) -> None:
    """
    Инвариант 5: первая строка не может быть input, если высота меньше средней input,
    нет box, текст центрирован или крупнее. Такая строка → HEADER или skipped (TEXT).
    """
    if len(rows) == 0 or len(container_bbox) < 4:
        return
    r0 = rows[0]
    if r0.row_type not in ("FIELD", "FIELD_HORIZONTAL", "FIELD_VERTICAL", "FIELD_INPUT_ONLY"):
        return
    median_font = float(baseline.get("median_font_height", 20.0))
    row_h = r0.y_max - r0.y_min
    # Медиана высоты input по остальным строкам
    other_heights = [
        (r.y_max - r.y_min) for r in rows[1:]
        if getattr(r, "input_bbox", None) and len(r.input_bbox) >= 4
    ]
    median_input_h = (sum(other_heights) / len(other_heights)) if other_heights else 50.0
    if row_h < median_input_h * HEADER_MAX_HEIGHT_RATIO and not _row_has_box(r0):
        ocr_in_row = [ob for ob in layout_ocr if _ocr_in_row(ob, r0)]
        any_centered = any(_is_centered(ob.get("bbox", []), container_bbox) for ob in ocr_in_row if ob.get("bbox"))
        any_large = any(_is_large_text(ob, median_font) for ob in ocr_in_row)
        if any_centered or any_large or (ocr_in_row and not _row_has_box(r0)):
            r0.row_type = "HEADER"


def enforce_input_classification_invariant(
    rows: List[FormRow],
    layout_ocr: List[Dict[str, Any]],
    container_bbox: List[float],
    baseline: Dict[str, Any],
) -> None:
    """
    Инвариант 2: row не может быть input, если внутри нет признаков поля (box, placeholder).
    Заголовок ≠ input. Кнопка ≠ input. Textarea ≠ обычный input.
    """
    if len(container_bbox) < 4:
        return
    median_font = float(baseline.get("median_font_height", 20.0))
    for r in rows:
        if r.row_type not in ("FIELD", "FIELD_HORIZONTAL", "FIELD_VERTICAL", "FIELD_INPUT_ONLY"):
            continue
        ocr_here = [ob for ob in layout_ocr if _ocr_in_row(ob, r)]
        input_bbox = getattr(r, "input_bbox", None) or [r.x_min, r.y_min, r.x_max, r.y_max]
        has_box = _row_has_box(r)
        has_placeholder_style = _row_has_placeholder_style_ocr(ocr_here, input_bbox) if len(input_bbox) >= 4 else False
        row_h = r.y_max - r.y_min
        # Нет box и нет placeholder-OCR → не input (header/text)
        if not has_box and not has_placeholder_style:
            if ocr_here:
                any_centered = any(_is_centered(ob.get("bbox", []), container_bbox) for ob in ocr_here if ob.get("bbox"))
                if any_centered or r.row_index == 0:
                    r.row_type = "HEADER"
                else:
                    r.row_type = "TEXT"
            continue
        # Высота >> обычного input при наличии box → textarea (уже может быть помечена)
        if has_box and row_h >= max(80, median_font * 3):
            if r.row_type == "FIELD_INPUT_ONLY" or r.row_type == "FIELD":
                r.row_type = "TEXTAREA"


def enforce_label_isolation(
    rows: List[FormRow],
    layout_ocr: List[Dict[str, Any]],
) -> None:
    """
    Инвариант 3: label не может быть классифицирован как input.
    Если row: нет box, есть текст, непосредственно над input-row → это label (TEXT или оставляем для FIELD_VERTICAL).
    """
    for i, r in enumerate(rows):
        if r.row_type not in ("FIELD", "FIELD_HORIZONTAL", "FIELD_VERTICAL", "FIELD_INPUT_ONLY"):
            continue
        if _row_has_box(r):
            continue
        ocr_here = [ob for ob in layout_ocr if _ocr_in_row(ob, r)]
        if not ocr_here:
            continue
        # Строка только с текстом (нет input_bbox от визуала или input_bbox = вся строка без реального box)
        if i + 1 < len(rows):
            next_row = rows[i + 1]
            if next_row.row_type in ("FIELD", "FIELD_HORIZONTAL", "FIELD_VERTICAL", "FIELD_INPUT_ONLY") and _row_has_box(next_row):
                gap = next_row.y_min - r.y_max
                if gap <= LABEL_ABOVE_MAX_GAP_PX:
                    r.row_type = "TEXT"


def enforce_slot_label_from_above(
    rows: List[FormRow],
    layout_ocr: List[Dict[str, Any]],
) -> None:
    """
    Инвариант 7 (slot assignment correction): если у строки input без label и есть текст непосредственно сверху
    в пределах vertical_gap < threshold — этот текст считается label, slot пересобирается (задаём label_bbox и row_type).
    """
    for i in range(1, len(rows)):
        r = rows[i]
        if r.row_type != "FIELD_INPUT_ONLY":
            continue
        if getattr(r, "label_bbox", None) and len(r.label_bbox) >= 4:
            continue
        prev = rows[i - 1]
        if prev.row_type not in ("HEADER", "TEXT"):
            continue
        gap = r.y_min - prev.y_max
        if gap > LABEL_ABOVE_MAX_GAP_PX:
            continue
        ocr_prev = [ob for ob in layout_ocr if _ocr_in_row(ob, prev)]
        if not ocr_prev:
            continue
        # Берём нижний OCR в предыдущей строке (ближайший к текущей)
        best = max(ocr_prev, key=lambda o: o["bbox"][3] if len(o.get("bbox", [])) >= 4 else 0)
        if len(best.get("bbox", [])) >= 4:
            r.label_bbox = list(best["bbox"])
            r.vertical_split_y = r.label_bbox[3]
            r.row_type = "FIELD_VERTICAL"
            r.column_count = 1
            r.vertical_separators = None


def enforce_form_row_invariants(
    rows: List[FormRow],
    layout_ocr: List[Dict[str, Any]],
    container_bbox: List[float],
    baseline: Dict[str, Any],
) -> None:
    """
    Применяет инварианты к списку строк (in-place). Вызывать после _post_process_rows.
    Границы строк (y_min, y_max, x_min, x_max) не изменяются по OCR — только диагностика.
    OCR влияет только на: row_type, label_bbox, helper_bbox, роли слотов.
    """
    if not rows:
        return
    enforce_row_boundary_invariant(rows, layout_ocr, container_bbox)
    enforce_textarea_bottom(rows, layout_ocr)
    enforce_header_protection(rows, layout_ocr, container_bbox, baseline)
    enforce_input_classification_invariant(rows, layout_ocr, container_bbox, baseline)
    enforce_label_isolation(rows, layout_ocr)
    enforce_slot_label_from_above(rows, layout_ocr)


def correct_slot_assignment_bboxes(
    assignments: List[Any],
    skeleton: Any,
    raw_ocr_boxes: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Инвариант 4 (slot): input bbox не пересекает label; при необходимости расширение по placeholder.
    Меняет только assignment.bbox; row.y_min/row.y_max не изменяются.
    """
    if not getattr(skeleton, "rows", None) or not assignments or not raw_ocr_boxes:
        return
    rows = skeleton.rows
    for a in assignments:
        if not getattr(a, "bbox", None) or a.bbox is None or len(a.bbox) < 4:
            continue
        slot = getattr(a, "slot", None)
        if not slot or slot.row_index >= len(rows):
            continue
        row = rows[slot.row_index]
        bbox = list(a.bbox)
        changed = False
        label_bbox = getattr(row, "label_bbox", None)
        if label_bbox and len(label_bbox) >= 4:
            new_top = label_bbox[3] + 2
            if bbox[1] < new_top and new_top < bbox[3]:
                bbox[1] = new_top
                changed = True
        slot_hint = getattr(slot, "expected_bbox_hint", None) or []
        if len(slot_hint) >= 4:
            for ob in raw_ocr_boxes:
                b = ob.get("bbox") or []
                if len(b) < 4:
                    continue
                if b[0] >= slot_hint[2] or b[2] <= slot_hint[0]:
                    continue
                if b[1] >= slot_hint[3] or b[3] <= slot_hint[1]:
                    continue
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
                    continue
                if (bbox[0] <= cx <= bbox[2]) or (min(b[0], bbox[0]) < max(b[2], bbox[2])):
                    if b[1] < bbox[1]:
                        bbox[1] = min(bbox[1], b[1])
                        changed = True
                    if b[3] > bbox[3]:
                        bbox[3] = max(bbox[3], b[3])
                        changed = True
                    if b[0] < bbox[0]:
                        bbox[0] = min(bbox[0], b[0])
                        changed = True
                    if b[2] > bbox[2]:
                        bbox[2] = max(bbox[2], b[2])
                        changed = True
        if changed:
            a.bbox = bbox


def analyze_ocr_orientation(
    row: FormRow,
    ocr_in_row: List[Dict[str, Any]],
) -> None:
    """
    Определяет ориентацию OCR относительно input: VERTICAL (label сверху), HORIZONTAL (label слева),
    или placeholder (текст внутри input). Результат в row.metadata["ocr_orientation"].
    Геометрию не меняет, только уточняет семантику.
    """
    orientation = "unknown"
    input_bbox = getattr(row, "input_bbox", None)
    if not input_bbox or len(input_bbox) < 4:
        if getattr(row, "metadata", None) is not None:
            row.metadata["ocr_orientation"] = orientation
        return
    ix1, iy1, ix2, iy2 = input_bbox[0], input_bbox[1], input_bbox[2], input_bbox[3]
    input_cx = (ix1 + ix2) / 2
    input_cy = (iy1 + iy2) / 2
    input_w = ix2 - ix1
    input_h = iy2 - iy1
    x_overlap_ratio = lambda a, b: (min(a[2], b[2]) - max(a[0], b[0])) / (b[2] - b[0]) if (b[2] > b[0]) else 0.0

    for ob in ocr_in_row:
        b = ob.get("bbox") or []
        if len(b) < 4:
            continue
        ob_bottom = b[3]
        ob_top = b[1]
        ob_left = b[0]
        ob_right = b[2]
        ob_cy = (b[1] + b[3]) / 2
        # Placeholder: OCR внутри input_bbox
        if b[0] >= ix1 and b[2] <= ix2 and b[1] >= iy1 and b[3] <= iy2:
            orientation = "placeholder"
            break
        # Vertical label: OCR полностью выше input, пересечение по X > 40%
        overlap = x_overlap_ratio(b, input_bbox)
        if ob_bottom <= iy1 and overlap >= 0.4:
            orientation = "vertical"
            break
        # Horizontal label: OCR слева от input, центр Y в пределах input
        if ob_right <= ix1 + (input_w * 0.2) and iy1 <= ob_cy <= iy2:
            orientation = "horizontal"
            break
    if not hasattr(row, "metadata"):
        row.metadata = {}
    row.metadata["ocr_orientation"] = orientation


def enforce_field_has_input_bbox(rows: List[FormRow]) -> None:
    """
    Debug-инвариант: row_type ∈ {FIELD_*, TEXTAREA} обязан иметь input_bbox.
    Если input_bbox == None — логировать invariant_violation и понижать row_type до TEXT.
    Вызывать после classify_rows, перед _remove_orphan_field_rows.
    """
    field_types = {"FIELD", "FIELD_HORIZONTAL", "FIELD_VERTICAL", "FIELD_INPUT_ONLY", "TEXTAREA"}
    for r in rows:
        if r.row_type not in field_types:
            continue
        has_input = (
            (getattr(r, "input_bbox", None) and len(r.input_bbox) >= 4)
            or (getattr(r, "input_bboxes", None) and len(r.input_bboxes) > 0)
        )
        if not has_input:
            logger.warning(
                "invariant_violation: row %d row_type=%s but input_bbox is None → downgrade to TEXT",
                r.row_index, r.row_type,
            )
            r.row_type = "TEXT"
            r.input_bbox = None
            if getattr(r, "input_bboxes", None):
                r.input_bboxes = None


def log_assignment_outside_row_invariant(
    assignments: List[Any],
    skeleton: Any,
) -> None:
    """
    Строгий debug: если assignment.bbox выходит за границы своей строки (row) —
    логировать invariant violation. Row не изменяется.
    Вызывать после всех стадий (в т.ч. после correct_slot_assignment_bboxes).
    """
    if not getattr(skeleton, "rows", None) or not assignments:
        return
    rows = skeleton.rows
    for a in assignments:
        if not getattr(a, "bbox", None) or a.bbox is None or len(a.bbox) < 4:
            continue
        slot = getattr(a, "slot", None)
        if not slot or slot.row_index >= len(rows):
            continue
        row = rows[slot.row_index]
        bbox = a.bbox
        if bbox[0] < row.x_min or bbox[2] > row.x_max or bbox[1] < row.y_min or bbox[3] > row.y_max:
            logger.warning(
                "assignment_outside_row: slot row_index=%d assignment bbox [%.0f,%.0f,%.0f,%.0f] "
                "outside row [x=%.0f..%.0f y=%.0f..%.0f] (row not changed)",
                slot.row_index, bbox[0], bbox[1], bbox[2], bbox[3],
                row.x_min, row.x_max, row.y_min, row.y_max,
            )
