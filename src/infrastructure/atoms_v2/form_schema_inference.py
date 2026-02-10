"""
FormSchemaInference — вывод логической схемы формы без bbox.

OCR используется только для роли строки (header, button_row, field_row, text) и слотов
(has_label, has_input, expected_input_type). OCR никогда не задаёт границы полей.

Вход: card (form_region), atoms внутри, OCR внутри.
Выход: FormSchema (form_type, columns, rows с ролями и слотами, flow_end_row_index).
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.form_schema_models import (
    ColumnHint,
    FormSchema,
    FormType,
    RowRole,
    RowSchema,
    SlotSchema,
)

logger = logging.getLogger(__name__)

# Реиспорт констант для ролей (копируем только те, что нужны для классификации, не для bbox)
from src.infrastructure.atoms_v2.card_field_layout_inference import (
    _elements_inside_card,
    _has_action_word_in_ocr,
    _is_header_row,
    _median_text_height,
    _row_detection,
    _row_is_form_end,
)
from src.infrastructure.atoms_v2.card_field_layout_inference import (
    LABEL_MAX_CHARS,
    PLACEHOLDER_HINTS,
)

# Текст-подсказки для textarea (только для expected_input_type, не для геометрии)
TEXTAREA_HINTS = frozenset({
    "comment", "message", "сообщение", "комментарий", "описание", "description",
    "bio", "notes", "заметки",
})


def _has_textarea_hint(ocr_list: List[Dict[str, Any]]) -> bool:
    """Есть ли в OCR подсказка на многострочное поле (только для роли слота)."""
    for ob in ocr_list:
        t = (ob.get("text") or "").strip().lower()
        if any(h in t for h in TEXTAREA_HINTS):
            return True
    return False


def _has_placeholder_hint(ocr_list: List[Dict[str, Any]]) -> bool:
    for ob in ocr_list:
        t = (ob.get("text") or "").strip().lower()
        if any(h in t for h in PLACEHOLDER_HINTS):
            return True
    return False


def _label_like_in_row(ocr_in_row: List[Dict[str, Any]]) -> bool:
    """В строке есть короткий текст, похожий на label (только для роли, не для bbox)."""
    for ob in ocr_in_row:
        text = (ob.get("text") or "").strip()
        if 0 < len(text) <= LABEL_MAX_CHARS:
            return True
    return False


def _infer_flow_end_row_index(
    rows: List[Dict[str, Any]],
    card_bbox: List[float],
) -> Optional[int]:
    """Индекс первой строки, которая задаёт конец формы (button/action). Ниже — полей нет."""
    if len(card_bbox) < 4 or not rows:
        return None
    sorted_rows = sorted(rows, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    for i, row in enumerate(sorted_rows):
        if _row_is_form_end(row, card_bbox):
            return i
    return None


def _row_role(
    row: Dict[str, Any],
    card_bbox: List[float],
    median_text_height: float,
) -> RowRole:
    """Роль строки только по OCR и наличию кнопки. Без bbox."""
    if _is_header_row(row, median_text_height):
        return "header"
    if row.get("has_button") or _has_action_word_in_ocr(row.get("ocr_inside") or []):
        return "button_row"
    # Узкая строка без label/placeholder — скорее текст
    ocr_in_row = row.get("ocr_inside") or []
    has_label = _label_like_in_row(ocr_in_row)
    has_placeholder = _has_placeholder_hint(ocr_in_row)
    if not has_label and not has_placeholder:
        card_w = card_bbox[2] - card_bbox[0] if len(card_bbox) >= 4 else 0
        row_w = row["bbox"][2] - row["bbox"][0]
        if card_w > 0 and row_w < card_w * 0.6:
            return "text"
    return "field_row"


def _slots_for_vertical_row(row: Dict[str, Any], card_bbox: List[float]) -> List[SlotSchema]:
    """
    Один слот на строку для вертикальной формы.
    has_label / has_input / expected_input_type — только из OCR, не из геометрии.
    """
    ocr_in_row = row.get("ocr_inside") or []
    has_label = _label_like_in_row(ocr_in_row)
    # field_row всегда имеет хотя бы один слот с has_input=True
    has_input = True
    expected = "textarea" if _has_textarea_hint(ocr_in_row) else "input"
    return [SlotSchema(column_index=0, has_label=has_label, has_input=has_input, expected_input_type=expected)]


def _infer_column_ratios_from_rows(
    rows: List[Dict[str, Any]],
    card_bbox: List[float],
    only_field_rows: bool,
) -> List[ColumnHint]:
    """
    По x-центрам элементов в строках выводим колонки как доли ширины card.
    Используется только для form_type=grid; геометрия card — для соотношений, не OCR.
    """
    if len(card_bbox) < 4:
        return []
    card_left = card_bbox[0]
    card_w = card_bbox[2] - card_bbox[0]
    if card_w <= 0:
        return []
    x_centers: List[float] = []
    for row in rows:
        if only_field_rows and _row_role(row, card_bbox, 20.0) != "field_row":
            continue
        for e in row.get("elements") or []:
            b = e.get("bbox", [0, 0, 0, 0])
            if len(b) >= 4:
                x_centers.append((b[0] + b[2]) / 2)
    if not x_centers:
        return []
    sorted_x = sorted(x_centers)
    tol = max(20, card_w * 0.25)
    clusters: List[List[float]] = []
    for cx in sorted_x:
        placed = False
        for c in clusters:
            if abs(cx - statistics.median(c)) <= tol:
                c.append(cx)
                placed = True
                break
        if not placed:
            clusters.append([cx])
    return [
        ColumnHint(
            x_center_ratio=(statistics.median(c) - card_left) / card_w,
            width_hint_ratio=1.0 / len(clusters),
        )
        for c in clusters
    ]


def _is_grid_like(rows: List[Dict[str, Any]], card_bbox: List[float]) -> bool:
    """Несколько колонок и повторяющаяся структура по строкам → grid."""
    columns = _infer_column_ratios_from_rows(rows, card_bbox, only_field_rows=True)
    if len(columns) < 2:
        return False
    field_row_count = sum(
        1 for r in rows
        if _row_role(r, card_bbox, _median_text_height([ob for row in rows for ob in row.get("ocr_inside") or []])) == "field_row"
    )
    return field_row_count >= 2


def infer_form_schema(
    card_bbox: List[float],
    atoms_inside: List[Dict[str, Any]],
    ocr_inside: List[Dict[str, Any]],
) -> Optional[FormSchema]:
    """
    Строит логическую схему формы. Не создаёт bbox.
    OCR только для ролей строк и слотов (has_label, has_input, expected_input_type).
    """
    if len(card_bbox) < 4:
        return None
    rows = _row_detection(atoms_inside, ocr_inside)
    if not rows:
        return None
    card_w = card_bbox[2] - card_bbox[0]
    card_h = card_bbox[3] - card_bbox[1]
    median_text_height = _median_text_height(ocr_inside)
    flow_end_idx = _infer_flow_end_row_index(rows, card_bbox)
    sorted_rows = sorted(rows, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    # Ограничиваем строки областью до flow_end (включительно эта строка — конец, поля выше неё)
    if flow_end_idx is not None:
        rows_before_end = sorted_rows[: flow_end_idx]
    else:
        rows_before_end = sorted_rows

    schema_rows: List[RowSchema] = []
    field_row_indices: List[int] = []
    for i, row in enumerate(sorted_rows):
        if flow_end_idx is not None and i > flow_end_idx:
            break
        role = _row_role(row, card_bbox, median_text_height)
        if role == "field_row":
            field_row_indices.append(len(schema_rows))
        slots: List[SlotSchema] = []
        if role == "field_row":
            slots = _slots_for_vertical_row(row, card_bbox)
        row_bbox = row.get("bbox", [0, 0, 0, 0])
        y_center_ratio = ((row_bbox[1] + row_bbox[3]) / 2 - card_bbox[1]) / card_h if card_h > 0 else None
        height_hint = (row_bbox[3] - row_bbox[1]) / card_h if card_h > 0 and len(row_bbox) >= 4 else None
        schema_rows.append(RowSchema(role=role, slots=slots, y_center_ratio=y_center_ratio, height_hint_ratio=height_hint))

    # form_type: grid если несколько колонок и повторяемость
    form_type: FormType = "vertical"
    columns_list: List[ColumnHint] = []
    if _is_grid_like(rows_before_end, card_bbox):
        form_type = "grid"
        columns_list = _infer_column_ratios_from_rows(rows_before_end, card_bbox, only_field_rows=True)
        # Для grid пересобираем слоты по колонкам: одна ячейка на колонку в каждой field_row
        schema_rows_grid: List[RowSchema] = []
        for sr in schema_rows:
            if sr.role == "field_row" and sr.slots and columns_list:
                slot = sr.slots[0]
                grid_slots = [
                    SlotSchema(column_index=j, has_label=slot.has_label, has_input=slot.has_input, expected_input_type=slot.expected_input_type)
                    for j in range(len(columns_list))
                ]
                schema_rows_grid.append(RowSchema(role=sr.role, slots=grid_slots, y_center_ratio=sr.y_center_ratio, height_hint_ratio=sr.height_hint_ratio))
            else:
                schema_rows_grid.append(sr)
        schema_rows = schema_rows_grid

    if not field_row_indices:
        return None

    # flow_end_row_index в терминах schema_rows
    flow_end_schema_index: Optional[int] = None
    if flow_end_idx is not None and flow_end_idx < len(schema_rows):
        flow_end_schema_index = flow_end_idx

    default_row_height = None
    if field_row_indices and card_h > 0:
        # подсказка высоты строки поля (доля от card)
        default_row_height = min(0.15, 1.0 / max(1, len(field_row_indices)))

    return FormSchema(
        form_type=form_type,
        columns=columns_list,
        rows=schema_rows,
        flow_end_row_index=flow_end_schema_index,
        default_row_height_ratio=default_row_height,
    )
