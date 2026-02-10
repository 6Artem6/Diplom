"""
MesoLayoutInference — мезоуровень: строки и колонки внутри блока (логическая схема без bbox полей).

Выход: schema (rows, slots, роли, flow_end) + row_bands с фиксированными высотами.
Полосы строк НЕ зависят от OCR/placeholder/atoms — только от числа field_row и типа слота (input/textarea).
INPUT_HEIGHT_RATIO, TEXTAREA_HEIGHT_RATIO, ROW_GAP_RATIO задают геометрию полос детерминированно.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.infrastructure.atoms_v2.form_schema_models import FormSchema

logger = logging.getLogger(__name__)

# Фиксированные доли высоты блока: полосы не зависят от OCR
INPUT_HEIGHT_RATIO = 0.065
TEXTAREA_HEIGHT_RATIO = 0.12
ROW_GAP_RATIO = 0.02


@dataclass
class RowBand:
    """Границы полосы строки в долях высоты блока (0..1). Детерминированы по типу слота."""
    y_min_ratio: float
    y_max_ratio: float


@dataclass
class MesoLayoutResult:
    """Результат мезоуровня: схема + полосы строк (без bbox полей)."""
    schema: FormSchema
    row_bands: List[RowBand]  # по одному на каждую строку schema.rows


def infer_rows_and_columns(
    block_bbox: List[float],
    atoms: List[Dict[str, Any]],
    ocr_inside: List[Dict[str, Any]],
) -> Optional[MesoLayoutResult]:
    """
    Строит логическую структуру (строки, слоты, роли) и полосы строк.
    OCR только для ролей строки/слота. row_bands — только из фиксированных констант и порядка field_row.
    """
    if len(block_bbox) < 4:
        return None
    try:
        from src.infrastructure.atoms_v2.form_schema_inference import infer_form_schema
    except ImportError:
        return None
    schema = infer_form_schema(block_bbox, atoms, ocr_inside)
    if not schema or not schema.rows:
        return None

    # Полосы только из фиксированных соотношений; не используем y_center_ratio / height_hint_ratio из схемы
    n = len(schema.rows)
    row_bands: List[RowBand] = []
    y_cur = 0.0
    for i in range(n):
        row = schema.rows[i]
        if row.role != "field_row":
            row_bands.append(RowBand(0.0, 0.0))
            continue
        if schema.flow_end_row_index is not None and i > schema.flow_end_row_index:
            row_bands.append(RowBand(0.0, 0.0))
            continue
        is_textarea = any(s.expected_input_type == "textarea" for s in row.slots)
        h_ratio = TEXTAREA_HEIGHT_RATIO if is_textarea else INPUT_HEIGHT_RATIO
        y_max = min(1.0, y_cur + h_ratio)
        row_bands.append(RowBand(y_min_ratio=y_cur, y_max_ratio=y_max))
        y_cur = y_max + ROW_GAP_RATIO
        if y_cur >= 1.0:
            break
    while len(row_bands) < n:
        row_bands.append(RowBand(0.0, 0.0))
    return MesoLayoutResult(schema=schema, row_bands=row_bands)
