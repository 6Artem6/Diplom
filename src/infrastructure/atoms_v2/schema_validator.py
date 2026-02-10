"""
SchemaValidator — проверка и принудительное приведение схемы и обнаруженных полей.

- Запрещает: больше полей, чем слотов; поля вне field_row (по полосам).
- Понижает confidence при несовпадении.
- Возвращает обрезанный список полей и предупреждения.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.form_schema_models import FormSchema

logger = logging.getLogger(__name__)

CONFIDENCE_PENALTY_ON_MISMATCH = 0.12


@dataclass
class ValidationResult:
    """Результат валидации (только проверка, без изменения данных)."""
    ok: bool
    slot_count: int
    field_count: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class EnforceResult:
    """Результат валидации с приведением: обрезанные поля, пониженный confidence при несовпадении."""
    ok: bool
    slot_count: int
    field_count: int
    fields: List[Tuple[List[float], str]] = field(default_factory=list)
    field_confidence: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _slot_count(schema: FormSchema) -> int:
    n = 0
    for row in schema.rows:
        if row.role != "field_row":
            continue
        for slot in row.slots:
            if slot.has_input:
                n += 1
    return n


def _field_row_band_ranges(
    schema: FormSchema,
    row_bands: Any,
    card_bbox: List[float],
) -> List[Tuple[float, float]]:
    """Возвращает (y_min, y_max) в пикселях для каждой field_row (только до flow_end)."""
    if len(card_bbox) < 4:
        return []
    card_y1, card_h = card_bbox[1], card_bbox[3] - card_bbox[1]
    out: List[Tuple[float, float]] = []
    for i, row in enumerate(schema.rows):
        if row.role != "field_row":
            continue
        if schema.flow_end_row_index is not None and i > schema.flow_end_row_index:
            continue
        if row_bands and i < len(row_bands):
            band = row_bands[i]
            y_min = card_y1 + band.y_min_ratio * card_h
            y_max = card_y1 + band.y_max_ratio * card_h
            out.append((y_min, y_max))
        else:
            out.append((card_y1, card_bbox[3]))
    return out


def validate_schema_vs_candidates(
    schema: FormSchema,
    detected_fields: List[Any],
) -> ValidationResult:
    """
    Сравнивает количество слотов с has_input и количество обнаруженных полей.
    detected_fields: список bbox (list of list) или список (bbox, type).
    """
    slot_count = _slot_count(schema)
    field_count = len(detected_fields)
    warnings: List[str] = []
    if slot_count > 0 and field_count == 0:
        warnings.append("schema has %d slots with has_input but no detected fields (possible phantom slots)" % slot_count)
    if field_count > slot_count and slot_count > 0:
        warnings.append("detected_fields=%d > slot_count=%d (extra visual candidates or merged)" % (field_count, slot_count))
    if slot_count == 0 and field_count > 0:
        warnings.append("detected_fields=%d but schema has no input slots (schema may be incomplete)" % field_count)
    ok = len(warnings) == 0
    return ValidationResult(ok=ok, slot_count=slot_count, field_count=field_count, warnings=warnings)


def validate_and_enforce(
    schema: FormSchema,
    detected_fields: List[Tuple[List[float], str]],
    row_bands: Any,
    card_bbox: List[float],
    base_confidence: float = 0.83,
) -> EnforceResult:
    """
    Приводит список полей к схеме: не больше slot_count; отбрасывает поля вне полос field_row;
    понижает confidence при несовпадении (лишние/недостающие поля).
    """
    slot_count = _slot_count(schema)
    bands_y = _field_row_band_ranges(schema, row_bands, card_bbox)
    kept: List[Tuple[List[float], str]] = []
    confidence_list: List[float] = []
    warnings: List[str] = []

    for bbox, ftype in detected_fields:
        if len(bbox) < 4:
            continue
        cy = (bbox[1] + bbox[3]) / 2
        in_band = any(y1 <= cy <= y2 for y1, y2 in bands_y)
        if not in_band and bands_y:
            warnings.append("field bbox center y=%.0f outside field_row bands, dropped" % cy)
            continue
        kept.append((bbox, ftype))
        confidence_list.append(base_confidence)

    if len(kept) > slot_count and slot_count > 0:
        kept = kept[:slot_count]
        confidence_list = confidence_list[:slot_count]
        for i in range(len(confidence_list)):
            confidence_list[i] = max(0.2, confidence_list[i] - CONFIDENCE_PENALTY_ON_MISMATCH)
        warnings.append("trimmed to slot_count=%d, confidence lowered" % slot_count)
    elif len(kept) != slot_count and slot_count > 0:
        for i in range(len(confidence_list)):
            confidence_list[i] = max(0.2, confidence_list[i] - CONFIDENCE_PENALTY_ON_MISMATCH)
        if len(kept) < slot_count:
            warnings.append("fields=%d < slot_count=%d (empty slots), confidence lowered" % (len(kept), slot_count))

    ok = len(warnings) == 0
    return EnforceResult(
        ok=ok,
        slot_count=slot_count,
        field_count=len(kept),
        fields=kept,
        field_confidence=confidence_list,
        warnings=warnings,
    )
