"""
Многоуровневое ориентирование: Macro → Meso → Micro → SchemaValidator.

Геометрия из изображения: полосы строк и колонки из визуальных кандидатов (и OCR baseline);
схема — фильтр и ограничитель, не источник bbox.
- Meso: только схема (роли, слоты, flow_end).
- Row bands: row_bands_from_anchors(card, schema, visual_candidates, ocr); фиксированные ratio только fallback.
- Grid columns: column_boundaries_from_visual(visual_candidates, card); schema.columns только fallback.
- Micro: лучший кандидат по многокритериальному скору; пустой слот допустим.
- Mixed: обрабатываем по строкам (field_row / button_row), не пропускаем.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CONFIDENCE_MULTILEVEL = 0.83


def _elements_inside_card(
    card_bbox: List[float],
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from src.infrastructure.atoms_v2.card_field_layout_inference import _elements_inside_card as _eic
    return _eic(card_bbox, atoms, raw_ocr_boxes)


def _ocr_in_row_bbox(ocr_boxes: List[Dict[str, Any]], row_bbox: List[float]) -> List[Dict[str, Any]]:
    if len(row_bbox) < 4:
        return []
    return [
        ob for ob in ocr_boxes
        if len((ob.get("bbox") or [])) >= 4
        and row_bbox[1] <= (ob["bbox"][1] + ob["bbox"][3]) / 2 <= row_bbox[3]
    ]


def _column_boundaries_for_grid_fallback(
    schema: Any,
    card_bbox: List[float],
) -> Optional[List[Tuple[float, float]]]:
    """Fallback: границы колонок из schema.columns только если визуальная кластеризация пуста."""
    if not schema.columns or len(card_bbox) < 4:
        return None
    card_x1 = card_bbox[0]
    card_w = card_bbox[2] - card_bbox[0]
    out: List[Tuple[float, float]] = []
    for col in schema.columns:
        cx = card_x1 + col.x_center_ratio * card_w
        half = (col.width_hint_ratio * card_w) / 2
        out.append((cx - half, cx + half))
    return out


def run_multilevel_field_inference(
    form_regions: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    image_path: Optional[str] = None,
    dark_theme: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Строго: поля только из visual_candidates в полосе; не более 1 bbox на слот; flow_end соблюдается.
    Mixed формы не обрабатываем — не добавляем полей, fallback сработает.
    """
    from src.infrastructure.atoms_v2.geometry_anchors import column_boundaries_from_visual, row_bands_from_anchors
    from src.infrastructure.atoms_v2.meso_layout_inference import infer_rows_and_columns
    from src.infrastructure.atoms_v2.micro_field_scanner import scan_fields
    from src.infrastructure.atoms_v2.schema_validator import validate_and_enforce
    from src.infrastructure.atoms_v2.visual_field_scanner import run_visual_field_scan

    log_lines: List[str] = []
    new_atoms: List[Dict[str, Any]] = []
    existing_ids = {a.get("id", "") for a in atoms if a.get("id")}

    if not form_regions:
        log_lines.append("multilevel_field: no form_regions, skip")
        return new_atoms, log_lines

    for card in form_regions:
        card_bbox = card.get("bbox", [0, 0, 0, 0])
        if len(card_bbox) < 4:
            continue
        card_id = card.get("id", "") or ("form_%s" % id(card))
        atoms_inside, ocr_inside = _elements_inside_card(card_bbox, atoms, raw_ocr_boxes)

        meso = infer_rows_and_columns(card_bbox, atoms_inside, ocr_inside)
        if not meso:
            log_lines.append("multilevel_field: card_id=%s no meso layout" % card_id)
            continue

        schema = meso.schema
        visual_bboxes, _ = run_visual_field_scan(image_path, [card], dark_theme=dark_theme)
        row_bands = row_bands_from_anchors(
            card_bbox, schema, visual_bboxes, ocr_inside, fallback_fixed_ratios=True,
        )
        col_boundaries = column_boundaries_from_visual(visual_bboxes, card_bbox)
        if not col_boundaries and schema.form_type in ("grid", "mixed") and schema.columns:
            col_boundaries = _column_boundaries_for_grid_fallback(schema, card_bbox)
        if schema.form_type == "vertical":
            col_boundaries = None

        detected_fields: List[Tuple[List[float], str]] = []
        block_h = card_bbox[3] - card_bbox[1]

        for i, row in enumerate(schema.rows):
            if row.role != "field_row":
                continue
            if schema.flow_end_row_index is not None and i > schema.flow_end_row_index:
                continue
            if i >= len(row_bands):
                continue
            band = row_bands[i]
            if band.y_max_ratio <= band.y_min_ratio:
                continue
            row_bbox = [
                card_bbox[0],
                card_bbox[1] + band.y_min_ratio * block_h,
                card_bbox[2],
                card_bbox[1] + band.y_max_ratio * block_h,
            ]
            slot_count = sum(1 for s in row.slots if s.has_input)
            if slot_count <= 0:
                continue
            ocr_in_row = _ocr_in_row_bbox(ocr_inside, row_bbox)
            use_cols = col_boundaries if schema.form_type in ("grid", "mixed") else None
            row_fields = scan_fields(
                row_bbox,
                ocr_in_row,
                visual_bboxes,
                slot_count=slot_count,
                column_boundaries=use_cols,
            )
            for bbox, ftype in row_fields:
                detected_fields.append((bbox, ftype))

        enforced = validate_and_enforce(schema, detected_fields, row_bands, card_bbox)
        for w in enforced.warnings:
            log_lines.append("multilevel_field: %s" % w)

        heights = [b[3] - b[1] for b, _ in enforced.fields if len(b) >= 4 and b[3] > b[1]]
        median_h = float(sorted(heights)[len(heights) // 2]) if heights else 40.0
        for idx, (bbox, ftype) in enumerate(enforced.fields):
            if len(bbox) < 4:
                continue
            atom_type = "textarea_candidate" if ftype == "textarea" else "input_candidate"
            aid = "multilevel_%s" % hashlib.sha256(str([round(x, 1) for x in bbox]).encode()).hexdigest()[:12]
            if aid in existing_ids:
                continue
            existing_ids.add(aid)
            conf = enforced.field_confidence[idx] if idx < len(enforced.field_confidence) else CONFIDENCE_MULTILEVEL
            new_atoms.append({
                "id": aid,
                "type": atom_type,
                "bbox": list(bbox),
                "confidence": conf,
                "source": "input_candidate_recovery",
                "recovery_source": "multilevel",
                "evidence": {"source": "multilevel", "macro_meso_micro": True},
            })
        log_lines.append(
            "multilevel_field: card_id=%s form_type=%s slots=%d fields=%d enforced_ok=%s"
            % (card_id, schema.form_type, enforced.slot_count, len(enforced.fields), enforced.ok)
        )

    log_lines.append("multilevel_field: total inferred=%d (source=multilevel)" % len(new_atoms))
    return new_atoms, log_lines
