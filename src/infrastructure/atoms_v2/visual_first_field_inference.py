"""
Visual First → Schema as Filter → Normalize.

Источник bbox — только физические границы (VisualFieldScanner внутри card).
Schema не создаёт bbox; она фильтрует: отбрасываем кандидатов в button_row / после flow_end / на кнопке.
Builder = normalizer: выравнивание, дедуп, без генерации из схемы.

«Границы — территория. Schema — карта.»
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.form_schema_models import FormSchema

logger = logging.getLogger(__name__)

# Кандидат отбрасывается, если перекрытие с кнопкой больше этой доли площади кандидата
VETO_ON_BUTTON_COVERAGE = 0.25
# Дедуп: IoU выше порога — один bbox
DEDUP_IOU_THRESHOLD = 0.65
# Высота выше этой доли от медианы → textarea_candidate
TEXTAREA_HEIGHT_RATIO = 1.8
CONFIDENCE_VISUAL_FIRST = 0.82


def _bbox_area(bbox: List[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _intersection_area(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _iou(a: List[float], b: List[float]) -> float:
    inter = _intersection_area(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(1e-9, union)


def _filter_by_buttons(
    visual_bboxes: List[List[float]],
    atoms_inside: List[Dict[str, Any]],
) -> List[List[float]]:
    """Убираем кандидатов, сильно перекрывающихся с кнопкой."""
    button_bboxes = [
        a.get("bbox", [0, 0, 0, 0])
        for a in atoms_inside
        if (a.get("type") or "").strip().lower() in ("button", "synthetic_btn")
        and len(a.get("bbox") or []) >= 4
    ]
    if not button_bboxes:
        return list(visual_bboxes)
    out: List[List[float]] = []
    for vb in visual_bboxes:
        area_v = _bbox_area(vb)
        if area_v <= 0:
            continue
        overlap_any = False
        for bb in button_bboxes:
            inter = _intersection_area(vb, bb)
            if inter / area_v >= VETO_ON_BUTTON_COVERAGE:
                overlap_any = True
                break
        if not overlap_any:
            out.append(vb)
    return out


def _filter_by_schema(
    visual_bboxes: List[List[float]],
    schema: Optional[FormSchema],
    card_bbox: List[float],
) -> List[List[float]]:
    """
    Отбрасываем кандидатов, чей центр Y попадает в button_row или в строку с индексом > flow_end_row_index.
    Schema — только фильтр, не источник bbox.
    """
    if not schema or not schema.rows or not visual_bboxes:
        return list(visual_bboxes)
    n_rows = len(schema.rows)
    card_y1 = card_bbox[1]
    card_h = card_bbox[3] - card_bbox[1]
    if card_h <= 0:
        return list(visual_bboxes)

    out: List[List[float]] = []
    for vb in visual_bboxes:
        cy = (vb[1] + vb[3]) / 2
        # Номер полосы (0..n_rows-1)
        row_index = int((cy - card_y1) / card_h * n_rows)
        row_index = max(0, min(row_index, n_rows - 1))
        role = schema.rows[row_index].role
        if role == "button_row":
            continue
        if schema.flow_end_row_index is not None and row_index > schema.flow_end_row_index:
            continue
        out.append(vb)
    return out


def _deduplicate_bboxes(bboxes: List[List[float]]) -> List[List[float]]:
    """Объединяем дубликаты по IoU; оставляем один (большая площадь)."""
    if len(bboxes) <= 1:
        return list(bboxes)
    used = [False] * len(bboxes)
    result: List[List[float]] = []
    for i in range(len(bboxes)):
        if used[i]:
            continue
        b_i = bboxes[i]
        best_j = -1
        best_area = _bbox_area(b_i)
        for j in range(i + 1, len(bboxes)):
            if used[j]:
                continue
            if _iou(b_i, bboxes[j]) >= DEDUP_IOU_THRESHOLD:
                area_j = _bbox_area(bboxes[j])
                if area_j >= best_area:
                    best_area = area_j
                    best_j = j
        if best_j >= 0:
            used[best_j] = True
            result.append(bboxes[best_j])
        else:
            result.append(b_i)
    return result


def _normalize_align_width(bboxes: List[List[float]], card_bbox: List[float]) -> List[List[float]]:
    """Лёгкая нормализация: выравнивание ширины по медиане (в пределах колонки не трогаем)."""
    if len(bboxes) <= 1 or len(card_bbox) < 4:
        return list(bboxes)
    widths = [b[2] - b[0] for b in bboxes]
    median_w = sorted(widths)[len(widths) // 2]
    result: List[List[float]] = []
    for b in bboxes:
        w = b[2] - b[0]
        if w <= 0:
            result.append(list(b))
            continue
        if abs(w - median_w) / max(median_w, 1e-9) <= 0.25:
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            new_x2 = x1 + median_w
            result.append([x1, y1, new_x2, y2])
        else:
            result.append(list(b))
    return result


def run_visual_first_field_inference(
    form_regions: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    image_path: Optional[str] = None,
    dark_theme: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Visual First: внутри каждой form_region получаем кандидатов от VisualFieldScanner,
    фильтруем по схеме (кнопки, flow_end), нормализуем. Не создаём bbox из схемы.
    Если по карточке нет визуальных кандидатов — для неё ничего не добавляем.
    """
    from src.infrastructure.atoms_v2.card_field_layout_inference import _elements_inside_card
    from src.infrastructure.atoms_v2.form_schema_inference import infer_form_schema
    from src.infrastructure.atoms_v2.visual_field_scanner import run_visual_field_scan

    log_lines: List[str] = []
    new_atoms: List[Dict[str, Any]] = []
    existing_ids = {a.get("id", "") for a in atoms if a.get("id")}

    if not form_regions:
        log_lines.append("visual_first_field: no form_regions, skip")
        return new_atoms, log_lines
    if not image_path:
        log_lines.append("visual_first_field: no image_path, skip (need VisualFieldScanner)")
        return new_atoms, log_lines

    for card in form_regions:
        card_id = card.get("id", "") or ("form_region_%s" % id(card))
        card_bbox = card.get("bbox", [0, 0, 0, 0])
        if len(card_bbox) < 4:
            continue

        # 1. Визуальные кандидаты — единственный источник bbox
        visual_bboxes, scan_log = run_visual_field_scan(image_path, [card], dark_theme=dark_theme)
        log_lines.extend(scan_log)
        if not visual_bboxes:
            log_lines.append("visual_first_field: card_id=%s no visual candidates" % card_id)
            continue

        atoms_inside, ocr_inside = _elements_inside_card(card_bbox, atoms, raw_ocr_boxes)
        schema = infer_form_schema(card_bbox, atoms_inside, ocr_inside)

        # 2. Schema как фильтр: убрать на кнопках и после flow_end
        filtered = _filter_by_buttons(visual_bboxes, atoms_inside)
        filtered = _filter_by_schema(filtered, schema, card_bbox)
        if not filtered:
            log_lines.append("visual_first_field: card_id=%s all candidates filtered by schema" % card_id)
            continue

        # 3. Normalize: dedup + лёгкое выравнивание ширины
        normalized = _deduplicate_bboxes(filtered)
        normalized = _normalize_align_width(normalized, card_bbox)

        heights = [b[3] - b[1] for b in normalized if b[3] > b[1]]
        median_h = float(sorted(heights)[len(heights) // 2]) if heights else 40.0

        for bbox in normalized:
            if len(bbox) < 4:
                continue
            h = bbox[3] - bbox[1]
            atom_type = (
                "textarea_candidate"
                if h >= median_h * TEXTAREA_HEIGHT_RATIO
                else "input_candidate"
            )
            aid = "visual_first_%s" % hashlib.sha256(str([round(x, 1) for x in bbox]).encode()).hexdigest()[:12]
            if aid in existing_ids:
                continue
            existing_ids.add(aid)
            new_atoms.append({
                "id": aid,
                "type": atom_type,
                "bbox": list(bbox),
                "confidence": CONFIDENCE_VISUAL_FIRST,
                "source": "input_candidate_recovery",
                "recovery_source": "visual_first",
                "evidence": {"source": "visual_first", "geometry": True},
            })
        log_lines.append(
            "visual_first_field: card_id=%s visual=%d filtered=%d normalized=%d"
            % (card_id, len(visual_bboxes), len(filtered), len(normalized))
        )

    log_lines.append("visual_first_field: total inferred=%d (source=visual_first)" % len(new_atoms))
    return new_atoms, log_lines
