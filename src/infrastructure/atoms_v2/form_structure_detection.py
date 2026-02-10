"""
FormStructureDetection — FORM-FIRST, INPUT-SECOND.

Находит области, похожие на формы (form_region), до поиска input.
Input ищется только внутри form_region.

Признаки формы:
- 2+ вертикально выровненных элементов (по OCR-линиям), одинаковая ширина (±10%)
- Повторяющаяся высота строк (±10%)
- Label слева или сверху (короткий текст слева от зоны ввода)
- Кнопка действия (Save / Submit / Create / Search) внутри или рядом
- Контейнер (card/panel); card ≠ input, input разрешён внутри card
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Пороги для признаков формы
FORM_MIN_LINES_INSIDE = 2
FORM_HEIGHT_TOLERANCE = 0.10   # ±10% высота строки
FORM_WIDTH_TOLERANCE = 0.15    # ±15% ширина выравнивания
LABEL_MAX_CHARS = 25
ACTION_WORDS = frozenset({"save", "submit", "create", "search", "apply", "send", "add", "ok", "go", "login"})
REGION_OCR_COVERAGE_MIN = 0.3  # OCR box считается внутри региона при coverage >= 0.3
FORM_REGION_MIN_CONFIDENCE = 0.4


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


def _coverage_in_outer(inner: List[float], outer: List[float]) -> float:
    area_inner = _bbox_area(inner)
    if area_inner <= 0:
        return 0.0
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / area_inner


def _point_inside_bbox(x: float, y: float, bbox: List[float]) -> bool:
    if len(bbox) < 4:
        return False
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _ocr_inside_region(
    raw_ocr_boxes: List[Dict[str, Any]],
    region_bbox: List[float],
) -> List[Dict[str, Any]]:
    """OCR-боксы, попадающие в регион (coverage >= порога)."""
    out: List[Dict[str, Any]] = []
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _coverage_in_outer(obbox, region_bbox) >= REGION_OCR_COVERAGE_MIN:
            out.append(ob)
    return out


def _group_ocr_into_lines(
    ocr_boxes: List[Dict[str, Any]],
    y_tolerance: float = 18,
) -> List[List[Dict[str, Any]]]:
    sorted_ocr = sorted(
        ocr_boxes,
        key=lambda b: ((b["bbox"][1] + b["bbox"][3]) / 2, b["bbox"][0]),
    )
    lines: List[List[Dict[str, Any]]] = []
    for ob in sorted_ocr:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        cy = (obbox[1] + obbox[3]) / 2
        placed = False
        for line in lines:
            if not line:
                continue
            first_cy = (line[0].get("bbox", [0, 0, 0, 0])[1] + line[0].get("bbox", [0, 0, 0, 0])[3]) / 2
            if abs(cy - first_cy) <= y_tolerance:
                line.append(ob)
                placed = True
                break
        if not placed:
            lines.append([ob])
    return lines


def _line_bbox(line: List[Dict[str, Any]]) -> List[float]:
    if not line:
        return [0.0, 0.0, 0.0, 0.0]
    bboxes = [o.get("bbox", [0, 0, 0, 0]) for o in line if len((o.get("bbox") or [])) >= 4]
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def _has_action_button_nearby(
    region_bbox: List[float],
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    max_distance_center_px: float = 200,
) -> bool:
    """Есть ли кнопка с action-словом внутри региона или рядом."""
    rcx = (region_bbox[0] + region_bbox[2]) / 2
    rcy = (region_bbox[1] + region_bbox[3]) / 2
    for a in atoms:
        t = (a.get("type") or "").strip().lower()
        if t != "button":
            continue
        abbox = a.get("bbox", [0, 0, 0, 0])
        if len(abbox) < 4:
            continue
        acx = (abbox[0] + abbox[2]) / 2
        acy = (abbox[1] + abbox[3]) / 2
        dist = ((rcx - acx) ** 2 + (rcy - acy) ** 2) ** 0.5
        if dist > max_distance_center_px:
            continue
        for ob in raw_ocr_boxes:
            obbox = ob.get("bbox", [0, 0, 0, 0])
            if len(obbox) < 4:
                continue
            if _coverage_in_outer(obbox, abbox) < 0.3:
                continue
            if any(w in (ob.get("text") or "").strip().lower() for w in ACTION_WORDS):
                return True
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _coverage_in_outer(obbox, region_bbox) < 0.5:
            continue
        if any(w in (ob.get("text") or "").strip().lower() for w in ACTION_WORDS):
            return True
    return False


def _score_region_as_form(
    region: Dict[str, Any],
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, bool]]:
    """
    Оценка региона как form_region.
    Возвращает (confidence 0–1, evidence dict).
    """
    rbbox = region.get("bbox", [0, 0, 0, 0])
    if len(rbbox) < 4:
        return 0.0, {}

    evidence: Dict[str, bool] = {
        "aligned_fields": False,
        "labels_present": False,
        "action_button": False,
    }

    ocr_inside = _ocr_inside_region(raw_ocr_boxes, rbbox)
    if len(ocr_inside) < 2:
        return 0.0, evidence

    lines = _group_ocr_into_lines(ocr_inside)
    if len(lines) < FORM_MIN_LINES_INSIDE:
        return 0.0, evidence

    line_bboxes = [_line_bbox(l) for l in lines]
    widths = [b[2] - b[0] for b in line_bboxes]
    heights = [b[3] - b[1] for b in line_bboxes]
    avg_w = sum(widths) / len(widths)
    avg_h = sum(heights) / len(heights)
    if avg_w <= 0 or avg_h <= 0:
        return 0.0, evidence

    aligned = all(abs(w - avg_w) / avg_w <= FORM_WIDTH_TOLERANCE for w in widths)
    height_ok = all(abs(h - avg_h) / avg_h <= FORM_HEIGHT_TOLERANCE for h in heights) or len(heights) >= 2
    if aligned or (len(lines) >= 2 and height_ok):
        evidence["aligned_fields"] = True

    for line in lines:
        for ob in line:
            text = (ob.get("text") or "").strip()
            if len(text) <= LABEL_MAX_CHARS and len(text) >= 1:
                obbox = ob.get("bbox", [0, 0, 0, 0])
                if len(obbox) >= 4:
                    ox_center = (obbox[0] + obbox[2]) / 2
                    region_left_third = rbbox[0] + (rbbox[2] - rbbox[0]) * 0.35
                    if ox_center < region_left_third:
                        evidence["labels_present"] = True
                        break
        if evidence["labels_present"]:
            break

    evidence["action_button"] = _has_action_button_nearby(rbbox, atoms, raw_ocr_boxes)

    score = 0.0
    if evidence["aligned_fields"]:
        score += 0.4
    if evidence["labels_present"]:
        score += 0.35
    if evidence["action_button"]:
        score += 0.25
    return min(1.0, score), evidence


def detect_form_regions(
    regions: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    min_confidence: float = FORM_REGION_MIN_CONFIDENCE,
) -> List[Dict[str, Any]]:
    """
    Находит регионы, похожие на формы.
    Выход: список { type: "form_region", bbox, confidence, evidence }.
    """
    if not regions or not raw_ocr_boxes:
        return []

    form_regions: List[Dict[str, Any]] = []
    for r in regions:
        rbbox = r.get("bbox", [0, 0, 0, 0])
        if len(rbbox) < 4:
            continue
        confidence, evidence = _score_region_as_form(r, atoms, raw_ocr_boxes)
        if confidence < min_confidence:
            continue
        form_regions.append({
            "type": "form_region",
            "id": r.get("id", "") or ("form_region_%d" % len(form_regions)),
            "bbox": list(rbbox),
            "confidence": confidence,
            "evidence": evidence,
        })
    return form_regions


def point_inside_any_form_region(
    x: float, y: float,
    form_regions: List[Dict[str, Any]],
) -> bool:
    """Точка (x, y) лежит внутри хотя бы одного form_region."""
    for fr in form_regions:
        bbox = fr.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4 and _point_inside_bbox(x, y, bbox):
            return True
    return False


def bbox_overlaps_form_region(
    bbox: List[float],
    form_regions: List[Dict[str, Any]],
    min_coverage: float = 0.3,
) -> bool:
    """Bbox пересекается с form_region (центр bbox внутри или coverage площади)."""
    if len(bbox) < 4:
        return False
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    for fr in form_regions:
        rbbox = fr.get("bbox", [0, 0, 0, 0])
        if len(rbbox) < 4:
            continue
        if _point_inside_bbox(cx, cy, rbbox):
            return True
        if _coverage_in_outer(bbox, rbbox) >= min_coverage:
            return True
    return False
