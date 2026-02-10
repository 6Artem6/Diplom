"""
Layout propagation внутри form_region: при ≥2 полях вычисляем шаг по Y/X и добавляем кандидаты в ожидаемых позициях.
Только внутри form_region. Не путать с карточками и layout.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROPAGATION_CONFIDENCE = 0.5
MIN_STEP_Y_PX = 30
MAX_STEP_Y_PX = 120
WEAK_EDGE_THRESHOLD = 30
ROI_MARGIN = 4
MIN_OVERLAP_WITH_FORM = 0.5


def _point_inside_bbox(x: float, y: float, bbox: List[float]) -> bool:
    if len(bbox) < 4:
        return False
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    if len(bbox) < 4:
        return (0.0, 0.0)
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _fields_in_form_region(
    field_bboxes: List[List[float]],
    form_bbox: List[float],
) -> List[List[float]]:
    """Поля, центр которых внутри form_region."""
    if len(form_bbox) < 4:
        return []
    out: List[List[float]] = []
    for b in field_bboxes:
        if len(b) < 4:
            continue
        cx, cy = _bbox_center(b)
        if _point_inside_bbox(cx, cy, form_bbox):
            out.append(b)
    return out


def _compute_step_y(bboxes: List[List[float]]) -> Optional[float]:
    """Медиана шага по Y между соседними полями (отсортированы по y1)."""
    if len(bboxes) < 2:
        return None
    sorted_bboxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
    steps: List[float] = []
    for i in range(1, len(sorted_bboxes)):
        dy = sorted_bboxes[i][1] - sorted_bboxes[i - 1][3]
        if MIN_STEP_Y_PX <= dy <= MAX_STEP_Y_PX:
            steps.append(dy)
    if not steps:
        return None
    return float(statistics.median(steps))


def _has_weak_contour_at(
    image_path: str,
    bbox: List[float],
) -> bool:
    """Есть ли слабый контур или цветовой перепад в ROI bbox."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    if len(bbox) < 4:
        return False
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    x1 = max(0, int(bbox[0]) - ROI_MARGIN)
    y1 = max(0, int(bbox[1]) - ROI_MARGIN)
    x2 = min(w, int(bbox[2]) + ROI_MARGIN)
    y2 = min(h, int(bbox[3]) + ROI_MARGIN)
    if x2 <= x1 or y2 <= y1:
        return False
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, WEAK_EDGE_THRESHOLD, WEAK_EDGE_THRESHOLD * 2)
    if np.sum(edges) > 100:
        return True
    return False


def run_layout_propagation(
    form_regions: List[Dict[str, Any]],
    field_bboxes: List[List[float]],
    image_path: Optional[str] = None,
) -> Tuple[List[Tuple[List[float], float, Dict[str, bool], str]], int]:
    """
    Для каждого form_region с ≥2 полями: вычисляем step_y (и при необходимости step_x),
    ищем ожидаемые позиции следующих полей; при слабом контуре/перепаде добавляем кандидата с пониженным confidence.
    Возвращает ([(bbox, confidence, evidence, source)], count_added).
    """
    added: List[Tuple[List[float], float, Dict[str, bool], str]] = []
    if not form_regions or len(field_bboxes) < 2 or not image_path:
        return added, 0

    evidence = {"geometry": True, "text_density": False, "alignment": True, "context": False}
    for fr in form_regions:
        fbbox = fr.get("bbox", [0, 0, 0, 0])
        if len(fbbox) < 4:
            continue
        in_form = _fields_in_form_region(field_bboxes, fbbox)
        if len(in_form) < 2:
            continue
        step_y = _compute_step_y(in_form)
        if step_y is None:
            continue
        sorted_in = sorted(in_form, key=lambda b: (b[1], b[0]))
        med_w = statistics.median(b[2] - b[0] for b in sorted_in)
        med_h = statistics.median(b[3] - b[1] for b in sorted_in)
        top_y = min(b[1] for b in sorted_in)
        bottom_y = max(b[3] for b in sorted_in)
        left_x = min(b[0] for b in sorted_in)
        right_x = max(b[2] for b in sorted_in)
        # Ожидаемые позиции выше первого поля
        y_above = top_y - step_y
        if y_above >= fbbox[1] and y_above + med_h <= fbbox[3]:
            expected_bbox = [left_x, y_above, left_x + med_w, y_above + med_h]
            if _point_inside_bbox((expected_bbox[0] + expected_bbox[2]) / 2, (expected_bbox[1] + expected_bbox[3]) / 2, fbbox):
                if _has_weak_contour_at(image_path, expected_bbox):
                    added.append((expected_bbox, PROPAGATION_CONFIDENCE, evidence, "propagation"))
        # Ожидаемые позиции ниже последнего поля
        y_below = bottom_y + step_y
        if y_below + med_h <= fbbox[3] and y_below >= fbbox[1]:
            expected_bbox = [left_x, y_below, left_x + med_w, y_below + med_h]
            if _point_inside_bbox((expected_bbox[0] + expected_bbox[2]) / 2, (expected_bbox[1] + expected_bbox[3]) / 2, fbbox):
                if _has_weak_contour_at(image_path, expected_bbox):
                    added.append((expected_bbox, PROPAGATION_CONFIDENCE, evidence, "propagation"))

    return added, len(added)
