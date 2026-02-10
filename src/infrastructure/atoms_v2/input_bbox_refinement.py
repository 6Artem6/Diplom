"""
Уточнение границ input по аналогии с кнопками: параллельные линии, цветовой перепад, отсечение label, нормализация ширины.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LABEL_MAX_CHARS = 25
LABEL_HEIGHT_RATIO = 0.5  # OCR выше input с высотой < 0.5 * input_height → label
LABEL_GAP_PX = 2
WIDTH_NORMALIZE_TOLERANCE = 0.15  # ±15% от медианы ширины
MIN_INPUT_WIDTH_PX = 40
ROI_EXPAND_PX = 8
EDGE_THRESHOLD = 50
LINE_EXTENT_MIN = 0.5  # линия покрывает ≥ 50% стороны bbox


def _bbox_center(bbox: List[float]) -> tuple:
    if len(bbox) < 4:
        return (0.0, 0.0)
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _cut_label_from_top(
    bbox: List[float],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> List[float]:
    """
    Если сверху от bbox есть OCR с текстом ≤25 символов и высотой < 0.5*input_height —
    считаем label и поднимаем верхнюю границу (не включаем label в поле).
    """
    if len(bbox) < 4 or not raw_ocr_boxes:
        return list(bbox)
    x1, y1, x2, y2 = bbox
    input_h = y2 - y1
    input_top = y1
    # Ищем OCR выше центра поля (сверху от bbox)
    best_bottom = y1
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        oy1, oy2 = obbox[1], obbox[3]
        text = (ob.get("text") or "").strip()
        if len(text) > LABEL_MAX_CHARS:
            continue
        if oy2 > input_top:
            continue
        oh = oy2 - oy1
        if oh >= LABEL_HEIGHT_RATIO * input_h:
            continue
        # OCR выше поля, короткий текст, маленькая высота → label
        if oy2 > best_bottom:
            best_bottom = oy2
    if best_bottom > y1:
        y1 = best_bottom + LABEL_GAP_PX
        if y2 - y1 < MIN_INPUT_WIDTH_PX:
            return list(bbox)
        return [x1, y1, x2, y2]
    return list(bbox)


def _snap_bbox_to_edges(
    bbox: List[float],
    image_path: str,
    img_shape: tuple,
) -> List[float]:
    """
    В ROI вокруг bbox ищем резкие перепады (Canny), подтягиваем границы к сильным вертикальным/горизонтальным линиям.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return list(bbox)
    if len(bbox) < 4:
        return list(bbox)
    img = cv2.imread(str(image_path))
    if img is None:
        return list(bbox)
    h, w = img.shape[:2]
    x1 = int(max(0, bbox[0] - ROI_EXPAND_PX))
    y1 = int(max(0, bbox[1] - ROI_EXPAND_PX))
    x2 = int(min(w, bbox[2] + ROI_EXPAND_PX))
    y2 = int(min(h, bbox[3] + ROI_EXPAND_PX))
    if x2 <= x1 or y2 <= y1:
        return list(bbox)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return list(bbox)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, EDGE_THRESHOLD, EDGE_THRESHOLD * 2)
    cw, ch = edges.shape[1], edges.shape[0]
    col_sum = np.sum(edges, axis=0)
    row_sum = np.sum(edges, axis=1)
    thresh_col = max(20, float(col_sum.max()) * 0.2)
    thresh_row = max(20, float(row_sum.max()) * 0.2)
    lx1_local = max(0, int(bbox[0]) - x1)
    ly1_local = max(0, int(bbox[1]) - y1)
    lx2_local = min(cw, int(bbox[2]) - x1)
    ly2_local = min(ch, int(bbox[3]) - y1)
    mid_x = (lx1_local + lx2_local) // 2
    mid_y = (ly1_local + ly2_local) // 2
    new_lx1 = lx1_local
    new_lx2 = lx2_local
    for i in range(mid_x, -1, -1):
        if i < len(col_sum) and col_sum[i] >= thresh_col:
            new_lx1 = i
            break
    for i in range(mid_x, cw):
        if i < len(col_sum) and col_sum[i] >= thresh_col:
            new_lx2 = i + 1
            break
    new_ly1 = ly1_local
    new_ly2 = ly2_local
    for i in range(mid_y, -1, -1):
        if i < len(row_sum) and row_sum[i] >= thresh_row:
            new_ly1 = i
            break
    for i in range(mid_y, ch):
        if i < len(row_sum) and row_sum[i] >= thresh_row:
            new_ly2 = i + 1
            break
    return [
        float(x1 + new_lx1),
        float(y1 + new_ly1),
        float(x1 + new_lx2),
        float(y1 + new_ly2),
    ]


def _normalize_width_to_median(
    bbox: List[float],
    other_field_bboxes: List[List[float]],
) -> List[float]:
    """
    Если в одной карточке 2+ поля с близкой шириной — выравниваем ширину текущего под медиану.
    """
    if len(bbox) < 4 or len(other_field_bboxes) < 1:
        return list(bbox)
    widths = [b[2] - b[0] for b in other_field_bboxes if len(b) >= 4]
    if len(widths) < 1:
        return list(bbox)
    import statistics
    median_w = statistics.median(widths)
    x1, y1, x2, y2 = bbox
    cur_w = x2 - x1
    if cur_w <= 0:
        return list(bbox)
    if abs(cur_w - median_w) / max(median_w, 1e-9) > WIDTH_NORMALIZE_TOLERANCE:
        return list(bbox)
    cx = (x1 + x2) / 2
    x1 = cx - median_w / 2
    x2 = cx + median_w / 2
    return [x1, y1, x2, y2]


def refine_input_bbox_like_button(
    bbox: List[float],
    image_path: Optional[str] = None,
    raw_ocr_boxes: Optional[List[Dict[str, Any]]] = None,
    other_field_bboxes_in_card: Optional[List[List[float]]] = None,
    img_shape: Optional[tuple] = None,
) -> List[float]:
    """
    Уточняет bbox поля ввода: отсечение label сверху, привязка к границам по краям, нормализация ширины.

    - raw_ocr_boxes: для отсечения label (OCR сверху ≤25 символов, высота < 0.5*input_height).
    - other_field_bboxes_in_card: для нормализации ширины по медиане (2+ поля в карточке).
    - image_path: для snap к границам по Canny (опционально).
    """
    if len(bbox) < 4:
        return list(bbox)
    out = list(bbox)
    raw_ocr_boxes = raw_ocr_boxes or []
    other_field_bboxes_in_card = other_field_bboxes_in_card or []

    # 1. Отсечь label сверху
    out = _cut_label_from_top(out, raw_ocr_boxes)

    # 2. Snap к границам по изображению (если передан image_path)
    if image_path:
        out = _snap_bbox_to_edges(out, image_path, img_shape or ())

    # 3. Нормализация ширины по медиане других полей в карточке
    if len(other_field_bboxes_in_card) >= 1:
        others = [b for b in other_field_bboxes_in_card if len(b) >= 4 and b != out]
        if len(others) >= 1:
            out = _normalize_width_to_median(out, others)

    return out
