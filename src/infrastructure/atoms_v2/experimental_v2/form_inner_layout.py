"""
Уровень 1 — FormInnerLayout: строки и колонки строго внутри FormContainer.bbox.

Типы строк: FIELD_HORIZONTAL, FIELD_VERTICAL, FIELD_INPUT_ONLY, TEXTAREA, ACTION, TEXT.
Высота textarea — по первой устойчивой горизонтальной линии (underline/border); fallback на bbox.
Внутренняя декомпозиция: label_bbox, input_bbox, helper_bbox по визуальным кандидатам и OCR.
"""

from __future__ import annotations

import logging
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.models import (
    FormColumn,
    FormContainer,
    FormRow,
    FormSkeleton,
    RowType,
    HeightMode,
)

logger = logging.getLogger(__name__)

MIN_ROW_HEIGHT = 14
TEXTAREA_HEIGHT_MEDIAN_K = 1.8
TEXTAREA_MIN_HEIGHT_PX = 60
ACTION_WORDS = frozenset({
    "save", "submit", "search", "send", "add", "ok", "login", "cancel", "apply",
    "отправить", "сохранить", "войти", "далее",
})
X_CLUSTER_TOLERANCE_RATIO = 0.12
LAYOUT_VERTICAL_X_VARIANCE_MAX = 0.04
X_OVERLAP_THRESHOLD_RATIO = 0.3
LABEL_ABOVE_MAX_GAP_PX = 50
HEADER_ZONE_TOP_RATIO = 0.18
FONT_HEADER_RATIO = 1.5
FONT_HINT_RATIO = 0.6
HELPER_FONT_MAX_RATIO = 0.85
BUTTON_CENTER_TOLERANCE = 0.25
# Зона поиска label: слева до 0.35*container_width, сверху до 0.5*row_height
LABEL_MAX_LEFT_RATIO = 0.35
LABEL_ABOVE_ROW_RATIO = 0.5
# Helper только если top(helper) ∈ [bottom(input), bottom(input)+1.2*input_height]
HELPER_BELOW_INPUT_RATIO = 1.2
BUTTON_WIDTH_MIN_RATIO = 0.25


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[len(s) // 2])


def normalize_ocr_for_layout(
    ocr_inside: List[Dict[str, Any]],
    container_bbox: List[float],
    image_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[List[float]]]:
    """
    Baseline формы + OCR gating. Возвращает (layout_ocr, baseline, header_bboxes).
    layout_ocr — только OCR, допущенный в row/slot inference (не header, не button, не hint, не placeholder).
    """
    if len(container_bbox) < 4:
        return [], {"median_font_height": 20.0}, []
    x1, y1, x2, y2 = container_bbox[0], container_bbox[1], container_bbox[2], container_bbox[3]
    w = x2 - x1
    h = y2 - y1
    header_zone_bottom = y1 + h * HEADER_ZONE_TOP_RATIO
    container_cx = (x1 + x2) / 2

    cand_for_baseline: List[Dict[str, Any]] = []
    for ob in ocr_inside:
        b = ob.get("bbox", [])
        if len(b) < 4:
            continue
        if b[0] < x1 or b[2] > x2 or b[1] < y1 or b[3] > y2:
            continue
        ob_h = b[3] - b[1]
        ob_cx = (b[0] + b[2]) / 2
        ob_w = b[2] - b[0]
        txt = (ob.get("text") or "").strip().lower()
        if any(w in txt for w in ACTION_WORDS):
            continue
        if b[1] < header_zone_bottom:
            continue
        centered = abs(ob_cx - container_cx) / w < BUTTON_CENTER_TOLERANCE if w > 0 else False
        if centered and ob_w >= w * BUTTON_WIDTH_MIN_RATIO:
            continue
        cand_for_baseline.append(ob)

    heights = [ob["bbox"][3] - ob["bbox"][1] for ob in cand_for_baseline if len(ob.get("bbox", [])) >= 4]
    median_font_height = _median(heights) if heights else 20.0
    baseline = {"median_font_height": median_font_height}

    header_bboxes: List[List[float]] = []
    layout_ocr: List[Dict[str, Any]] = []
    for ob in ocr_inside:
        b = ob.get("bbox", [])
        if len(b) < 4:
            continue
        if b[0] < x1 or b[2] > x2 or b[1] < y1 or b[3] > y2:
            continue
        ob_h = b[3] - b[1]
        ob_cx = (b[0] + b[2]) / 2
        ob_w = b[2] - b[0]
        txt = (ob.get("text") or "").strip().lower()

        if ob_h > median_font_height * FONT_HEADER_RATIO:
            if b[1] < header_zone_bottom or (b[1] + b[3]) / 2 < header_zone_bottom + 30:
                header_bboxes.append(list(b))
            continue
        if ob_h < median_font_height * FONT_HINT_RATIO:
            continue
        if any(w in txt for w in ACTION_WORDS):
            continue
        centered = abs(ob_cx - container_cx) / w < BUTTON_CENTER_TOLERANCE if w > 0 else False
        if centered and ob_w >= w * BUTTON_WIDTH_MIN_RATIO:
            continue
        layout_ocr.append(ob)

    return layout_ocr, baseline, header_bboxes


TEXTAREA_VISUAL_HEIGHT_PX = 90
TEXTAREA_MEDIAN_RATIO = 2.0   # height ≥ 2.0 × median_input_height
TEXTAREA_MIN_WIDTH_RATIO = 0.5  # width ≥ 0.5 container
TEXTAREA_FRAME_RATIO = 0.6     # visible frame ratio ≥ 0.6
TALL_CONTOUR_MEDIAN_RATIO = 1.8
TALL_CONTOUR_MIN_HEIGHT_PX = 90
TALL_CONTOUR_MIN_WIDTH_RATIO = 0.5
Y_OVERLAP_GAP_PX = 40
ROW_SNAP_PADDING_PX = 6
ROW_OVERLAP_SMALL_PX = 15
ROW_OVERLAP_MERGE_RATIO = 0.35
GRID_X_OVERLAP_THRESHOLD = 0.3
MIN_ROW_HEIGHT_PX = 24
MIN_ROW_WIDTH_RATIO = 0.6
MAX_ROW_HEIGHT_RATIO = 0.25
MIN_ROW_WIDTH_PX = 140
MAX_ROW_HEIGHT_NON_TEXTAREA = 120
ROW_INPUT_TOP_PAD = 8
ROW_OCR_PADDING_PX = 4
# OCR не может расширять строку вверх более чем на MAX_LABEL_HEIGHT
MAX_LABEL_HEIGHT_PX = 60
# Layout по OCR относительно input: label сверху / слева
LABEL_ABOVE_INPUT_TOP_GAP_PX = 8
LABEL_LEFT_INPUT_GAP_PX = 10
LABEL_ABOVE_OVERLAP_X_MIN = 0.3
LABEL_LEFT_OVERLAP_Y_MIN = 0.4
PLACEHOLDER_VERTICAL_MARGIN_RATIO = 0.25
BUTTON_ROW_MAX_HEIGHT_PX = 60
BUTTON_ROW_ASPECT_MIN = 3.0
TEXTAREA_HEIGHT_THRESHOLD_PX = 90
GRID_COLUMN_X_TOLERANCE_PX = 12
MIN_INPUT_WIDTH_RATIO = 0.4
MIN_INPUT_RIGHT_RATIO = 0.9
GRID_MIN_INPUT_WIDTH = 60
GRID_MIN_GAP_X = 40
GRID_MIN_X_DISTANCE_RATIO = 0.15
BUTTON_ASPECT_MIN = 3.0
BUTTON_HEIGHT_MIN = 28
BUTTON_HEIGHT_MAX = 70
MIN_ANCHORS_FOR_CONTAINER = 3
MIN_INPUT_BBOX_INSIDE_CONTAINER = 2
REJECT_ALL_BBOX_HEIGHT_LT = 40
TEXTAREA_MIN_WIDTH_PX = 120
TEXTAREA_MAX_ASPECT = 8.0
TEXTAREA_CENTER_Y_MIN_RATIO = 0.2
PLACEHOLDER_X_OVERLAP_MIN = 0.8
PLACEHOLDER_TEXT_MAX_LEN = 25
HORIZONTAL_SEPARATOR_GAP_PX = 40


def _is_button_bbox(bbox: List[float]) -> bool:
    """BBox считается кнопкой: aspect_ratio >= 3, height в [28, 70]."""
    if len(bbox) < 4:
        return False
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if h <= 0:
        return False
    aspect = w / h
    return aspect >= BUTTON_ASPECT_MIN and BUTTON_HEIGHT_MIN <= h <= BUTTON_HEIGHT_MAX


def validate_container_with_ocr(container_bbox: List[float], ocr_inside: List[Dict[str, Any]]) -> bool:
    """I1: внутри контейнера ≥ 2 OCR-блока в разных Y-диапазонах (не форма иначе)."""
    if len(container_bbox) < 4:
        return False
    x1, y1, x2, y2 = container_bbox[0], container_bbox[1], container_bbox[2], container_bbox[3]
    centers_y: List[float] = []
    for ob in ocr_inside:
        b = ob.get("bbox") or []
        if len(b) < 4:
            continue
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            centers_y.append(cy)
    if len(centers_y) < 2:
        return False
    band = 20.0
    ys = sorted(centers_y)
    if ys[-1] - ys[0] < band:
        return False
    return True


def validate_container_with_visual(container_bbox: List[float], visual_candidates: List[List[float]]) -> bool:
    """
    Главный инвариант: внутри ≥ 3 строк (анкоров), ≥ 2 candidate field элементов (не кнопка).
    Отклоняется если: height < 120, внутри только 1 bbox, все внутренние bbox height < 40, контейнер — кнопка.
    """
    if len(container_bbox) < 4:
        return False
    cw = container_bbox[2] - container_bbox[0]
    ch = container_bbox[3] - container_bbox[1]
    if ch < 120:
        return False
    if cw <= 0 or ch <= 0:
        return False
    aspect = cw / ch
    if aspect >= BUTTON_ASPECT_MIN and BUTTON_HEIGHT_MIN <= ch <= BUTTON_HEIGHT_MAX:
        return False
    inside = [b for b in visual_candidates if len(b) >= 4 and _bbox_in_container(b, container_bbox)]
    if len(inside) < 2:
        return False
    if all((b[3] - b[1]) < REJECT_ALL_BBOX_HEIGHT_LT for b in inside):
        return False
    non_button = [b for b in inside if not _is_button_bbox(b)]
    if len(non_button) < MIN_INPUT_BBOX_INSIDE_CONTAINER:
        return False
    anchors = collect_field_row_anchors(visual_candidates, container_bbox)
    if len(anchors) < MIN_ANCHORS_FOR_CONTAINER:
        return False
    return True


def _is_valid_label_text(text: str) -> bool:
    """Label считается валидным только если len(normalized) >= 2 и не только спецсимволы."""
    if not text or not isinstance(text, str):
        return False
    normalized = re.sub(r"\s+", " ", text.strip())
    if len(normalized) < 2:
        return False
    letters_digits = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9]", "", normalized)
    return len(letters_digits) >= 1


def _bbox_in_container(b: List[float], c: List[float]) -> bool:
    if len(b) < 4 or len(c) < 4:
        return False
    return c[0] <= b[0] and b[2] <= c[2] and c[1] <= b[1] and b[3] <= c[3]


def _bbox_fully_inside(inner: List[float], outer: List[float]) -> bool:
    if len(inner) < 4 or len(outer) < 4:
        return False
    return (outer[0] <= inner[0] and inner[2] <= outer[2]
            and outer[1] <= inner[1] and inner[3] <= outer[3])


def _x_overlap_ratio(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix1 = max(a[0], b[0])
    ix2 = min(a[2], b[2])
    if ix2 <= ix1:
        return 0.0
    wa = a[2] - a[0]
    return (ix2 - ix1) / wa if wa > 0 else 0.0


def _overlap_x(a: List[float], b: List[float]) -> float:
    """Доля пересечения по X относительно b: intersection_w / b.width."""
    if len(a) < 4 or len(b) < 4:
        return 0.0
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    bw = b[2] - b[0]
    return iw / bw if bw > 0 else 0.0


def _overlap_y(a: List[float], b: List[float]) -> float:
    """Доля пересечения по Y относительно b: intersection_h / b.height."""
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    bh = b[3] - b[1]
    return ih / bh if bh > 0 else 0.0


def _is_placeholder_vertical_zone(ob: Dict[str, Any], input_bbox: List[float]) -> bool:
    """OCR.center_y ∈ [input.top + 0.25h, input.bottom - 0.25h] — не используется как label."""
    b = ob.get("bbox") or []
    if len(b) < 4 or len(input_bbox) < 4:
        return False
    cy = (b[1] + b[3]) / 2
    iy_min, iy_max = input_bbox[1], input_bbox[3]
    h = iy_max - iy_min
    margin = h * PLACEHOLDER_VERTICAL_MARGIN_RATIO
    return (iy_min + margin) <= cy <= (iy_max - margin)


def _is_placeholder_ocr(ob: Dict[str, Any], input_bbox: List[float]) -> bool:
    """OCR считается placeholder: X-overlap с input > 80%, полностью внутри input, text < 25 символов."""
    b = ob.get("bbox") or []
    if len(b) < 4 or len(input_bbox) < 4:
        return False
    if _x_overlap_ratio(b, input_bbox) < PLACEHOLDER_X_OVERLAP_MIN:
        return False
    if not _bbox_fully_inside(b, input_bbox):
        return False
    txt = (ob.get("text") or "").strip()
    return len(txt) < PLACEHOLDER_TEXT_MAX_LEN


def _y_overlap_or_near(b1: List[float], b2: List[float], gap: float = Y_OVERLAP_GAP_PX) -> bool:
    if len(b1) < 4 or len(b2) < 4:
        return False
    if b1[3] < b2[1] - gap or b2[3] < b1[1] - gap:
        return False
    return True


def _horizontal_separator_y_in_container(
    image_path: Optional[str], container_bbox: List[float],
) -> List[float]:
    """Горизонтальные линии (Canny + Hough) в контейнере. Возвращает список Y в глобальных координатах."""
    if not image_path or len(container_bbox) < 4:
        return []
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        x1, y1, x2, y2 = int(container_bbox[0]), int(container_bbox[1]), int(container_bbox[2]), int(container_bbox[3])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
        out: List[float] = []
        if lines is not None:
            for line in lines:
                xa, ya, xb, yb = line[0]
                if abs(yb - ya) <= 2:
                    out.append(float(y1 + (ya + yb) / 2))
            out = list(dict.fromkeys([round(y, 0) for y in out]))
        return sorted(out)
    except Exception:
        return []


def collect_field_row_anchors(
    visual_candidates: List[List[float]],
    container_bbox: List[float],
    image_path: Optional[str] = None,
    layout_ocr: Optional[List[Dict[str, Any]]] = None,
    baseline: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Якоря строк только из bbox визуальных полей. Кластеризация по Y-overlap.
    Tall (h>=TEXTAREA_VISUAL_HEIGHT_PX) не сливаются с обычными полями.
    Если image_path задан: горизонтальные линии (Canny+Hough) принудительно разделяют anchors при gap < 40px.
    Если layout_ocr и baseline заданы: при наличии в одном Y-кластере крупного текста (font > baseline*1.35)
    и bbox нормальной высоты кластер разделяется — отдельный якорь под заголовок (from_ocr_header).
    """
    if len(container_bbox) < 4 or not visual_candidates:
        return []
    inside = [b for b in visual_candidates if len(b) >= 4 and _bbox_in_container(b, container_bbox)]
    if not inside:
        return []
    inside = sorted(inside, key=lambda b: (b[1], b[0]))

    # gap = min(0.5 * median_input_height, 20px) для кластеризации по Y
    heights_all = [b[3] - b[1] for b in inside if len(b) >= 4]
    median_h = _median(heights_all) if heights_all else 40.0
    overlap_gap = min(0.5 * median_h, 20.0)
    tall_height_threshold = max(TALL_CONTOUR_MIN_HEIGHT_PX, TALL_CONTOUR_MEDIAN_RATIO * median_h)

    separator_ys = _horizontal_separator_y_in_container(image_path, container_bbox)

    def _is_tall(bb: List[float]) -> bool:
        return len(bb) >= 4 and (bb[3] - bb[1]) >= tall_height_threshold

    def _line_splits(b1: List[float], b2: List[float]) -> bool:
        if not separator_ys or len(b1) < 4 or len(b2) < 4:
            return False
        top1, bottom1 = b1[1], b1[3]
        top2, bottom2 = b2[1], b2[3]
        gap_lo = min(bottom1, bottom2)
        gap_hi = max(top1, top2)
        for ly in separator_ys:
            if gap_lo <= ly <= gap_hi:
                if min(abs(ly - bottom1), abs(ly - top2)) < HORIZONTAL_SEPARATOR_GAP_PX:
                    return True
            if top2 > bottom1 and bottom1 <= ly <= top2:
                if min(ly - bottom1, top2 - ly) < HORIZONTAL_SEPARATOR_GAP_PX:
                    return True
        return False

    clusters: List[List[List[float]]] = []
    for b in inside:
        placed = False
        b_tall = _is_tall(b)
        for cl in clusters:
            for c in cl:
                if not _y_overlap_or_near(b, c, overlap_gap):
                    continue
                if _line_splits(c, b) or _line_splits(b, c):
                    continue
                if _is_tall(c) != b_tall:
                    continue
                cl.append(b)
                placed = True
                break
            if placed:
                break
        if not placed:
            clusters.append([b])

    anchors = []
    for cl in clusters:
        y_min = min(bb[1] for bb in cl)
        y_max = max(bb[3] for bb in cl)
        anchors.append({
            "bboxes": list(cl),
            "y_min": y_min,
            "y_max": y_max,
            "x_min": min(bb[0] for bb in cl),
            "x_max": max(bb[2] for bb in cl),
        })

    # Разделение кластера: крупный текст (OCR > baseline*1.35) и bbox нормальной высоты в одном Y → два якоря
    if layout_ocr and baseline and len(container_bbox) >= 4:
        header_font_min = float(baseline.get("median_font_height", 20.0)) * 1.35
        expanded: List[Dict[str, Any]] = []
        for a in anchors:
            bboxes = list(a["bboxes"])
            a_y_min, a_y_max = a["y_min"], a["y_max"]
            large_ocr_in_range = [
                ob for ob in layout_ocr
                if len(ob.get("bbox") or []) >= 4
                and (ob["bbox"][3] - ob["bbox"][1]) > header_font_min
                and not (ob["bbox"][3] < a_y_min or ob["bbox"][1] > a_y_max)
            ]
            if not large_ocr_in_range or not bboxes:
                expanded.append(a)
                continue
            remaining = list(bboxes)
            for ob in large_ocr_in_range:
                ob_bbox = list(ob["bbox"])
                ob_h = ob_bbox[3] - ob_bbox[1]
                next_remaining = []
                for bb in remaining:
                    if len(bb) < 4:
                        next_remaining.append(bb)
                        continue
                    bb_h = bb[3] - bb[1]
                    overlap_lo = max(bb[1], ob_bbox[1])
                    overlap_hi = min(bb[3], ob_bbox[3])
                    overlap_h = max(0, overlap_hi - overlap_lo)
                    if overlap_h > 0.3 * min(bb_h, ob_h):
                        continue
                    next_remaining.append(bb)
                remaining = next_remaining
                expanded.append({
                    "bboxes": [ob_bbox],
                    "y_min": ob_bbox[1],
                    "y_max": ob_bbox[3],
                    "x_min": ob_bbox[0],
                    "x_max": ob_bbox[2],
                    "from_ocr_header": True,
                })
            if remaining:
                expanded.append({
                    "bboxes": remaining,
                    "y_min": min(bb[1] for bb in remaining),
                    "y_max": max(bb[3] for bb in remaining),
                    "x_min": min(bb[0] for bb in remaining),
                    "x_max": max(bb[2] for bb in remaining),
                })
        anchors = expanded

    anchors.sort(key=lambda a: a["y_min"])
    return anchors


def _apply_row_invariants(rows: List[FormRow], container_bbox: List[float]) -> None:
    """
    Инвариант строк: верх = min(top(input), top(label)) - padding; только по input-геометрии.
    height >= 24px, height <= 0.25*container_height (не textarea), width >= 0.6*container_width.
    """
    if len(container_bbox) < 4 or not rows:
        return
    c_x_min, c_y_min, c_x_max, c_y_max = container_bbox[0], container_bbox[1], container_bbox[2], container_bbox[3]
    container_w = c_x_max - c_x_min
    container_h = c_y_max - c_y_min
    max_row_h = min(MAX_ROW_HEIGHT_NON_TEXTAREA, container_h * MAX_ROW_HEIGHT_RATIO) if container_h > 0 else MAX_ROW_HEIGHT_NON_TEXTAREA
    min_row_w = max(MIN_ROW_WIDTH_PX, container_w * MIN_ROW_WIDTH_RATIO) if container_w > 0 else MIN_ROW_WIDTH_PX
    for r in rows:
        inputs = (r.input_bboxes if r.input_bboxes else [r.input_bbox]) if r.input_bbox or r.input_bboxes else []
        if inputs:
            iy_min = min(ib[1] for ib in inputs)
            iy_max = max(ib[3] for ib in inputs)
            r.y_min = min(r.y_min, iy_min - ROW_INPUT_TOP_PAD)
            r.y_min = max(r.y_min, c_y_min, iy_min - MAX_LABEL_HEIGHT_PX)
            r.y_max = max(r.y_max, iy_max)
            r.y_max = max(r.y_max, iy_max + ROW_OCR_PADDING_PX)
        row_h = r.y_max - r.y_min
        row_w = r.x_max - r.x_min
        is_textarea = r.row_type == "TEXTAREA"
        if row_h < MIN_ROW_HEIGHT_PX:
            r.y_max = r.y_min + MIN_ROW_HEIGHT_PX
        elif not is_textarea and row_h > max_row_h:
            r.y_max = r.y_min + max_row_h
        if row_w < min_row_w:
            center_x = (r.x_min + r.x_max) / 2
            half = min_row_w / 2
            r.x_min = max(c_x_min, center_x - half)
            r.x_max = min(c_x_max, center_x + half)
            if r.x_max - r.x_min < min_row_w:
                r.x_min = max(c_x_min, r.x_max - min_row_w)
                if r.x_min >= r.x_max:
                    r.x_max = min(c_x_max, r.x_min + min_row_w)


def _remove_orphan_field_rows(rows: List[FormRow]) -> None:
    """Удалить строки: row_type не ACTION/TEXTAREA/HEADER и нет input_bbox. In-place + переиндексация."""
    keep: List[FormRow] = []
    for r in rows:
        if r.row_type in ("ACTION", "TEXTAREA", "HEADER"):
            keep.append(r)
        elif r.input_bbox is not None or (getattr(r, "input_bboxes", None) and len(r.input_bboxes) > 0):
            keep.append(r)
    rows.clear()
    rows.extend(keep)
    for i, r in enumerate(rows):
        r.row_index = i


def _normalize_row_overlaps(rows: List[FormRow], container_y1: float, container_y2: float) -> None:
    """
    Post-pass: строки не пересекаются по Y. Сортировка по y_min; при overlap < threshold — раздвинуть,
    при значимом overlap — объединить. Переиндексация row_index.
    """
    if len(rows) < 2:
        return
    rows.sort(key=lambda r: r.y_min)
    i = 0
    while i < len(rows):
        r = rows[i]
        r.row_index = i
        if i + 1 >= len(rows):
            break
        rnext = rows[i + 1]
        overlap = min(r.y_max, rnext.y_max) - rnext.y_min
        if overlap <= 0:
            i += 1
            continue
        h1, h2 = r.y_max - r.y_min, rnext.y_max - rnext.y_min
        small_threshold = max(ROW_OVERLAP_SMALL_PX, 0.2 * min(h1, h2))
        if overlap < small_threshold:
            mid = (r.y_max + rnext.y_min) / 2.0
            r.y_max = mid
            rnext.y_min = mid
            i += 1
        else:
            r.y_max = max(r.y_max, rnext.y_max)
            rows.pop(i + 1)
            for j in range(i + 1, len(rows)):
                rows[j].row_index = j
    for r in rows:
        r.y_min = max(container_y1, r.y_min)
        r.y_max = min(container_y2, r.y_max)


def _tall_contours_inside_container(
    image_path: str,
    container_bbox: List[float],
    median_input_height: Optional[float] = None,
) -> List[List[float]]:
    """
    Визуальные контуры: height ≥ max(1.8 × median_input_height, 90px), width ≥ 0.5 container.
    Не участвует OCR.
    """
    import cv2
    if len(container_bbox) < 4:
        return []
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    x1, y1, x2, y2 = int(container_bbox[0]), int(container_bbox[1]), int(container_bbox[2]), int(container_bbox[3])
    container_w = x2 - x1
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height_threshold = max(
        TALL_CONTOUR_MIN_HEIGHT_PX,
        (TALL_CONTOUR_MEDIAN_RATIO * (median_input_height or 50.0)),
    )
    min_width = max(60, container_w * TALL_CONTOUR_MIN_WIDTH_RATIO) if container_w > 0 else 60
    out: List[List[float]] = []
    for c in contours:
        if cv2.contourArea(c) < 800:
            continue
        rx, ry, rw, rh = cv2.boundingRect(c)
        if rh < height_threshold:
            continue
        if rw < min_width:
            continue
        out.append([float(x1 + rx), float(y1 + ry), float(x1 + rx + rw), float(y1 + ry + rh)])
    return out


DEMO_ROW_LABEL_TOP_PAD = 20
DEMO_ROW_BOTTOM_PAD = 8
DEMO_BUTTON_ASPECT_MIN = 3.0
DEMO_BUTTON_HEIGHT_MIN = 28
DEMO_BUTTON_HEIGHT_MAX = 70


def _build_rows_demo_mode(
    container: FormContainer,
    visual_candidates: List[List[float]],
) -> List[FormRow]:
    """
    Demo_mode: одна строка = один input_bbox. Строго по визуальным кандидатам.
    Кнопка (aspect >= 3, height 28–70) → ACTION, иначе FIELD_VERTICAL. Grid отключён.
    """
    if len(container.bbox) < 4 or not visual_candidates:
        return []
    x1, y1, x2, y2 = container.bbox[0], container.bbox[1], container.bbox[2], container.bbox[3]
    inside = [
        b for b in visual_candidates
        if len(b) >= 4
        and x1 <= b[0] and b[2] <= x2 and y1 <= b[1] and b[3] <= y2
    ]
    if not inside:
        return []
    inside.sort(key=lambda b: (b[1] + b[3]) / 2)
    rows: List[FormRow] = []
    for i, bbox in enumerate(inside):
        rw = bbox[2] - bbox[0]
        rh = bbox[3] - bbox[1]
        aspect = rw / max(1e-9, rh)
        is_button = (
            DEMO_BUTTON_HEIGHT_MIN <= rh <= DEMO_BUTTON_HEIGHT_MAX
            and aspect >= DEMO_BUTTON_ASPECT_MIN
        )
        row_type: RowType = "ACTION" if is_button else "FIELD_VERTICAL"
        row_y_min = max(y1, bbox[1] - DEMO_ROW_LABEL_TOP_PAD)
        row_y_max = min(y2, bbox[3] + DEMO_ROW_BOTTOM_PAD)
        r = FormRow(
            row_index=i,
            y_min=row_y_min,
            y_max=row_y_max,
            x_min=x1,
            x_max=x2,
            column_count=1,
            row_type=row_type,
            vertical_separators=None,
            input_bbox=bbox if not is_button else None,
            action_bbox=bbox if is_button else None,
        )
        rows.append(r)
    return rows


def _post_process_rows_demo(
    container: FormContainer,
    rows: List[FormRow],
    layout_ocr: List[Dict[str, Any]],
) -> None:
    """Demo_mode: только привязка label сверху по OCR. Placeholder игнорируется."""
    for r in rows:
        if r.row_type != "FIELD_VERTICAL" or not r.input_bbox or len(r.input_bbox) < 4:
            continue
        ix_min, iy_min, ix_max, iy_max = r.input_bbox[0], r.input_bbox[1], r.input_bbox[2], r.input_bbox[3]
        label_above = [
            ob for ob in layout_ocr
            if len((ob.get("bbox") or [])) >= 4
            and ob["bbox"][3] <= iy_min + LABEL_ABOVE_INPUT_TOP_GAP_PX
            and _overlap_x(ob["bbox"], r.input_bbox) >= LABEL_ABOVE_OVERLAP_X_MIN
            and not _bbox_fully_inside(ob["bbox"], r.input_bbox)
        ]
        if label_above:
            best = min(label_above, key=lambda o: o["bbox"][1])
            if _is_valid_label_text((best.get("text") or "").strip()):
                r.label_bbox = list(best["bbox"])
                r.vertical_split_y = r.label_bbox[3]


def _bbox_has_visible_frame(image_path: Optional[str], bbox: List[float], frame_width_ratio: float = 0.8) -> bool:
    """
    Bbox имеет явную рамку (контур): горизонтальные края покрывают >= frame_width_ratio ширины.
    Используется для детекции TEXTAREA. При отсутствии image_path возвращает True (не отсекаем).
    """
    if not image_path or len(bbox) < 4:
        return True
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(image_path))
        if img is None:
            return True
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if x2 <= x1 or y2 <= y1:
            return True
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return True
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        w = edges.shape[1]
        if w < 2:
            return True
        # Нижняя и верхняя полоски: доля колонок с хотя бы одним ребром
        h = edges.shape[0]
        band = max(2, h // 6)
        top_band = edges[:band, :]
        bottom_band = edges[-band:, :] if h > band else edges
        top_span = np.sum(top_band > 0, axis=0)
        bottom_span = np.sum(bottom_band > 0, axis=0)
        cols_with_edge_top = np.sum(top_span > 0)
        cols_with_edge_bottom = np.sum(bottom_span > 0)
        return (cols_with_edge_top >= w * frame_width_ratio or cols_with_edge_bottom >= w * frame_width_ratio)
    except Exception:
        return True


def _bbox_has_low_color_variance(image_path: Optional[str], bbox: List[float], max_std: float = 45.0) -> bool:
    """Низкая дисперсия цвета в bbox (однородная заливка, напр. кнопка). При отсутствии image_path — False."""
    if not image_path or len(bbox) < 4:
        return False
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if x2 <= x1 or y2 <= y1:
            return False
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray)) <= max_std
    except Exception:
        return False


def _rows_from_visual_anchors(
    anchors: List[Dict[str, Any]],
    container_bbox: List[float],
    ocr_raw_for_action: List[Dict[str, Any]],
    image_path: Optional[str] = None,
) -> Tuple[List[FormRow], List[List[float]], List[int], float]:
    """
    Строки только из визуальных якорей. TEXTAREA: h >= max(80, 1.6*median), один bbox, явная рамка (контур > 80% ширины).
    Один input в строке: row прижат к input_bbox ± padding. Несколько без X-overlap → GRID + vertical_separators.
    """
    if len(container_bbox) < 4 or not anchors:
        return [], [], [], container_bbox[1]
    x1, y1, x2, y2 = container_bbox[0], container_bbox[1], container_bbox[2], container_bbox[3]
    form_start_y = y1
    skipped_bboxes: List[List[float]] = []
    rows: List[FormRow] = []
    textarea_indices: List[int] = []

    heights_no_textarea: List[float] = []
    for a in anchors:
        for bb in a["bboxes"]:
            if len(bb) >= 4:
                h = bb[3] - bb[1]
                if h < TEXTAREA_VISUAL_HEIGHT_PX:
                    heights_no_textarea.append(h)
    median_input_height = _median(heights_no_textarea) if heights_no_textarea else 40.0
    textarea_threshold = max(TEXTAREA_VISUAL_HEIGHT_PX, TEXTAREA_MEDIAN_RATIO * median_input_height)

    for idx, a in enumerate(anchors):
        bboxes = a["bboxes"]
        if a.get("from_ocr_header") and bboxes:
            bb = bboxes[0]
            row_y_min = max(y1, bb[1] - 4)
            row_y_max = min(y2, bb[3] + 4)
            if row_y_max <= row_y_min:
                continue
            r = FormRow(
                row_index=len(rows),
                y_min=row_y_min,
                y_max=row_y_max,
                x_min=float(a["x_min"]),
                x_max=float(a["x_max"]),
                column_count=1,
                row_type="HEADER",
                vertical_split_y=None,
                input_bbox=None,
                input_bboxes=None,
                vertical_separators=None,
                action_bbox=None,
            )
            rows.append(r)
            continue
        max_h = max(bb[3] - bb[1] for bb in bboxes)
        is_textarea_visual = max_h >= textarea_threshold

        if len(bboxes) == 1:
            iy_min, iy_max = bboxes[0][1], bboxes[0][3]
            row_y_min = max(y1, iy_min - min(ROW_INPUT_TOP_PAD, MAX_LABEL_HEIGHT_PX))
            row_y_max = max(iy_max, min(bboxes[0][3] + ROW_SNAP_PADDING_PX, y2))
        else:
            row_y_min = max(a["y_min"], y1)
            row_y_max = min(a["y_max"], y2)
        if row_y_max <= row_y_min:
            continue

        n_by_x = len(set(round((bb[0] + bb[2]) / 2, 0) for bb in bboxes))
        row_type: RowType = "FIELD_HORIZONTAL"
        column_count = 1
        vertical_split_y: Optional[float] = None
        input_bboxes_row: Optional[List[List[float]]] = None
        vertical_separators_row: Optional[List[float]] = None
        action_bbox_row: Optional[List[float]] = None

        container_h = y2 - y1
        container_w = x2 - x1
        min_textarea_w = max(TEXTAREA_MIN_WIDTH_PX, container_w * TEXTAREA_MIN_WIDTH_RATIO) if container_w > 0 else TEXTAREA_MIN_WIDTH_PX
        if is_textarea_visual and len(bboxes) == 1:
            bb = bboxes[0]
            bw = bb[2] - bb[0]
            bh = bb[3] - bb[1]
            aspect = bw / max(1e-9, bh)
            center_y = (bb[1] + bb[3]) / 2
            has_frame = _bbox_has_visible_frame(image_path, bb, frame_width_ratio=TEXTAREA_FRAME_RATIO)
            if (bw >= min_textarea_w and aspect <= TEXTAREA_MAX_ASPECT
                    and center_y > y1 + container_h * TEXTAREA_CENTER_Y_MIN_RATIO
                    and has_frame):
                row_type = "TEXTAREA"
                column_count = 1
                textarea_indices.append(len(rows))
        # ACTION только если: width ≥ 0.4 container, height ≤ 1.2×median, не в верхних 25%, (frame OR low_variance)
        action_zone_top = y1 + container_h * 0.25 if container_h > 0 else y1
        def _can_be_action(bb: List[float]) -> bool:
            if len(bb) < 4 or container_w <= 0 or median_input_height <= 0:
                return False
            bw, bh = bb[2] - bb[0], bb[3] - bb[1]
            center_y = (bb[1] + bb[3]) / 2
            if bw < container_w * 0.4 or bh > median_input_height * 1.2 or center_y <= action_zone_top:
                return False
            return _bbox_has_visible_frame(image_path, bb, frame_width_ratio=0.6) or _bbox_has_low_color_variance(image_path, bb)
        if row_type != "TEXTAREA":
            if len(bboxes) == 1 and _is_button_bbox(bboxes[0]) and _can_be_action(bboxes[0]):
                row_type = "ACTION"
                column_count = 1
            elif len(bboxes) == 1:
                for ob in ocr_raw_for_action:
                    txt = (ob.get("text") or "").strip().lower()
                    if any(w in txt for w in ACTION_WORDS):
                        b = ob.get("bbox", [])
                        if len(b) >= 4 and row_y_min <= (b[1] + b[3]) / 2 <= row_y_max and row_y_min <= b[1] and b[3] <= row_y_max:
                            if (b[0] + b[2]) / 2 >= x1 and (b[0] + b[2]) / 2 <= x2:
                                if _can_be_action(bboxes[0]):  # проверка по визуальному bbox строки
                                    row_type = "ACTION"
                                break
        elif len(bboxes) > 1:
            non_button_bboxes = [bb for bb in bboxes if not _is_button_bbox(bb)]
            button_bboxes = [bb for bb in bboxes if _is_button_bbox(bb)]
            if not non_button_bboxes and button_bboxes and _can_be_action(button_bboxes[0]):
                row_type = "ACTION"
                column_count = 1
                action_bbox_row = list(button_bboxes[0])
            else:
                use_bboxes = non_button_bboxes if non_button_bboxes else bboxes
                sorted_bboxes = sorted(use_bboxes, key=lambda bb: (bb[0] + bb[2]) / 2)
                max_overlap_ratio = 0.0
                for i in range(len(sorted_bboxes) - 1):
                    b1, b2 = sorted_bboxes[i], sorted_bboxes[i + 1]
                    w1, w2 = b1[2] - b1[0], b2[2] - b2[0]
                    overlap = min(b1[2], b2[2]) - max(b1[0], b2[0])
                    if w1 > 0 and w2 > 0:
                        max_overlap_ratio = max(max_overlap_ratio, overlap / min(w1, w2))
                if max_overlap_ratio < GRID_X_OVERLAP_THRESHOLD and len(sorted_bboxes) >= 1:
                    column_count = len(sorted_bboxes)
                    input_bboxes_row = [list(bb) for bb in sorted_bboxes]
                    vertical_separators_row = [
                        (sorted_bboxes[i][2] + sorted_bboxes[i + 1][0]) / 2.0
                        for i in range(len(sorted_bboxes) - 1)
                    ] if len(sorted_bboxes) >= 2 else None
                if button_bboxes:
                    action_bbox_row = list(button_bboxes[0])
                if max_overlap_ratio >= GRID_X_OVERLAP_THRESHOLD or len(sorted_bboxes) < 1:
                    n_by_x = len(sorted_bboxes)
                    column_count = n_by_x
        elif n_by_x > 1:
            column_count = n_by_x
        else:
            tops = sorted(bb[1] for bb in bboxes)
            bottoms = sorted(bb[3] for bb in bboxes)
            if len(bboxes) >= 2 and bottoms[0] < tops[-1] - 10:
                vertical_split_y = (bottoms[0] + tops[-1]) / 2.0
                row_type = "FIELD_HORIZONTAL"

        input_bbox_row: Optional[List[float]] = None
        if len(bboxes) == 1 and not _is_button_bbox(bboxes[0]):
            input_bbox_row = list(bboxes[0])
        elif len(bboxes) == 1:
            pass
        elif input_bboxes_row:
            input_bbox_row = [
                min(bb[0] for bb in input_bboxes_row),
                min(bb[1] for bb in input_bboxes_row),
                max(bb[2] for bb in input_bboxes_row),
                max(bb[3] for bb in input_bboxes_row),
            ]
        elif bboxes:
            input_bbox_row = [
                min(bb[0] for bb in bboxes),
                min(bb[1] for bb in bboxes),
                max(bb[2] for bb in bboxes),
                max(bb[3] for bb in bboxes),
            ]
        if input_bbox_row is not None and row_type == "ACTION":
            row_type = "FIELD_HORIZONTAL"
        r = FormRow(
            row_index=len(rows),
            y_min=row_y_min,
            y_max=row_y_max,
            x_min=float(a["x_min"]),
            x_max=float(a["x_max"]),
            column_count=column_count,
            row_type=row_type,
            vertical_split_y=vertical_split_y,
            input_bbox=input_bbox_row,
            input_bboxes=input_bboxes_row,
            vertical_separators=vertical_separators_row,
            action_bbox=action_bbox_row,
        )
        rows.append(r)
    return rows, skipped_bboxes, textarea_indices, form_start_y


def find_first_horizontal_line_below(
    image_path: str,
    container_bbox: List[float],
    input_top: float,
    input_bottom: float,
    input_left: float,
    input_right: float,
) -> Tuple[Optional[float], float]:
    """
    Первая устойчивая горизонтальная линия ниже input (underline, border, контраст).
    Возвращает (y_bottom_global, confidence). confidence < 1 при fallback на bbox.
    """
    import cv2
    import numpy as np
    if len(container_bbox) < 4:
        return None, 0.0
    img = cv2.imread(str(image_path))
    if img is None:
        return None, 0.0
    x1, y1, x2, y2 = int(container_bbox[0]), int(container_bbox[1]), int(container_bbox[2]), int(container_bbox[3])
    search_top = int(max(y1, input_bottom))
    search_bottom = min(int(container_bbox[3]), search_top + 400)
    if search_bottom <= search_top:
        return None, 0.0
    left = int(max(x1, input_left - 20))
    right = int(min(x2, input_right + 20))
    crop = img[search_top:search_bottom, left:right]
    if crop.size == 0:
        return None, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    row_sum = np.sum(edges, axis=1)
    h_crop = crop.shape[0]
    kernel = min(15, max(5, h_crop // 10))
    if kernel % 2 == 0:
        kernel += 1
    try:
        smooth = np.convolve(row_sum, np.ones(kernel) / kernel, mode="same")
    except Exception:
        smooth = row_sum
    thresh = float(np.max(smooth)) * 0.4 if hasattr(smooth, "__len__") and len(smooth) else 0
    for i in range(1, len(smooth) - 1):
        if float(smooth[i]) >= thresh and float(smooth[i]) >= float(smooth[i - 1]) and float(smooth[i]) >= float(smooth[i + 1]):
            y_global = float(search_top + i)
            return y_global, 0.85
    return None, 0.5


def _post_process_rows(
    image_path: str,
    container: FormContainer,
    rows: List[FormRow],
    layout_ocr: List[Dict[str, Any]],
    baseline: Dict[str, Any],
) -> None:
    """Уточнение высоты textarea по горизонтальной линии; FIELD_INPUT_ONLY / FIELD_VERTICAL / TEXT; декомпозиция label/input/helper. Используется только layout_ocr (после gating)."""
    if len(container.bbox) < 4 or not rows:
        return
    x1, y1, x2, y2 = container.bbox[0], container.bbox[1], container.bbox[2], container.bbox[3]
    w_container = x2 - x1
    container_cx = (x1 + x2) / 2
    median_font = float(baseline.get("median_font_height", 20.0))

    for r in rows:
        if r.row_type == "TEXTAREA":
            ib = getattr(r, "input_bbox", None)
            if ib and len(ib) >= 4:
                i_top, i_bottom, i_left, i_right = ib[1], ib[3], ib[0], ib[2]
            else:
                i_top, i_bottom, i_left, i_right = r.y_min, r.y_max, r.x_min, r.x_max
            y_bottom, conf = find_first_horizontal_line_below(
                image_path, container.bbox,
                i_top, i_bottom, i_left, i_right,
            )
            if y_bottom is not None and y_bottom > r.y_max:
                r.y_max = y_bottom
            else:
                # Нижняя граница только по CV: линия не найдена — использовать input_bbox[3]. OCR не участвует.
                if ib and len(ib) >= 4:
                    r.y_max = ib[3]
            r.height_confidence = conf

        ocr_in_row = [
            ob for ob in layout_ocr
            if len((ob.get("bbox") or [])) >= 4
            and r.y_min <= (ob["bbox"][1] + ob["bbox"][3]) / 2 <= r.y_max
            and r.x_min <= (ob["bbox"][0] + ob["bbox"][2]) / 2 <= r.x_max
        ]
        row_center_y = (r.y_min + r.y_max) / 2

        # HEADER не перезаписываем в TEXT — семантический классификатор оставляет HEADER.

        if r.row_type in ("FIELD", "FIELD_HORIZONTAL", "FIELD_VERTICAL", "FIELD_INPUT_ONLY"):
            input_bbox: List[float] = getattr(r, "input_bbox", None) or [r.x_min, r.y_min, r.x_max, r.y_max]
            ix_min, iy_min, ix_max, iy_max = input_bbox[0], input_bbox[1], input_bbox[2], input_bbox[3]
            row_y_min_new = max(y1, iy_min - min(ROW_INPUT_TOP_PAD, MAX_LABEL_HEIGHT_PX))
            r.y_min = min(r.y_min, row_y_min_new)
            r.y_min = max(r.y_min, y1, iy_min - MAX_LABEL_HEIGHT_PX)
            r.y_max = max(r.y_max, iy_max)

            def _is_placeholder_or_inside(ob: Dict[str, Any]) -> bool:
                b = ob.get("bbox") or []
                if len(b) < 4:
                    return True
                if _bbox_fully_inside(b, input_bbox):
                    return True
                if _is_placeholder_ocr(ob, input_bbox):
                    return True
                if _is_placeholder_vertical_zone(ob, input_bbox):
                    return True
                return False

            ocr_label_above = [
                ob for ob in layout_ocr
                if len((ob.get("bbox") or [])) >= 4
                and ob["bbox"][3] <= iy_min + LABEL_ABOVE_INPUT_TOP_GAP_PX
                and _overlap_x(ob["bbox"], input_bbox) >= LABEL_ABOVE_OVERLAP_X_MIN
                and not _is_placeholder_or_inside(ob)
            ]
            ocr_label_left = [
                ob for ob in layout_ocr
                if len((ob.get("bbox") or [])) >= 4
                and ob["bbox"][2] <= ix_min + LABEL_LEFT_INPUT_GAP_PX
                and _overlap_y(ob["bbox"], input_bbox) >= LABEL_LEFT_OVERLAP_Y_MIN
                and not _is_placeholder_or_inside(ob)
            ]
            ocr_label_right = [
                ob for ob in layout_ocr
                if len((ob.get("bbox") or [])) >= 4
                and ob["bbox"][0] >= ix_max - LABEL_LEFT_INPUT_GAP_PX
                and _overlap_y(ob["bbox"], input_bbox) >= LABEL_LEFT_OVERLAP_Y_MIN
                and not _is_placeholder_or_inside(ob)
            ]
            label_bbox = None
            right_label_bbox = None
            if ocr_label_above and r.row_type != "TEXTAREA":
                best = min(ocr_label_above, key=lambda o: o["bbox"][1])
                if _is_valid_label_text((best.get("text") or "").strip()):
                    label_bbox = list(best["bbox"])
                    r.row_type = "FIELD_VERTICAL"
                    r.vertical_separators = None
                    r.column_count = 1
                    r.vertical_split_y = label_bbox[3]
            elif ocr_label_left and r.row_type != "TEXTAREA":
                best = max(ocr_label_left, key=lambda o: o["bbox"][2])
                if _is_valid_label_text((best.get("text") or "").strip()) and (best["bbox"][2] - best["bbox"][0]) >= 20:
                    label_bbox = list(best["bbox"])
                    r.row_type = "FIELD_HORIZONTAL"
            elif ocr_label_right and r.row_type != "TEXTAREA":
                best = min(ocr_label_right, key=lambda o: o["bbox"][0])
                if _is_valid_label_text((best.get("text") or "").strip()) and (best["bbox"][2] - best["bbox"][0]) >= 20:
                    right_label_bbox = list(best["bbox"])
                    r.row_type = "FIELD_HORIZONTAL"
            else:
                r.row_type = "FIELD_INPUT_ONLY"
            input_h = iy_max - iy_min
            helper_top_min = iy_max
            helper_top_max = iy_max + input_h * HELPER_BELOW_INPUT_RATIO if input_h > 0 else iy_max + 40
            ocr_below = [
                ob for ob in ocr_in_row
                if (ob["bbox"][1] + ob["bbox"][3]) / 2 >= row_center_y
                and ob["bbox"][3] <= r.y_max
                and helper_top_min <= ob["bbox"][1] <= helper_top_max
                and not _is_placeholder_ocr(ob, input_bbox)
            ]
            helper_bbox = None
            for ob in ocr_below:
                if _is_placeholder_ocr(ob, input_bbox):
                    continue
                ob_h = ob["bbox"][3] - ob["bbox"][1]
                if ob_h >= median_font * HELPER_FONT_MAX_RATIO:
                    continue
                ob_cx = (ob["bbox"][0] + ob["bbox"][2]) / 2
                if w_container > 0 and abs(ob_cx - container_cx) / w_container < BUTTON_CENTER_TOLERANCE:
                    continue
                txt = (ob.get("text") or "").strip()
                if not txt or not _is_valid_label_text(txt):
                    continue
                helper_bbox = list(ob["bbox"])
                break
            r.label_bbox = label_bbox
            r.right_label_bbox = right_label_bbox
            r.input_bbox = input_bbox
            r.helper_bbox = helper_bbox


def _ocr_heights(ocr_inside: List[Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    for ob in ocr_inside:
        b = ob.get("bbox", [])
        if len(b) >= 4:
            out.append(b[3] - b[1])
    return out


def _build_rows_inside_container(
    image_path: str,
    container: FormContainer,
    ocr_inside: Optional[List[Dict[str, Any]]] = None,
    ocr_raw_for_action: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[FormRow], List[List[float]], List[int], float]:
    """
    Строки на основе layout_ocr (после gating). Кластеризация и median только по layout_ocr.
    ACTION-детекция — по ocr_raw_for_action (полный OCR), если передан.
    """
    import cv2
    import numpy as np

    ocr_inside = ocr_inside or []
    action_ocr = ocr_raw_for_action if ocr_raw_for_action is not None else ocr_inside
    bbox = container.bbox
    if len(bbox) < 4:
        return [], [], [], bbox[1]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    form_start_y = y1

    # Высота OCR как proxy font_size
    ocr_heights = _ocr_heights(ocr_inside)
    median_font = _median(ocr_heights) if ocr_heights else 20.0

    # Header формы: OCR с font_size > median выше первой "input"-подобной зоны
    # Собираем Y-позиции OCR по вертикали
    ocr_sorted: List[Tuple[float, float, float, Dict]] = []  # (y_center, y_bottom, height, ob)
    for ob in ocr_inside:
        b = ob.get("bbox", [])
        if len(b) < 4:
            continue
        if b[0] < x1 or b[2] > x2 or b[1] < y1 or b[3] > y2:
            continue
        h = b[3] - b[1]
        yc = (b[1] + b[3]) / 2
        ocr_sorted.append((yc, b[3], h, ob))
    ocr_sorted.sort(key=lambda t: t[0])

    # Первая строка с "полем" — условно OCR с небольшим кол-вом символов (поле) или следующая после label
    # Header: все OCR-блоки с h > median_font выше первого блока с коротким текстом (потенциальный label/input)
    first_input_y = y2
    for _, yb, h, ob in ocr_sorted:
        txt = (ob.get("text") or "").strip()
        if len(txt) <= 2 or (len(txt) < 30 and h <= median_font * 1.5):
            first_input_y = min(first_input_y, yb)
            break

    skipped_bboxes: List[List[float]] = []
    header_bottom = y1
    for yc, yb, h, ob in ocr_sorted:
        if yb > first_input_y:
            break
        if h > median_font * 1.2:
            b = ob.get("bbox", [])
            if len(b) >= 4:
                skipped_bboxes.append(list(b))
            header_bottom = max(header_bottom, yb)
    if skipped_bboxes:
        form_start_y = header_bottom + 4.0

    # Логические строки: группируем OCR по вертикальной близости (одна строка = label + поле зона)
    if not ocr_sorted:
        # Fallback: edge-based как раньше, но с типами
        img = cv2.imread(str(image_path))
        if img is None:
            return [], skipped_bboxes, [], form_start_y
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            return [], skipped_bboxes, [], form_start_y
        h_crop, w_crop = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        row_edges = edges.sum(axis=1)
        kernel = min(5, max(3, h_crop // 30))
        if kernel % 2 == 0:
            kernel += 1
        try:
            row_edges_smooth = np.convolve(row_edges, np.ones(kernel) / kernel, mode="same")
        except Exception:
            row_edges_smooth = row_edges
        threshold = float(np.max(row_edges_smooth)) * 0.15 if hasattr(row_edges_smooth, "__len__") and len(row_edges_smooth) else 0
        n = len(row_edges_smooth) if hasattr(row_edges_smooth, "__len__") else 0
        in_row = False
        row_starts: List[int] = []
        row_ends: List[int] = []
        for i in range(1, n):
            val = float(row_edges_smooth[i]) if i < len(row_edges_smooth) else 0
            if not in_row and val > threshold:
                in_row = True
                row_starts.append(i)
            elif in_row and val <= threshold:
                in_row = False
                row_ends.append(i)
        if in_row:
            row_ends.append(n)
        rows = []
        for idx in range(min(len(row_starts), len(row_ends))):
            a, b = row_starts[idx], row_ends[idx]
            if b - a < MIN_ROW_HEIGHT:
                continue
            ry_min = y1 + a
            ry_max = y1 + b
            if ry_min < form_start_y:
                continue
            rows.append(FormRow(
                row_index=len(rows),
                y_min=ry_min,
                y_max=ry_max,
                x_min=x1,
                x_max=x2,
                column_count=1,
                row_type="FIELD",
                height_mode="adaptive",
            ))
        if rows:
            heights = [r.y_max - r.y_min for r in rows]
            median_h = _median(heights)
            textarea_indices: List[int] = []
            for r in rows:
                if (r.y_max - r.y_min) >= median_h * TEXTAREA_HEIGHT_MEDIAN_K:
                    r.row_type = "TEXTAREA"
                    textarea_indices.append(r.row_index)
            for r in rows:
                if r.row_type == "FIELD":
                    for ob in action_ocr:
                        txt = (ob.get("text") or "").strip().lower()
                        if any(w in txt for w in ACTION_WORDS):
                            b = ob.get("bbox", [])
                            if len(b) >= 4 and r.y_min <= (b[1]+b[3])/2 <= r.y_max:
                                r.row_type = "ACTION"
                                break
            return rows, skipped_bboxes, textarea_indices, form_start_y
        return [], skipped_bboxes, [], form_start_y

    # Группируем OCR в логические строки по Y (близкие по вертикали = одна строка)
    row_groups: List[List[Tuple[float, float, float, Dict]]] = []
    tol = median_font * 2.0 if median_font > 0 else 24.0
    for t in ocr_sorted:
        yc, yb, h, ob = t
        if yc < form_start_y:
            continue
        placed = False
        for g in row_groups:
            g_y_center = sum(x[0] for x in g) / len(g)
            if abs(yc - g_y_center) <= tol:
                g.append(t)
                placed = True
                break
        if not placed:
            row_groups.append([t])

    rows = []
    textarea_indices: List[int] = []
    for idx, g in enumerate(row_groups):
        if not g:
            continue
        ys = [x[1] for x in g]
        ytops = [ob["bbox"][1] for _, _, _, ob in g if len(ob.get("bbox", [])) >= 4]
        ybottoms = [ob["bbox"][3] for _, _, _, ob in g if len(ob.get("bbox", [])) >= 4]
        ry_min = (min(ytops) - 6) if ytops else (min(ys) - 6)
        ry_max = (max(ybottoms) + 6) if ybottoms else (max(ys) + 6)
        row_h = ry_max - ry_min
        if row_h < MIN_ROW_HEIGHT:
            continue
        # Не размножаем по первой высоте — каждая строка своя высота (adaptive)
        texts = [((ob.get("text") or "").strip().lower(), ob) for _, _, _, ob in g]
        is_action = any(any(w in t for w in ACTION_WORDS) for t, _ in texts)
        is_header = any(h > median_font * 1.2 for _, _, h, _ in g)
        row_type: RowType = "FIELD"
        if is_action:
            row_type = "ACTION"
        elif is_header and idx == 0:
            row_type = "HEADER"
        else:
            # TEXTAREA по высоте позже, после сбора всех row heights
            pass

        rows.append(FormRow(
            row_index=len(rows),
            y_min=ry_min,
            y_max=ry_max,
            x_min=x1,
            x_max=x2,
            column_count=1,
            row_type=row_type,
            height_mode="adaptive",
        ))

    # TEXTAREA: высота > median(row_height) * k
    if rows:
        heights = [r.y_max - r.y_min for r in rows]
        median_h = _median(heights)
        for r in rows:
            if r.row_type != "FIELD":
                continue
            if (r.y_max - r.y_min) >= median_h * TEXTAREA_HEIGHT_MEDIAN_K:
                r.row_type = "TEXTAREA"
                textarea_indices.append(r.row_index)
        # Кнопка внизу не считается input-row: ACTION уже выставлен по тексту
        for r in rows:
            if r.row_type == "FIELD":
                for ob in action_ocr:
                    txt = (ob.get("text") or "").strip().lower()
                    if any(w in txt for w in ACTION_WORDS):
                        b = ob.get("bbox", [])
                        if len(b) >= 4 and r.y_min <= (b[1] + b[3]) / 2 <= r.y_max:
                            r.row_type = "ACTION"
                            break

    return rows, skipped_bboxes, textarea_indices, form_start_y


def _infer_layout_type(
    rows: List[FormRow],
    ocr_inside: List[Dict[str, Any]],
    container: FormContainer,
) -> str:
    """Grid только если все строки FIELD_HORIZONTAL и хотя бы одна с column_count > 1."""
    if not rows or len(container.bbox) < 4:
        return "vertical"
    field_types = {"FIELD", "FIELD_HORIZONTAL", "FIELD_VERTICAL", "FIELD_INPUT_ONLY", "TEXTAREA"}
    if not all(r.row_type == "FIELD_HORIZONTAL" for r in rows if r.row_type in field_types):
        return "vertical"
    if not any(r.column_count > 1 for r in rows):
        return "vertical"
    return "grid"


def _build_columns_inside_container(
    image_path: str,
    container: FormContainer,
    rows: List[FormRow],
    ocr_inside: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[FormColumn], List[Tuple[float, float]]]:
    """Колонки только по X-центрам визуальных контуров (поля). OCR не создаёт колонки."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None or not rows:
        return [], []
    bbox = container.bbox
    if len(bbox) < 4:
        return [], []
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return [], []
    w_crop = crop.shape[1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    x_centers: List[float] = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 200:
            continue
        rx, _, rw, _ = cv2.boundingRect(c)
        x_centers.append(x1 + rx + rw / 2.0)

    if not x_centers:
        columns = [FormColumn(col_index=0, x_min=float(x1), x_max=float(x2))]
        return columns, [(float(x1), float(x2))]

    x_centers.sort()
    tol = w_crop * X_CLUSTER_TOLERANCE_RATIO
    clusters: List[List[float]] = []
    for x in x_centers:
        placed = False
        for cl in clusters:
            if abs(x - statistics.median(cl)) <= tol:
                cl.append(x)
                placed = True
                break
        if not placed:
            clusters.append([x])

    columns = []
    boundaries = []
    if len(clusters) >= 2:
        for i, cl in enumerate(clusters):
            cx = statistics.median(cl)
            half = w_crop / max(len(clusters), 1) / 2
            cx1 = max(float(x1), cx - half)
            cx2 = min(float(x2), cx + half)
            columns.append(FormColumn(col_index=i, x_min=cx1, x_max=cx2))
            boundaries.append((cx1, cx2))
        boundaries.sort(key=lambda p: p[0])
    else:
        columns.append(FormColumn(col_index=0, x_min=float(x1), x_max=float(x2)))
        boundaries.append((float(x1), float(x2)))
    return columns, boundaries


def build_form_inner_layout(
    image_path: str,
    container: FormContainer,
    ocr_inside: Optional[List[Dict[str, Any]]] = None,
    visual_candidates: Optional[List[List[float]]] = None,
    demo_mode: bool = False,
) -> Tuple[Optional[FormSkeleton], Dict[str, Any]]:
    """
    FormInnerLayout. Если передан non-empty visual_candidates — строки строятся только из CV (якоря по Y);
    OCR не задаёт границы строк и не участвует в определении textarea. OCR используется только для label/helper/placeholder.
    demo_mode: одна строка = один input_bbox, FIELD_VERTICAL, grid отключён, placeholder игнорируется.
    """
    ocr_inside = ocr_inside or []
    layout_ocr, baseline, header_bboxes = normalize_ocr_for_layout(ocr_inside, container.bbox, image_path)
    if demo_mode and visual_candidates and len(container.bbox) >= 4:
        rows = _build_rows_demo_mode(container, visual_candidates)
        skipped_bboxes, textarea_row_indices, form_start_y = [], [], container.bbox[1]
        _post_process_rows_demo(container, rows, layout_ocr)
        _apply_row_invariants(rows, container.bbox)
        from src.infrastructure.atoms_v2.experimental_v2.form_invariants_patch import enforce_form_row_invariants
        enforce_form_row_invariants(rows, layout_ocr, container.bbox, baseline)
        from src.infrastructure.atoms_v2.experimental_v2.row_semantic_classifier import classify_rows
        from src.infrastructure.atoms_v2.experimental_v2.form_invariants_patch import enforce_field_has_input_bbox
        classify_rows(rows, layout_ocr, container.bbox, baseline, image_path=image_path)
        enforce_field_has_input_bbox(rows)
        _remove_orphan_field_rows(rows)
        rows_debug = []
        layout_type = "vertical"
        columns = [FormColumn(col_index=0, x_min=container.bbox[0], x_max=container.bbox[2])]
        column_boundaries = [(container.bbox[0], container.bbox[2])]
        for r in rows:
            r.column_count = 1
            r.vertical_separators = None
        skeleton = FormSkeleton(
            form_region=container,
            rows=rows,
            columns=columns,
            column_boundaries=column_boundaries,
            layout_type=layout_type,
        )
        diag = {"n_rows": len(rows), "n_columns": 1, "layout_type": layout_type, "skipped_bboxes": skipped_bboxes, "textarea_row_indices": textarea_row_indices, "rows_debug": rows_debug}
        return skeleton, diag
    if visual_candidates and len(container.bbox) >= 4:
        all_visual = list(visual_candidates)
        median_vis = _median([b[3] - b[1] for b in visual_candidates if len(b) >= 4]) if visual_candidates else 50.0
        tall = _tall_contours_inside_container(image_path, container.bbox, median_input_height=median_vis)
        for t in tall:
            if not any(_y_overlap_or_near(t, v) for v in all_visual):
                all_visual.append(t)
        anchors = collect_field_row_anchors(
            all_visual, container.bbox, image_path,
            layout_ocr=layout_ocr, baseline=baseline,
        )
        if anchors:
            rows, skipped_bboxes, textarea_row_indices, form_start_y = _rows_from_visual_anchors(
                anchors, container.bbox, ocr_raw_for_action=ocr_inside, image_path=image_path,
            )
            if rows and len(container.bbox) >= 4:
                _normalize_row_overlaps(rows, container.bbox[1], container.bbox[3])
        else:
            rows, skipped_bboxes, textarea_row_indices, form_start_y = [], [], [], container.bbox[1]
    else:
        rows, skipped_bboxes, textarea_row_indices, form_start_y = _build_rows_inside_container(
            image_path, container, layout_ocr, ocr_raw_for_action=ocr_inside
        )
    for b in header_bboxes:
        if len(b) >= 4 and b not in skipped_bboxes:
            skipped_bboxes.append(b)
    if header_bboxes:
        form_start_y = max(form_start_y, max(b[3] for b in header_bboxes if len(b) >= 4) + 4.0)
    _post_process_rows(image_path, container, rows, layout_ocr, baseline)
    _apply_row_invariants(rows, container.bbox)
    from src.infrastructure.atoms_v2.experimental_v2.form_invariants_patch import enforce_form_row_invariants
    enforce_form_row_invariants(rows, layout_ocr, container.bbox, baseline)
    from src.infrastructure.atoms_v2.experimental_v2.row_semantic_classifier import classify_rows
    from src.infrastructure.atoms_v2.experimental_v2.form_invariants_patch import enforce_field_has_input_bbox
    classify_rows(rows, layout_ocr, container.bbox, baseline, image_path=image_path)
    enforce_field_has_input_bbox(rows)
    _remove_orphan_field_rows(rows)

    rows_debug: List[Dict[str, Any]] = []
    for r in rows:
        ib = r.input_bbox or [r.x_min, r.y_min, r.x_max, r.y_max]
        ix_min, ix_max = ib[0], ib[2]
        ocr_label_only: List[List[float]] = []
        for ob in layout_ocr:
            b = ob.get("bbox") or []
            if len(b) < 4:
                continue
            if b[2] < ix_min or b[0] > ix_max:
                continue
            if b[3] <= r.y_min + 25 or b[2] <= ix_min + 10:
                ocr_label_only.append(list(b))
        rows_debug.append({
            "row_index": r.row_index,
            "row_y_from_visual": (float(r.y_min), float(r.y_max)),
            "ocr_considered_for_label_only": ocr_label_only,
        })

    if not rows:
        n_default = max(2, int((container.bbox[3] - form_start_y) // 50))
        step = (container.bbox[3] - form_start_y) / n_default
        for i in range(n_default):
            rows.append(FormRow(
                row_index=i,
                y_min=form_start_y + i * step,
                y_max=form_start_y + (i + 1) * step,
                x_min=container.bbox[0],
                x_max=container.bbox[2],
                column_count=1,
                row_type="FIELD_HORIZONTAL",
                height_mode="fixed",
            ))

    layout_type = _infer_layout_type(rows, ocr_inside, container)

    def _grid_allowed(rows_list: List[FormRow]) -> bool:
        rows_with_cols = [r for r in rows_list if getattr(r, "input_bboxes", None) and len(r.input_bboxes) > 1]
        if len(rows_with_cols) < 2:
            return False
        for r in rows_with_cols:
            for bb in r.input_bboxes:
                if (bb[2] - bb[0]) < GRID_MIN_INPUT_WIDTH:
                    return False
            for i in range(len(r.input_bboxes) - 1):
                if r.input_bboxes[i + 1][0] - r.input_bboxes[i][2] < GRID_MIN_GAP_X:
                    return False
        n_cols = len(rows_with_cols[0].input_bboxes)
        container_w = container.bbox[2] - container.bbox[0] if len(container.bbox) >= 4 else 0
        min_x_dist = container_w * GRID_MIN_X_DISTANCE_RATIO if container_w > 0 else GRID_MIN_GAP_X
        for r in rows_with_cols:
            for i in range(len(r.input_bboxes) - 1):
                c1 = (r.input_bboxes[i][0] + r.input_bboxes[i][2]) / 2
                c2 = (r.input_bboxes[i + 1][0] + r.input_bboxes[i + 1][2]) / 2
                if abs(c2 - c1) < min_x_dist:
                    return False
        for col_idx in range(n_cols):
            centers = [(r.input_bboxes[col_idx][0] + r.input_bboxes[col_idx][2]) / 2 for r in rows_with_cols]
            med = _median(centers)
            if any(abs(c - med) > GRID_COLUMN_X_TOLERANCE_PX for c in centers):
                return False
        return True

    if layout_type == "vertical" or not _grid_allowed(rows):
        if layout_type == "grid":
            layout_type = "vertical"
        columns = [FormColumn(col_index=0, x_min=container.bbox[0], x_max=container.bbox[2])]
        column_boundaries = [(container.bbox[0], container.bbox[2])]
    else:
        grid_row = next((r for r in rows if getattr(r, "input_bboxes", None) and len(r.input_bboxes) > 1), None)
        if grid_row and grid_row.input_bboxes:
            column_boundaries = [(bb[0], bb[2]) for bb in grid_row.input_bboxes]
            columns = [
                FormColumn(col_index=i, x_min=bb[0], x_max=bb[2])
                for i, bb in enumerate(grid_row.input_bboxes)
            ]
        else:
            columns, column_boundaries = _build_columns_inside_container(image_path, container, rows, ocr_inside)
        for r in rows:
            if r.row_type == "TEXTAREA":
                r.column_count = 1
            else:
                r.column_count = len(columns)

    skeleton = FormSkeleton(
        form_region=container,
        rows=rows,
        columns=columns if columns else None,
        column_boundaries=column_boundaries if column_boundaries else [(container.bbox[0], container.bbox[2])],
        layout_type=layout_type,
    )
    diag = {
        "n_rows": len(rows),
        "n_columns": len(columns),
        "layout_type": layout_type,
        "skipped_bboxes": skipped_bboxes,
        "textarea_row_indices": textarea_row_indices,
        "rows_debug": rows_debug,
    }
    return skeleton, diag


def visualize_rows(
    image_path: str,
    container: FormContainer,
    rows: List[FormRow],
    output_path: str,
) -> None:
    """Сохраняет rows.png."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    from src.infrastructure.debug_draw import putText_visible, rectangle_visible

    out = img.copy()
    for r in rows:
        y1, y2 = int(r.y_min), int(r.y_max)
        rectangle_visible(out, (int(r.x_min), y1), (int(r.x_max), y2), (0, 180, 180), 1)
        putText_visible(out, "R%d" % r.row_index, (int(r.x_min) + 2, y1 + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), (0, 0, 0), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_inner_layout: saved %s", output_path)


def visualize_rows_with_types(
    image_path: str,
    rows: List[FormRow],
    output_path: str,
) -> None:
    """Сохраняет rows_with_types.png — цвет по row_type."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    colors = {
        "HEADER": (0, 140, 200), "TEXT": (0, 140, 200),
        "FIELD": (0, 180, 180), "FIELD_HORIZONTAL": (0, 180, 180), "FIELD_VERTICAL": (0, 180, 140), "FIELD_INPUT_ONLY": (0, 180, 120),
        "TEXTAREA": (180, 0, 180), "ACTION": (0, 140, 200), "SPACER": (80, 80, 80),
    }
    from src.infrastructure.debug_draw import putText_visible, rectangle_visible

    for r in rows:
        y1, y2 = int(r.y_min), int(r.y_max)
        color = colors.get(r.row_type, (100, 100, 100))
        rectangle_visible(out, (int(r.x_min), y1), (int(r.x_max), y2), color, 1)
        putText_visible(out, "%s R%d" % (r.row_type, r.row_index), (int(r.x_min) + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), (0, 0, 0), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_inner_layout: saved %s", output_path)


def visualize_skipped_rows(
    image_path: str,
    skipped_bboxes: List[List[float]],
    output_path: str,
) -> None:
    """Сохраняет skipped_rows.png — зоны header, вынесенные из RowDetector."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    from src.infrastructure.debug_draw import putText_visible, rectangle_visible

    for b in skipped_bboxes:
        if len(b) >= 4:
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            rectangle_visible(out, (x1, y1), (x2, y2), (0, 100, 180), 2)
            putText_visible(out, "skipped", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), (0, 0, 0), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_inner_layout: saved %s", output_path)


def visualize_textarea_rows(
    image_path: str,
    rows: List[FormRow],
    textarea_row_indices: List[int],
    output_path: str,
) -> None:
    """Сохраняет textarea_rows.png — только TEXTAREA-строки."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    from src.infrastructure.debug_draw import putText_visible, rectangle_visible

    for r in rows:
        if r.row_index in textarea_row_indices or r.row_type == "TEXTAREA":
            y1, y2 = int(r.y_min), int(r.y_max)
            rectangle_visible(out, (int(r.x_min), y1), (int(r.x_max), y2), (180, 0, 180), 2)
            putText_visible(out, "TEXTAREA R%d" % r.row_index, (int(r.x_min) + 2, y1 + 16),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), (0, 0, 0), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_inner_layout: saved %s", output_path)


def visualize_rows_debug(
    image_path: str,
    container: FormContainer,
    rows_debug: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Debug overlay: row_y_from_visual (границы строки из CV) и ocr_considered_for_label_only (OCR не расширяет row).
    """
    import cv2
    img = cv2.imread(str(image_path))
    if img is None or len(container.bbox) < 4:
        return
    from src.infrastructure.debug_draw import putText_visible, rectangle_visible

    out = img.copy()
    x1, y1, x2, y2 = int(container.bbox[0]), int(container.bbox[1]), int(container.bbox[2]), int(container.bbox[3])
    rectangle_visible(out, (x1, y1), (x2, y2), (80, 80, 80), 1)
    for d in rows_debug:
        ry = d.get("row_y_from_visual")
        if ry and len(ry) >= 2:
            rymin, rymax = int(ry[0]), int(ry[1])
            rectangle_visible(out, (x1, rymin), (x2, rymax), (0, 180, 180), 1)
            putText_visible(out, "row_y_from_visual R%d" % d.get("row_index", -1), (x1 + 2, rymin + 14),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), (0, 0, 0), 1)
        for b in d.get("ocr_considered_for_label_only") or []:
            if len(b) >= 4:
                bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                rectangle_visible(out, (bx1, by1), (bx2, by2), (0, 180, 0), 1)
                putText_visible(out, "ocr_label_only", (bx1, by1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), (0, 0, 0), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_inner_layout: saved debug overlay %s", output_path)
