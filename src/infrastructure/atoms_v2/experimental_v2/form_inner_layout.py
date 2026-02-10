"""
Уровень 1 — FormInnerLayout: строки и колонки строго внутри FormContainer.bbox.

Типы строк: FIELD_HORIZONTAL, FIELD_VERTICAL, FIELD_INPUT_ONLY, TEXTAREA, ACTION, TEXT.
Высота textarea — по первой устойчивой горизонтальной линии (underline/border); fallback на bbox.
Внутренняя декомпозиция: label_bbox, input_bbox, helper_bbox по визуальным кандидатам и OCR.
"""

from __future__ import annotations

import logging
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
HEADER_ZONE_TOP_RATIO = 0.12
FONT_HEADER_RATIO = 1.4
FONT_HINT_RATIO = 0.6
HELPER_FONT_MAX_RATIO = 0.85
BUTTON_CENTER_TOLERANCE = 0.25
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


TEXTAREA_VISUAL_HEIGHT_PX = 80
Y_OVERLAP_GAP_PX = 40


def _bbox_in_container(b: List[float], c: List[float]) -> bool:
    if len(b) < 4 or len(c) < 4:
        return False
    return c[0] <= b[0] and b[2] <= c[2] and c[1] <= b[1] and b[3] <= c[3]


def _y_overlap_or_near(b1: List[float], b2: List[float], gap: float = Y_OVERLAP_GAP_PX) -> bool:
    if len(b1) < 4 or len(b2) < 4:
        return False
    if b1[3] < b2[1] - gap or b2[3] < b1[1] - gap:
        return False
    return True


def collect_field_row_anchors(
    visual_candidates: List[List[float]],
    container_bbox: List[float],
) -> List[Dict[str, Any]]:
    """
    Якоря строк только из bbox визуальных полей. Кластеризация по Y-overlap.
    Tall (h>=TEXTAREA_VISUAL_HEIGHT_PX) не сливаются с обычными полями.
    """
    if len(container_bbox) < 4 or not visual_candidates:
        return []
    inside = [b for b in visual_candidates if len(b) >= 4 and _bbox_in_container(b, container_bbox)]
    if not inside:
        return []
    inside = sorted(inside, key=lambda b: (b[1], b[0]))

    def _is_tall(bb: List[float]) -> bool:
        return len(bb) >= 4 and (bb[3] - bb[1]) >= TEXTAREA_VISUAL_HEIGHT_PX

    clusters: List[List[List[float]]] = []
    for b in inside:
        placed = False
        b_tall = _is_tall(b)
        for cl in clusters:
            for c in cl:
                if not _y_overlap_or_near(b, c):
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

    anchors: List[Dict[str, Any]] = []
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
    anchors.sort(key=lambda a: a["y_min"])
    return anchors


def _tall_contours_inside_container(image_path: str, container_bbox: List[float]) -> List[List[float]]:
    """Визуальные контуры высотой >= TEXTAREA_VISUAL_HEIGHT_PX (textarea-like). Не участвует OCR."""
    import cv2
    if len(container_bbox) < 4:
        return []
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    x1, y1, x2, y2 = int(container_bbox[0]), int(container_bbox[1]), int(container_bbox[2]), int(container_bbox[3])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[List[float]] = []
    for c in contours:
        if cv2.contourArea(c) < 800:
            continue
        rx, ry, rw, rh = cv2.boundingRect(c)
        if rh < TEXTAREA_VISUAL_HEIGHT_PX:
            continue
        if rw < 60:
            continue
        out.append([float(x1 + rx), float(y1 + ry), float(x1 + rx + rw), float(y1 + ry + rh)])
    return out


def _rows_from_visual_anchors(
    anchors: List[Dict[str, Any]],
    container_bbox: List[float],
    ocr_raw_for_action: List[Dict[str, Any]],
) -> Tuple[List[FormRow], List[List[float]], List[int], float]:
    """
    Строки только из визуальных якорей. Границы row = min(top)…max(bottom) bbox кластера.
    OCR не расширяет row, не задаёт высоту. TEXTAREA только по высоте bbox (>=80px). ACTION по OCR в зоне строки.
    """
    if len(container_bbox) < 4 or not anchors:
        return [], [], [], container_bbox[1]
    x1, y1, x2, y2 = container_bbox[0], container_bbox[1], container_bbox[2], container_bbox[3]
    form_start_y = y1
    skipped_bboxes: List[List[float]] = []
    rows: List[FormRow] = []
    textarea_indices: List[int] = []

    for idx, a in enumerate(anchors):
        bboxes = a["bboxes"]
        row_y_min = max(a["y_min"], y1)
        row_y_max = min(a["y_max"], y2)
        if row_y_max <= row_y_min:
            continue
        n_by_x = len(set(round((bb[0] + bb[2]) / 2, 0) for bb in bboxes))
        max_h = max(bb[3] - bb[1] for bb in bboxes)
        is_textarea_visual = max_h >= TEXTAREA_VISUAL_HEIGHT_PX

        row_type: RowType = "FIELD_HORIZONTAL"
        column_count = 1
        vertical_split_y: Optional[float] = None

        if is_textarea_visual:
            row_type = "TEXTAREA"
            column_count = 1
            textarea_indices.append(len(rows))
        elif len(bboxes) == 1:
            for ob in ocr_raw_for_action:
                txt = (ob.get("text") or "").strip().lower()
                if any(w in txt for w in ACTION_WORDS):
                    b = ob.get("bbox", [])
                    if len(b) >= 4 and row_y_min <= (b[1] + b[3]) / 2 <= row_y_max and row_y_min <= b[1] and b[3] <= row_y_max:
                        if (b[0] + b[2]) / 2 >= x1 and (b[0] + b[2]) / 2 <= x2:
                            row_type = "ACTION"
                            break
        elif n_by_x > 1:
            column_count = n_by_x
        elif len(bboxes) >= 2:
            tops = sorted(bb[1] for bb in bboxes)
            bottoms = sorted(bb[3] for bb in bboxes)
            if bottoms[0] < tops[-1] - 10:
                vertical_split_y = (bottoms[0] + tops[-1]) / 2.0
                row_type = "FIELD_VERTICAL"

        input_bbox_row: Optional[List[float]] = None
        if len(bboxes) == 1:
            input_bbox_row = list(bboxes[0])
        elif bboxes:
            input_bbox_row = [
                min(bb[0] for bb in bboxes),
                min(bb[1] for bb in bboxes),
                max(bb[2] for bb in bboxes),
                max(bb[3] for bb in bboxes),
            ]
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
            y_bottom, conf = find_first_horizontal_line_below(
                image_path, container.bbox,
                r.y_min, r.y_max, r.x_min, r.x_max,
            )
            if y_bottom is not None and y_bottom > r.y_max:
                r.y_max = y_bottom
            r.height_confidence = conf

        ocr_in_row = [
            ob for ob in layout_ocr
            if len((ob.get("bbox") or [])) >= 4
            and r.y_min <= (ob["bbox"][1] + ob["bbox"][3]) / 2 <= r.y_max
            and r.x_min <= (ob["bbox"][0] + ob["bbox"][2]) / 2 <= r.x_max
        ]
        row_center_y = (r.y_min + r.y_max) / 2
        LABEL_ABOVE_GAP = 25

        if r.row_type == "HEADER":
            r.row_type = "TEXT"

        if r.row_type in ("FIELD", "FIELD_HORIZONTAL", "FIELD_VERTICAL", "FIELD_INPUT_ONLY"):
            input_bbox: List[float] = getattr(r, "input_bbox", None) or [r.x_min, r.y_min, r.x_max, r.y_max]
            ix_min, iy_min, ix_max, iy_max = input_bbox[0], input_bbox[1], input_bbox[2], input_bbox[3]
            ocr_above_row = [
                ob for ob in layout_ocr
                if len((ob.get("bbox") or [])) >= 4
                and ob["bbox"][3] <= r.y_min + LABEL_ABOVE_GAP
                and not (ob["bbox"][2] < ix_min or ob["bbox"][0] > ix_max)
            ]
            ocr_left_of_input = [
                ob for ob in layout_ocr
                if len((ob.get("bbox") or [])) >= 4
                and ob["bbox"][2] <= ix_min + 10
                and (ob["bbox"][1] + ob["bbox"][3]) / 2 >= iy_min - 5
                and (ob["bbox"][1] + ob["bbox"][3]) / 2 <= iy_max + 5
            ]
            label_bbox = None
            if ocr_above_row:
                best = min(ocr_above_row, key=lambda o: o["bbox"][1])
                lb = best["bbox"]
                if lb[2] - lb[0] >= (ix_max - ix_min) * X_OVERLAP_THRESHOLD_RATIO:
                    label_bbox = list(lb)
                    if r.row_type != "TEXTAREA":
                        r.row_type = "FIELD_VERTICAL"
                    next_top = min((ob["bbox"][1] for ob in ocr_in_row), default=r.y_min)
                    r.vertical_split_y = (lb[3] + next_top) / 2.0
            elif ocr_left_of_input:
                best = max(ocr_left_of_input, key=lambda o: o["bbox"][2])
                lb = best["bbox"]
                if lb[2] - lb[0] >= 20:
                    label_bbox = list(lb)
                    if r.row_type != "TEXTAREA":
                        r.row_type = "FIELD_HORIZONTAL"
            if r.row_type in ("FIELD_HORIZONTAL", "FIELD_VERTICAL") and not label_bbox:
                r.row_type = "FIELD_INPUT_ONLY"
            ocr_below = [ob for ob in ocr_in_row if (ob["bbox"][1] + ob["bbox"][3]) / 2 >= row_center_y]
            helper_bbox = None
            for ob in ocr_below:
                ob_h = ob["bbox"][3] - ob["bbox"][1]
                if ob_h >= median_font * HELPER_FONT_MAX_RATIO:
                    continue
                ob_cx = (ob["bbox"][0] + ob["bbox"][2]) / 2
                if w_container > 0 and abs(ob_cx - container_cx) / w_container < BUTTON_CENTER_TOLERANCE:
                    continue
                helper_bbox = list(ob["bbox"])
                break
            r.label_bbox = label_bbox
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
        ry_min = min(ytops) if ytops else min(ys) - 20
        ry_max = max(ys) + (median_font * 0.5 if median_font > 0 else 10)
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
) -> Tuple[Optional[FormSkeleton], Dict[str, Any]]:
    """
    FormInnerLayout. Если передан non-empty visual_candidates — строки строятся только из CV (якоря по Y);
    OCR не задаёт границы строк и не участвует в определении textarea. OCR используется только для label/helper/placeholder.
    Иначе — fallback на OCR+edges (_build_rows_inside_container).
    """
    ocr_inside = ocr_inside or []
    layout_ocr, baseline, header_bboxes = normalize_ocr_for_layout(ocr_inside, container.bbox, image_path)
    if visual_candidates and len(container.bbox) >= 4:
        all_visual = list(visual_candidates)
        tall = _tall_contours_inside_container(image_path, container.bbox)
        for t in tall:
            if not any(_y_overlap_or_near(t, v) for v in all_visual):
                all_visual.append(t)
        anchors = collect_field_row_anchors(all_visual, container.bbox)
        if anchors:
            rows, skipped_bboxes, textarea_row_indices, form_start_y = _rows_from_visual_anchors(
                anchors, container.bbox, ocr_raw_for_action=ocr_inside
            )
        else:
            rows, skipped_bboxes, textarea_row_indices, form_start_y = _build_rows_inside_container(
                image_path, container, layout_ocr, ocr_raw_for_action=ocr_inside
            )
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

    if layout_type == "vertical":
        columns = [FormColumn(col_index=0, x_min=container.bbox[0], x_max=container.bbox[2])]
        column_boundaries = [(container.bbox[0], container.bbox[2])]
        for r in rows:
            r.column_count = 1
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
    out = img.copy()
    for r in rows:
        y1, y2 = int(r.y_min), int(r.y_max)
        cv2.rectangle(out, (int(r.x_min), y1), (int(r.x_max), y2), (0, 255, 255), 1)
        cv2.putText(out, "R%d" % r.row_index, (int(r.x_min) + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
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
        "HEADER": (255, 200, 0), "TEXT": (255, 200, 0),
        "FIELD": (0, 255, 255), "FIELD_HORIZONTAL": (0, 255, 255), "FIELD_VERTICAL": (100, 255, 200), "FIELD_INPUT_ONLY": (0, 255, 180),
        "TEXTAREA": (255, 150, 255), "ACTION": (0, 200, 255), "SPACER": (180, 180, 180),
    }
    for r in rows:
        y1, y2 = int(r.y_min), int(r.y_max)
        color = colors.get(r.row_type, (200, 200, 200))
        cv2.rectangle(out, (int(r.x_min), y1), (int(r.x_max), y2), color, 1)
        cv2.putText(out, "%s R%d" % (r.row_type, r.row_index), (int(r.x_min) + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
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
    for b in skipped_bboxes:
        if len(b) >= 4:
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(out, "skipped", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
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
    for r in rows:
        if r.row_index in textarea_row_indices or r.row_type == "TEXTAREA":
            y1, y2 = int(r.y_min), int(r.y_max)
            cv2.rectangle(out, (int(r.x_min), y1), (int(r.x_max), y2), (255, 100, 255), 2)
            cv2.putText(out, "TEXTAREA R%d" % r.row_index, (int(r.x_min) + 2, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_inner_layout: saved %s", output_path)
