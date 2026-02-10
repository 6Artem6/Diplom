"""
MicroFieldScanner — микроуровень: один bbox на слот из визуальных кандидатов в полосе.

Источник истины: визуальные кандидаты. Сначала выбираем лучшего по многокритериальному скору,
потом проверяем, что он подходит под слот. Пустой слот допустим; фантом — нет.
- Скоринг: aspect ratio (input-подобный = вытянутый), штраф за внутренний текст, выравнивание по вертикали,
  расстояние до кнопок. Площадь — слабый сигнал.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

ROW_OVERLAP_Y_RATIO = 0.4
WIDTH_MATCH_TOLERANCE = 0.20
DEDUP_IOU_THRESHOLD = 0.70
TEXTAREA_HEIGHT_RATIO = 1.8
# Скоринг: порог ниже которого слот оставляем пустым
INPUT_SCORE_MIN = 0.15
# Aspect: input-подобный — вытянутый по ширине (w/h в диапазоне)
INPUT_ASPECT_MIN = 1.8
INPUT_ASPECT_MAX = 25.0
# Штраф за долю площади bbox, занятую OCR
TEXT_COVERAGE_PENALTY = 0.5


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


def _candidate_in_row(vis_bbox: List[float], row_bbox: List[float]) -> bool:
    if len(vis_bbox) < 4 or len(row_bbox) < 4:
        return False
    cy = (vis_bbox[1] + vis_bbox[3]) / 2
    if row_bbox[1] <= cy <= row_bbox[3]:
        return True
    row_h = row_bbox[3] - row_bbox[1]
    if row_h <= 0:
        return False
    overlap_y = max(0, min(vis_bbox[3], row_bbox[3]) - max(vis_bbox[1], row_bbox[1]))
    return overlap_y / row_h >= ROW_OVERLAP_Y_RATIO


def _deduplicate(bboxes: List[List[float]], iou_threshold: float = DEDUP_IOU_THRESHOLD) -> List[List[float]]:
    """IoU ≥ threshold → оставляем один (большая площадь)."""
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
            if _iou(b_i, bboxes[j]) >= iou_threshold:
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


def _align_width_to_median(bboxes: List[List[float]]) -> List[List[float]]:
    if len(bboxes) <= 1:
        return list(bboxes)
    widths = [b[2] - b[0] for b in bboxes]
    median_w = sorted(widths)[len(widths) // 2]
    out: List[List[float]] = []
    for b in bboxes:
        w = b[2] - b[0]
        if w <= 0:
            out.append(list(b))
            continue
        if abs(w - median_w) / max(median_w, 1e-9) <= WIDTH_MATCH_TOLERANCE:
            out.append([b[0], b[1], b[0] + median_w, b[3]])
        else:
            out.append(list(b))
    return out


def _input_likelihood_score(
    bbox: List[float],
    ocr_inside_row: List[Dict[str, Any]],
    all_candidates: List[List[float]],
    row_bbox: List[float],
    is_likely_button: bool,
) -> float:
    """
    Многокритериальный скор «похожести на input»: aspect, текст внутри, выравнивание, не кнопка.
    Площадь не доминирует.
    """
    if is_likely_button or len(bbox) < 4:
        return 0.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if h <= 0:
        return 0.0
    aspect = w / h
    aspect_ok = INPUT_ASPECT_MIN <= aspect <= INPUT_ASPECT_MAX
    aspect_score = 0.7 if aspect_ok else max(0.0, 0.4 - abs(aspect - 2.0) * 0.1)
    text_penalty = 0.0
    area_b = _bbox_area(bbox)
    if area_b > 0:
        for ob in ocr_inside_row:
            obbox = ob.get("bbox", [0, 0, 0, 0])
            if len(obbox) < 4:
                continue
            inter = _intersection_area(bbox, obbox)
            if inter / area_b >= 0.2:
                text_penalty += TEXT_COVERAGE_PENALTY
    text_score = max(0.0, 1.0 - text_penalty)
    x_center = (bbox[0] + bbox[2]) / 2
    others_x = [(b[0] + b[2]) / 2 for b in all_candidates if b != bbox and len(b) >= 4]
    align_score = 0.0
    if others_x:
        median_x = sorted(others_x)[len(others_x) // 2]
        if abs(x_center - median_x) < 50:
            align_score = 0.2
    return aspect_score + text_score + align_score


def _take_at_most_one_per_slot(
    bboxes: List[List[float]],
    slot_count: int,
    row_bbox: List[float],
    ocr_inside_row: List[Dict[str, Any]],
    is_button_fn: Any,
    column_boundaries: Optional[List[Tuple[float, float]]] = None,
) -> List[List[float]]:
    """
    Не более slot_count bbox. Выбор по скору «input-подобности», не по площади.
    Если лучший кандидат ниже порога — слот пустой (пустой слот допустим).
    """
    if not bboxes or slot_count <= 0:
        return []
    scored: List[Tuple[float, List[float]]] = []
    for b in bboxes:
        sc = _input_likelihood_score(
            b, ocr_inside_row, bboxes, row_bbox, is_likely_button=is_button_fn(b),
        )
        scored.append((sc, b))
    if column_boundaries is None or len(column_boundaries) == 0:
        best = max(scored, key=lambda x: x[0])
        if best[0] >= INPUT_SCORE_MIN:
            return [best[1]]
        return []
    result: List[List[float]] = []
    for (cx1, cx2) in column_boundaries:
        in_col = [(s, b) for s, b in scored if cx1 <= (b[0] + b[2]) / 2 <= cx2]
        if not in_col:
            continue
        best = max(in_col, key=lambda x: x[0])
        if best[0] >= INPUT_SCORE_MIN:
            result.append(best[1])
        if len(result) >= slot_count:
            break
    return result[:slot_count]


def scan_fields(
    row_bbox: List[float],
    ocr_inside_row: List[Dict[str, Any]],
    visual_candidates: List[List[float]],
    slot_count: int = 1,
    column_boundaries: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[List[float], Literal["input", "textarea"]]]:
    """
    Не более одного bbox на слот. Только визуальные кандидаты в row_bbox.
    slot_count: максимум bbox в этой строке (1 для vertical, N для grid).
    column_boundaries: для grid — границы колонок (x1, x2); для vertical — None.
    """
    if len(row_bbox) < 4 or slot_count <= 0:
        return []
    in_row = [v for v in visual_candidates if _candidate_in_row(v, row_bbox)]
    if not in_row:
        return []

    action_words = frozenset({
        "save", "submit", "search", "send", "add", "ok", "login", "cancel", "apply",
        "отправить", "сохранить", "войти", "далее",
    })

    def _is_likely_button(bbox: List[float]) -> bool:
        for ob in ocr_inside_row:
            txt = (ob.get("text") or "").strip().lower()
            if not any(w in txt for w in action_words):
                continue
            obbox = ob.get("bbox", [0, 0, 0, 0])
            if len(obbox) < 4:
                continue
            if _intersection_area(bbox, obbox) / max(_bbox_area(bbox), 1e-9) >= 0.3:
                return True
        return False

    def _is_button(b: List[float]) -> bool:
        return _is_likely_button(b)

    in_row = [b for b in in_row if not _is_button(b)]
    if not in_row:
        return []

    in_row = _deduplicate(in_row)
    in_row = _align_width_to_median(in_row)
    in_row = _take_at_most_one_per_slot(
        in_row, slot_count, row_bbox, ocr_inside_row, _is_button, column_boundaries,
    )
    if not in_row:
        return []

    heights = [b[3] - b[1] for b in in_row if b[3] > b[1]]
    median_h = float(sorted(heights)[len(heights) // 2]) if heights else 36.0
    result: List[Tuple[List[float], Literal["input", "textarea"]]] = []
    for b in in_row:
        h = b[3] - b[1]
        t: Literal["input", "textarea"] = "textarea" if h >= median_h * TEXTAREA_HEIGHT_RATIO else "input"
        result.append((list(b), t))
    return result
