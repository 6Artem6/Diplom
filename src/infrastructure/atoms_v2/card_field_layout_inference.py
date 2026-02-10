"""
CardFieldLayoutInference — восстановление input/textarea из структуры card/form_region, не из контуров.

Input — ожидаемая пустота в структуре формы. Card первична, layout важнее пикселей.
Вызывается ПОСЛЕ FormStructureDetection и postprocess, ДО semantic_validation и input_candidate_recovery (fallback).
"""

from __future__ import annotations

import hashlib
import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Row detection: элементы в одной строке если |y_center1 - y_center2| <= max(8px, 0.25 * median_height)
Y_TOLERANCE_PX = 8
Y_TOLERANCE_RATIO = 0.25
LABEL_MAX_CHARS = 25
LABEL_GAP_MIN_PX = 6
LABEL_GAP_MAX_PX = 12
LABEL_GAP_PX = 9  # 6–12px по спецификации
CARD_PADDING_RATIO_MIN = 0.05
CARD_PADDING_RATIO_MAX = 0.08
FIELD_ROW_MIN_WIDTH_RATIO = 0.60  # строка занимает ≥ 60% ширины card → candidate field
WIDTH_CLUSTER_TOLERANCE = 0.15  # ±15% для кластера ширины
TEXTAREA_HEIGHT_RATIO = 2.2  # textarea только если height >= 2.2× median (консервативно)
CONFIDENCE_LAYOUT_INFERENCE = 0.8
CONFIDENCE_LAYOUT_INFERENCE_LOW = 0.7
FIELD_ON_BUTTON_COVERAGE_MAX = 0.25
DEDUP_IOU_THRESHOLD = 0.70
# form_flow_end: строка с primary button или action-словами → всё ниже non-form-area
FORM_END_BUTTON_WIDTH_RATIO = 0.30  # кнопка шириной ≥ 30% card → конец формы
# header_row: OCR height >= 1.4× median_text_height → заголовок, не поле
HEADER_HEIGHT_RATIO = 1.4
# Колонки: поле только в одной из колонок, x_center ± offset
COLUMN_X_OFFSET_PX = 12
COLUMN_CLUSTER_TOLERANCE_RATIO = 0.25  # кластер колонок по x_center
# Orphan: без label/placeholder нужен baseline или width match
BASELINE_TOLERANCE_PX = 10
WIDTH_MATCH_TOLERANCE = 0.15
# Hard cap: полей не больше чем label-like строк + 2
MAX_FIELDS_OVER_LABELS = 2

ACTION_WORDS = frozenset({
    "save", "submit", "create", "search", "apply", "send", "add", "ok", "go", "login",
    "отправить", "сохранить", "создать", "поиск", "войти", "добавить", "отмена", "cancel",
    "далее",
})
PLACEHOLDER_HINTS = frozenset({
    "введите", "email@", "+7", "enter", "type", "search", "placeholder",
    "имя", "email", "телефон", "комментарий", "сообщение", "пароль", "password",
})


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
    return _intersection_area(inner, outer) / area_inner


def _point_inside_bbox(x: float, y: float, bbox: List[float]) -> bool:
    if len(bbox) < 4:
        return False
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _elements_inside_card(
    card_bbox: List[float],
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Элементы внутри card: атомы (button, layout, …) и OCR, чей bbox пересекается с card."""
    atoms_inside: List[Dict[str, Any]] = []
    ocr_inside: List[Dict[str, Any]] = []
    if len(card_bbox) < 4:
        return atoms_inside, ocr_inside
    for a in atoms:
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        if _point_inside_bbox(cx, cy, card_bbox) or _coverage_in_outer(bbox, card_bbox) >= 0.3:
            t = (a.get("type") or "").strip().lower()
            if t in ("button", "synthetic_btn", "layout", "text_block", "link", "container_candidate"):
                atoms_inside.append(a)
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _coverage_in_outer(obbox, card_bbox) >= 0.2 or _point_inside_bbox(
            (obbox[0] + obbox[2]) / 2, (obbox[1] + obbox[3]) / 2, card_bbox
        ):
            ocr_inside.append(ob)
    return atoms_inside, ocr_inside


def _row_detection(
    atoms_inside: List[Dict[str, Any]],
    ocr_inside: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Кластеризация по вертикали: элементы в одной строке если
    |y_center1 - y_center2| <= max(8px, 0.25 * median_height).
    Возвращает список row_candidates: { bbox (union), ocr_inside, has_button, mean_height, elements }.
    """
    elements: List[Dict[str, Any]] = []
    for a in atoms_inside:
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            elements.append({"bbox": bbox, "type": "atom", "atom": a})
    for ob in ocr_inside:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) >= 4:
            elements.append({"bbox": obbox, "type": "ocr", "ocr": ob})
    if not elements:
        return []
    heights = [e["bbox"][3] - e["bbox"][1] for e in elements if e["bbox"][3] > e["bbox"][1]]
    median_h = float(statistics.median(heights)) if heights else 24.0
    y_tol = max(Y_TOLERANCE_PX, median_h * Y_TOLERANCE_RATIO)
    sorted_el = sorted(elements, key=lambda e: (e["bbox"][1] + e["bbox"][3]) / 2)
    rows: List[Dict[str, Any]] = []
    current_row: List[Dict[str, Any]] = [sorted_el[0]]
    y_prev = (sorted_el[0]["bbox"][1] + sorted_el[0]["bbox"][3]) / 2
    for i in range(1, len(sorted_el)):
        y_cur = (sorted_el[i]["bbox"][1] + sorted_el[i]["bbox"][3]) / 2
        if abs(y_cur - y_prev) <= y_tol:
            current_row.append(sorted_el[i])
        else:
            if current_row:
                rows.append(_make_row(current_row, ocr_inside))
            current_row = [sorted_el[i]]
        y_prev = (y_cur + y_prev) / 2 if current_row else y_cur
    if current_row:
        rows.append(_make_row(current_row, ocr_inside))
    return rows


def _make_row(elements: List[Dict[str, Any]], ocr_inside: List[Dict[str, Any]]) -> Dict[str, Any]:
    bboxes = [e["bbox"] for e in elements]
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    row_bbox = [x1, y1, x2, y2]
    ocr_in_row: List[Dict[str, Any]] = []
    for ob in ocr_inside:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        cy = (obbox[1] + obbox[3]) / 2
        if row_bbox[1] <= cy <= row_bbox[3]:
            ocr_in_row.append(ob)
    has_button = any(e.get("type") == "atom" and (e.get("atom", {}).get("type") or "").strip().lower() in ("button", "synthetic_btn") for e in elements)
    heights = [b[3] - b[1] for b in bboxes if b[3] > b[1]]
    mean_height = float(statistics.mean(heights)) if heights else 24.0
    return {
        "bbox": row_bbox,
        "ocr_inside": ocr_in_row,
        "has_button": has_button,
        "mean_height": mean_height,
        "elements": elements,
    }


def _has_action_word_in_ocr(ocr_list: List[Dict[str, Any]]) -> bool:
    for ob in ocr_list:
        t = (ob.get("text") or "").strip().lower()
        if any(w in t for w in ACTION_WORDS):
            return True
    return False


def _row_is_form_end(row: Dict[str, Any], card_bbox: List[float]) -> bool:
    """Строка задаёт конец формы: primary button (ширина ≥ 30% card) или action-слова."""
    if len(card_bbox) < 4:
        return False
    card_w = card_bbox[2] - card_bbox[0]
    if row.get("has_button"):
        for e in row.get("elements") or []:
            if e.get("type") != "atom":
                continue
            a = e.get("atom", {})
            if (a.get("type") or "").strip().lower() not in ("button", "synthetic_btn"):
                continue
            bbox = a.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                bw = bbox[2] - bbox[0]
                if card_w > 0 and bw >= card_w * FORM_END_BUTTON_WIDTH_RATIO:
                    return True
    if _has_action_word_in_ocr(row.get("ocr_inside") or []):
        return True
    return False


def _compute_form_flow_end_y(rows: List[Dict[str, Any]], card_bbox: List[float]) -> Optional[float]:
    """Y-координата: ниже неё никаких полей (все строки ниже — non-form-area)."""
    if len(card_bbox) < 4 or not rows:
        return None
    sorted_rows = sorted(rows, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    for row in sorted_rows:
        if _row_is_form_end(row, card_bbox):
            return float(row["bbox"][1])
    return None


def _median_text_height(ocr_inside: List[Dict[str, Any]]) -> float:
    """Медианная высота OCR-боксов в card (для определения header)."""
    heights = []
    for ob in ocr_inside:
        bbox = ob.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4 and bbox[3] > bbox[1]:
            heights.append(bbox[3] - bbox[1])
    return float(statistics.median(heights)) if heights else 20.0


def _is_header_row(row: Dict[str, Any], median_text_height: float) -> bool:
    """Заголовок: есть OCR с высотой ≥ 1.4 × median_text_height."""
    if median_text_height <= 0:
        return False
    for ob in row.get("ocr_inside") or []:
        bbox = ob.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            h = bbox[3] - bbox[1]
            if h >= HEADER_HEIGHT_RATIO * median_text_height:
                return True
    return False


def _count_label_like_ocr_in_card(ocr_inside: List[Dict[str, Any]], card_bbox: List[float]) -> int:
    """Число OCR-строк, похожих на label (≤25 символов, слева или сверху от типичной позиции поля)."""
    if len(card_bbox) < 4 or not ocr_inside:
        return 0
    card_left = card_bbox[0]
    card_w = card_bbox[2] - card_bbox[0]
    left_third = card_left + card_w * 0.35
    count = 0
    seen_y: List[float] = []
    for ob in ocr_inside:
        text = (ob.get("text") or "").strip()
        if len(text) > LABEL_MAX_CHARS or len(text) == 0:
            continue
        bbox = ob.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        if cx <= left_third:
            if not any(abs(cy - sy) < 15 for sy in seen_y):
                seen_y.append(cy)
                count += 1
    return count


def _infer_column_centers(
    field_bboxes_x_centers: List[float],
    card_w: float,
) -> List[float]:
    """Кластеризация x_center полей → 1, 2 или 3 колонки. Возвращает список x-центров колонок."""
    if not field_bboxes_x_centers:
        return []
    sorted_cx = sorted(field_bboxes_x_centers)
    tol = max(20, card_w * COLUMN_CLUSTER_TOLERANCE_RATIO)
    clusters: List[List[float]] = []
    for cx in sorted_cx:
        placed = False
        for c in clusters:
            if abs(cx - statistics.median(c)) <= tol:
                c.append(cx)
                placed = True
                break
        if not placed:
            clusters.append([cx])
    return [float(statistics.median(c)) for c in clusters]


def _field_center_in_column(cx: float, column_centers: List[float]) -> bool:
    """Центр поля попадает в одну из колонок (± COLUMN_X_OFFSET_PX)."""
    if not column_centers:
        return True
    for cc in column_centers:
        if abs(cx - cc) <= COLUMN_X_OFFSET_PX:
            return True
    return False


def _has_placeholder_hint(ocr_list: List[Dict[str, Any]]) -> bool:
    for ob in ocr_list:
        t = (ob.get("text") or "").strip().lower()
        if any(h in t for h in PLACEHOLDER_HINTS):
            return True
    return False


def _label_left_or_top(row_bbox: List[float], ocr_in_row: List[Dict[str, Any]], card_bbox: List[float]) -> Tuple[bool, float]:
    """Есть ли label (OCR ≤ 25 символов) слева или сверху. Возвращает (has_label, label_right_x для x1 поля)."""
    if len(row_bbox) < 4 or len(card_bbox) < 4:
        return False, card_bbox[0]
    row_left, row_top = row_bbox[0], row_bbox[1]
    card_left = card_bbox[0]
    has_label = False
    label_right_x = card_left
    for ob in ocr_in_row:
        text = (ob.get("text") or "").strip()
        if len(text) > LABEL_MAX_CHARS or len(text) == 0:
            continue
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        ox1, ox2, oy1, oy2 = obbox[0], obbox[2], obbox[1], obbox[3]
        # label слева: OCR заканчивается левее начала строки, пересекается по Y
        if ox2 <= row_left + 80 and oy1 < row_bbox[3] and oy2 > row_bbox[1]:
            has_label = True
            if ox2 > label_right_x:
                label_right_x = ox2
        # label сверху: OCR выше строки, пересекается по X
        if oy2 <= row_top + 40 and ox1 < row_bbox[2] and ox2 > row_bbox[0]:
            has_label = True
    if not has_label:
        label_right_x = row_left
    return has_label, label_right_x


def _classify_rows(
    rows: List[Dict[str, Any]],
    card_bbox: List[float],
    form_flow_end_y: Optional[float],
    median_text_height: float,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Разделяет на button-row (skip), header-row, field-row.
    Строки ниже form_flow_end_y не рассматриваются. После header_row поле разрешено только при label+placeholder или baseline match.
    """
    if len(card_bbox) < 4:
        return [], 0, 0
    card_w = card_bbox[2] - card_bbox[0]
    if form_flow_end_y is not None:
        rows = [r for r in rows if r["bbox"][1] < form_flow_end_y]
    sorted_rows = sorted(rows, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    field_rows: List[Dict[str, Any]] = []
    button_skipped = 0
    text_skipped = 0
    after_header = False
    accepted_baselines: List[float] = []
    heights = [r["mean_height"] for r in sorted_rows]
    median_row_h = float(statistics.median(heights)) if heights else 30.0
    repeat_count = sum(1 for r in sorted_rows if abs(r["mean_height"] - median_row_h) / max(median_row_h, 1e-9) <= 0.25)
    for row in sorted_rows:
        if _is_header_row(row, median_text_height):
            after_header = True
            text_skipped += 1
            continue
        if row["has_button"] or _has_action_word_in_ocr(row.get("ocr_inside") or []):
            button_skipped += 1
            continue
        row_w = row["bbox"][2] - row["bbox"][0]
        width_ok = card_w > 0 and row_w >= card_w * FIELD_ROW_MIN_WIDTH_RATIO
        has_label, label_right_x = _label_left_or_top(row["bbox"], row.get("ocr_inside") or [], card_bbox)
        has_placeholder = _has_placeholder_hint(row.get("ocr_inside") or [])
        repeats = repeat_count >= 2 and abs(row["mean_height"] - median_row_h) / max(median_row_h, 1e-9) <= 0.25
        row_cy = (row["bbox"][1] + row["bbox"][3]) / 2
        baseline_match = any(abs(row_cy - b) <= BASELINE_TOLERANCE_PX for b in accepted_baselines)
        if after_header:
            if not (has_label and has_placeholder) and not baseline_match:
                text_skipped += 1
                continue
        if has_label or has_placeholder or width_ok or repeats or baseline_match:
            after_header = False
            row["label_right_x"] = label_right_x
            row["has_label"] = has_label
            row["has_placeholder"] = has_placeholder
            field_rows.append(row)
            accepted_baselines.append(row_cy)
        else:
            text_skipped += 1
    return field_rows, button_skipped, text_skipped


def _filter_orphan_fields(
    bboxes_with_meta: List[Tuple[List[float], bool, bool]],
) -> List[Tuple[List[float], bool, bool]]:
    """Убрать поля без label и без placeholder, если нет baseline или width match с другими."""
    if len(bboxes_with_meta) <= 1:
        return list(bboxes_with_meta)
    kept: List[Tuple[List[float], bool, bool]] = []
    for bbox, has_label, has_placeholder in bboxes_with_meta:
        if has_label or has_placeholder:
            kept.append((bbox, has_label, has_placeholder))
            continue
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        baseline_ok = any(
            abs(cy - (b[1] + b[3]) / 2) <= BASELINE_TOLERANCE_PX
            for b, _, _ in bboxes_with_meta if b != bbox
        )
        width_ok = any(
            b != bbox and abs(w - (b[2] - b[0])) / max(w, 1e-9) <= WIDTH_MATCH_TOLERANCE
            for b, _, _ in bboxes_with_meta
        )
        if baseline_ok or width_ok:
            kept.append((bbox, has_label, has_placeholder))
    return kept


def _recover_field_bbox(
    row: Dict[str, Any],
    card_bbox: List[float],
    median_input_height: float,
) -> List[float]:
    """
    BBox поля: x по label/card, высота ТОЛЬКО из median_input_height (OCR не задаёт y1/y2).
    y_center = центр строки, y1/y2 = center ± height/2.
    """
    if len(card_bbox) < 4:
        return row["bbox"][:]
    card_left, card_right = card_bbox[0], card_bbox[2]
    card_w = card_right - card_left
    padding = card_w * (CARD_PADDING_RATIO_MIN + CARD_PADDING_RATIO_MAX) / 2
    x1 = max(row.get("label_right_x", row["bbox"][0]) + LABEL_GAP_PX, card_left + padding)
    x2 = card_right - padding
    if x2 <= x1:
        x2 = row["bbox"][2]
    row_cy = (row["bbox"][1] + row["bbox"][3]) / 2
    h = max(20, median_input_height)
    y1 = row_cy - h / 2
    y2 = row_cy + h / 2
    return [float(x1), float(y1), float(x2), float(y2)]


def _normalize_widths(
    field_bboxes: List[List[float]],
    card_bbox: List[float],
) -> List[List[float]]:
    """Кластеризация по ширине (±15%), медианная ширина, выравнивание по левому краю."""
    if len(field_bboxes) <= 1 or len(card_bbox) < 4:
        return list(field_bboxes)
    widths = [b[2] - b[0] for b in field_bboxes]
    median_w = statistics.median(widths)
    result: List[List[float]] = []
    for b in field_bboxes:
        w = b[2] - b[0]
        if w <= 0:
            result.append(list(b))
            continue
        if abs(w - median_w) / max(median_w, 1e-9) <= WIDTH_CLUSTER_TOLERANCE:
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            new_x2 = x1 + median_w
            result.append([x1, y1, new_x2, y2])
        else:
            result.append(list(b))
    return result


def _field_overlaps_button(bbox: List[float], atoms_inside: List[Dict[str, Any]]) -> bool:
    """True если поле перекрывается кнопкой более чем на FIELD_ON_BUTTON_COVERAGE_MAX (инпут не поверх кнопки)."""
    area_b = _bbox_area(bbox)
    if area_b <= 0:
        return False
    for a in atoms_inside:
        t = (a.get("type") or "").strip().lower()
        if t not in ("button", "synthetic_btn"):
            continue
        abox = a.get("bbox", [0, 0, 0, 0])
        if len(abox) < 4:
            continue
        inter = _intersection_area(bbox, abox)
        if inter / area_b >= FIELD_ON_BUTTON_COVERAGE_MAX:
            return True
    return False


def _iou_bbox(a: List[float], b: List[float]) -> float:
    inter = _intersection_area(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(1e-9, union)


def _dedup_bboxes_by_iou(
    bboxes_with_meta: List[Tuple[List[float], bool]],
) -> List[Tuple[List[float], bool]]:
    """Объединяет дубликаты по IoU > DEDUP_IOU_THRESHOLD; оставляет один bbox (большая площадь), label_linked = any."""
    if len(bboxes_with_meta) <= 1:
        return list(bboxes_with_meta)
    kept: List[Tuple[List[float], bool]] = []
    used = [False] * len(bboxes_with_meta)
    for i in range(len(bboxes_with_meta)):
        if used[i]:
            continue
        bbox_i, linked_i = bboxes_with_meta[i]
        best_j = -1
        best_area = _bbox_area(bbox_i)
        for j in range(i + 1, len(bboxes_with_meta)):
            if used[j]:
                continue
            bbox_j, linked_j = bboxes_with_meta[j]
            if _iou_bbox(bbox_i, bbox_j) >= DEDUP_IOU_THRESHOLD:
                area_j = _bbox_area(bbox_j)
                if area_j >= best_area:
                    best_area = area_j
                    best_j = j
        if best_j >= 0:
            used[i] = True
            used[best_j] = True
            bbox_best, linked_best = bboxes_with_meta[best_j]
            kept.append((list(bbox_best), linked_i or linked_best))
        else:
            kept.append((list(bbox_i), linked_i))
    return kept


def _dedup_bboxes_by_iou_triple(
    bboxes_with_meta: List[Tuple[List[float], bool, bool]],
) -> List[Tuple[List[float], bool, bool]]:
    """То же для (bbox, label_linked, has_placeholder); при слиянии label_linked and has_placeholder = any."""
    if len(bboxes_with_meta) <= 1:
        return list(bboxes_with_meta)
    kept: List[Tuple[List[float], bool, bool]] = []
    used = [False] * len(bboxes_with_meta)
    for i in range(len(bboxes_with_meta)):
        if used[i]:
            continue
        bbox_i, lb_i, ph_i = bboxes_with_meta[i]
        best_j = -1
        best_area = _bbox_area(bbox_i)
        for j in range(i + 1, len(bboxes_with_meta)):
            if used[j]:
                continue
            bbox_j, _, _ = bboxes_with_meta[j]
            if _iou_bbox(bbox_i, bbox_j) >= DEDUP_IOU_THRESHOLD:
                area_j = _bbox_area(bbox_j)
                if area_j >= best_area:
                    best_area = area_j
                    best_j = j
        if best_j >= 0:
            used[i] = True
            used[best_j] = True
            bbox_best, lb_best, ph_best = bboxes_with_meta[best_j]
            kept.append((list(bbox_best), lb_i or lb_best, ph_i or ph_best))
        else:
            kept.append((list(bbox_i), lb_i, ph_i))
    return kept


def _make_atom_id(bbox: List[float], prefix: str = "card_layout") -> str:
    h = hashlib.sha256(str([round(x, 1) for x in bbox]).encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


def run_card_field_layout_inference(
    form_regions: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Восстановление input/textarea из структуры card/form_region.
    Возвращает (atoms_to_append, log_lines).
    """
    log_lines: List[str] = []
    new_atoms: List[Dict[str, Any]] = []
    existing_ids = {a.get("id", "") for a in atoms if a.get("id")}

    if not form_regions:
        log_lines.append("card_field_layout_inference: no form_regions, skip")
        return new_atoms, log_lines

    for card in form_regions:
        card_id = card.get("id", "") or ("form_region_%s" % id(card))
        card_bbox = card.get("bbox", [0, 0, 0, 0])
        if len(card_bbox) < 4:
            continue
        atoms_inside, ocr_inside = _elements_inside_card(card_bbox, atoms, raw_ocr_boxes)
        if not atoms_inside and not ocr_inside:
            log_lines.append("card_field_layout_inference: card_id=%s no elements inside, skip" % card_id)
            continue
        rows = _row_detection(atoms_inside, ocr_inside)
        n_rows = len(rows)
        form_flow_end_y = _compute_form_flow_end_y(rows, card_bbox)
        median_text_height = _median_text_height(ocr_inside)
        field_rows, button_skipped, text_skipped = _classify_rows(
            rows, card_bbox, form_flow_end_y, median_text_height
        )
        n_field_rows = len(field_rows)
        if n_field_rows == 0:
            log_lines.append(
                "card_field_layout_inference: card_id=%s rows=%d field_rows=0 button_skipped=%d text_skipped=%d"
                % (card_id, n_rows, button_skipped, text_skipped)
            )
            continue
        median_input_height = float(statistics.median([r["mean_height"] for r in field_rows])) if field_rows else 36.0
        recovered: List[Tuple[List[float], bool, bool]] = [
            (_recover_field_bbox(r, card_bbox, median_input_height), r.get("has_label", False), r.get("has_placeholder", False))
            for r in field_rows
        ]
        recovered = [(b, lb, ph) for b, lb, ph in recovered if not _field_overlaps_button(b, atoms_inside)]
        if not recovered:
            log_lines.append(
                "card_field_layout_inference: card_id=%s all recovered overlapped button, skip" % card_id
            )
            continue
        card_w = card_bbox[2] - card_bbox[0]
        x_centers = [(b[0] + b[2]) / 2 for b, _, _ in recovered]
        column_centers = _infer_column_centers(x_centers, card_w)
        if column_centers:
            recovered = [(b, lb, ph) for b, lb, ph in recovered if _field_center_in_column((b[0] + b[2]) / 2, column_centers)]
        if not recovered:
            log_lines.append("card_field_layout_inference: card_id=%s no fields in columns, skip" % card_id)
            continue
        recovered = _filter_orphan_fields(recovered)
        if not recovered:
            log_lines.append("card_field_layout_inference: card_id=%s all orphans filtered, skip" % card_id)
            continue
        bboxes_only = [b for b, _, _ in recovered]
        normalized_bboxes = _normalize_widths(bboxes_only, card_bbox)
        normalized_with_meta: List[Tuple[List[float], bool, bool]] = [
            (normalized_bboxes[i], recovered[i][1], recovered[i][2]) for i in range(len(recovered))
        ]
        normalized_with_meta = _dedup_bboxes_by_iou_triple(normalized_with_meta)
        label_count = _count_label_like_ocr_in_card(ocr_inside, card_bbox)
        cap = label_count + MAX_FIELDS_OVER_LABELS
        if len(normalized_with_meta) > cap:
            normalized_with_meta = normalized_with_meta[:cap]
            log_lines.append("card_field_layout_inference: card_id=%s hard_cap applied (max=%d)" % (card_id, cap))
        n_normalized = len(normalized_with_meta)
        heights = [b[3] - b[1] for b, _, _ in normalized_with_meta if len(b) >= 4 and b[3] > b[1]]
        median_h = float(statistics.median(heights)) if heights else 36.0
        for bbox, label_linked, _ in normalized_with_meta:
            if len(bbox) < 4:
                continue
            h = bbox[3] - bbox[1]
            atom_type = "textarea_candidate" if h >= median_h * TEXTAREA_HEIGHT_RATIO else "input_candidate"
            aid = _make_atom_id(bbox)
            if aid in existing_ids:
                continue
            existing_ids.add(aid)
            evidence = {"source": "card_layout_inference", "label_linked": label_linked}
            new_atoms.append({
                "id": aid,
                "type": atom_type,
                "bbox": list(bbox),
                "confidence": CONFIDENCE_LAYOUT_INFERENCE,
                "source": "input_candidate_recovery",
                "recovery_source": "card_layout_inference",
                "evidence": evidence,
            })
        log_lines.append(
            "card_field_layout_inference: card_id=%s rows=%d field_rows=%d recovered=%d normalized=%d button_skipped=%d text_skipped=%d"
            % (card_id, n_rows, n_field_rows, len(recovered), n_normalized, button_skipped, text_skipped)
        )

    log_lines.append(
        "card_field_layout_inference: total inferred=%d (source=card_layout_inference)"
        % len(new_atoms)
    )
    return new_atoms, log_lines
