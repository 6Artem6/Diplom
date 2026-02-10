"""
InputCandidateRecovery — двухфазная модель: гипотеза → подтверждение.

Phase A: InputSeedDetection — где МОЖЕТ быть input (input_seed, не atom, не финальный bbox).
Phase B: InputBBoxRefinement — есть ли в ROI реальный контейнер (Canny в ROI); без подтверждения — поля нет.

OCR = слабый сигнал (текст, label), не геометрическая основа bbox. Bbox только из Phase B (контур по Canny).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Phase A: seed ROI (не финальный bbox) ---
ROI_MARGIN_Y_PX = 35
ROI_MARGIN_X_PX = 150
INPUT_HEIGHT_MIN_PX = 24
INPUT_HEIGHT_MAX_PX = 80
INPUT_ASPECT_MIN = 3.0
INPUT_ASPECT_MAX = 20.0
MIN_OCR_LINE_HEIGHT = 12

# --- Kill-switch: seed внутри button/image/table → discard; внутри container/layout → разрешить ---
SEED_INSIDE_FORBIDDEN_COVERAGE = 0.5
DISCARD_INSIDE_TYPES = ("button", "card", "table", "image")  # только они убивают seed
ALLOW_INSIDE_TYPES = ("container_candidate", "layout")  # форма/карточка — разрешаем seed внутри

# --- Fallback: повторяющиеся горизонтальные полосы 32–56px (структурные признаки формы) ---
FALLBACK_STRIP_HEIGHT_MIN = 32
FALLBACK_STRIP_HEIGHT_MAX = 56
FALLBACK_STRIP_HEIGHT = 40  # высота полосы в пикселях
FALLBACK_STRIP_STEP = 28
FALLBACK_MIN_STRIP_WIDTH = 80
FALLBACK_ASPECT_MIN = 2.5  # ширина ≥ 2.5× высоты (признак поля, не кнопки)

# --- Dedup ---
DEDUP_IOU_MIN = 0.6
DEDUP_ASPECT_RATIO_TOL = 0.25
DEDUP_HEIGHT_RATIO_TOL = 0.10

# --- Phase B: Canny / контуры ---
CANNY_LOW, CANNY_HIGH = 40, 120
ADAPTIVE_BLOCK = 11
CONTOUR_MIN_AREA = 200
CONTOUR_ASPECT_MIN = 2.5
CONTOUR_ASPECT_MAX = 22.0
CONTOUR_HEIGHT_MIN = 20
CONTOUR_HEIGHT_MAX = 85
# Border-only эвристика: input = тонкий бордер, высота 24–60px; card/outline-button отсекаем
BORDER_INPUT_HEIGHT_MIN = 24
BORDER_INPUT_HEIGHT_MAX = 60
BORDER_ASPECT_MIN_INPUT = 2.5  # ниже — слишком квадратно (outline-button)

# --- Veto: input не может лежать на кнопке / карточке ---
VETO_REFINED_TYPES = ("button", "synthetic_btn", "clickable", "card")
VETO_REFINED_ON_BUTTON_COVERAGE = 0.25  # доля refined, попавшая в veto-тип → discard

# --- Header zone: заголовок страницы не input ---
HEADER_ZONE_PX = 70  # bbox целиком выше этой линии → отбросить (title, не поле ввода)

# --- Merge двух bbox одного поля ---
MERGE_VERTICAL_OVERLAP_MIN = 0.6
MERGE_CENTER_X_MAX_PX = 15
MERGE_HEIGHT_RATIO_TOL = 0.20
MERGE_OCCUPIED_SEED_COVERAGE = 0.5  # seed внутри уже найденного input → discard

# --- Scoring (только для подтверждённых контуров) ---
CONFIDENCE_LOW = 0.4
LABEL_OFFSET_PX = 120
ACTION_WORDS = frozenset({"search", "submit", "save", "login", "send", "go", "ok", "apply"})
# Textarea: высота > TYPICAL_INPUT_HEIGHT_PX * TEXTAREA_HEIGHT_RATIO → textarea_candidate
TYPICAL_INPUT_HEIGHT_PX = 40
TEXTAREA_HEIGHT_RATIO = 1.8


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
    """Доля площади inner, попадающая в outer."""
    area_inner = _bbox_area(inner)
    if area_inner <= 0:
        return 0.0
    return _intersection_area(inner, outer) / area_inner


def _get_discard_bboxes(atoms: List[Dict[str, Any]]) -> List[List[float]]:
    """Bbox'ы button/card/table/image — seed внутри них отбрасывается. Контейнер/layout не убивают."""
    out: List[List[float]] = []
    for a in atoms:
        t = (a.get("type") or "").strip().lower()
        if t not in DISCARD_INSIDE_TYPES:
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            out.append(bbox)
    return out


def _refined_in_header(refined_bbox: List[float], header_px: float = HEADER_ZONE_PX) -> bool:
    """True если bbox целиком в зоне заголовка (заголовок страницы — не input)."""
    if len(refined_bbox) < 4:
        return False
    return refined_bbox[3] <= header_px


def _refined_on_button(refined_bbox: List[float], atoms: List[Dict[str, Any]]) -> bool:
    """
    Жёсткий veto: input не может лежать на кнопке или карточке.
    Если refined_bbox пересекается с button/synthetic_btn/clickable/card и
    coverage = (refined ∩ atom) / area(refined) ≥ 0.25 → True (discard).
    """
    area_refined = _bbox_area(refined_bbox)
    if area_refined <= 0:
        return False
    for a in atoms:
        t = (a.get("type") or "").strip().lower()
        if t not in VETO_REFINED_TYPES:
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        inter = _intersection_area(refined_bbox, bbox)
        if inter / area_refined >= VETO_REFINED_ON_BUTTON_COVERAGE:
            return True
    return False


def _seed_inside_forbidden(seed_roi_bbox: List[float], discard_bboxes: List[List[float]]) -> bool:
    """True если seed лежит внутри button/image/table (доля площади seed внутри > порога). Контейнер ≠ кнопка."""
    for fb in discard_bboxes:
        if _coverage_in_outer(seed_roi_bbox, fb) > SEED_INSIDE_FORBIDDEN_COVERAGE:
            return True
    return False


def _has_label_left(
    ocr_line_bbox: List[float],
    raw_ocr_boxes: List[Dict[str, Any]],
    max_offset_px: float = LABEL_OFFSET_PX,
) -> Tuple[bool, float]:
    """Есть ли OCR слева от линии; возвращает (has_label, rightmost_x of label или 0)."""
    x1 = ocr_line_bbox[0]
    best_x2 = 0.0
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        ox2 = obbox[2]
        if ox2 <= x1 and (x1 - ox2) <= max_offset_px:
            if ox2 > best_x2:
                best_x2 = ox2
    return best_x2 > 0, best_x2


def _has_action_button_nearby(
    center_xy: Tuple[float, float],
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    max_px: float = 150,
) -> bool:
    cx, cy = center_xy
    for a in atoms:
        t = (a.get("type") or "").strip().lower()
        if t != "button":
            continue
        abbox = a.get("bbox", [0, 0, 0, 0])
        if len(abbox) < 4:
            continue
        acx = (abbox[0] + abbox[2]) / 2
        acy = (abbox[1] + abbox[3]) / 2
        if ((cx - acx) ** 2 + (cy - acy) ** 2) ** 0.5 > max_px:
            continue
        for ob in raw_ocr_boxes:
            obbox = ob.get("bbox", [0, 0, 0, 0])
            if len(obbox) < 4 or _coverage_in_outer(obbox, abbox) < 0.3:
                continue
            if any(w in (ob.get("text") or "").strip().lower() for w in ACTION_WORDS):
                return True
    return False


# --- Phase A: InputSeedDetection ---


def _group_ocr_into_lines(
    raw_ocr_boxes: List[Dict[str, Any]],
    y_tolerance: float = 18,
) -> List[List[Dict[str, Any]]]:
    sorted_ocr = sorted(
        [o for o in raw_ocr_boxes if len((o.get("bbox") or [])) >= 4],
        key=lambda b: ((b["bbox"][1] + b["bbox"][3]) / 2, b["bbox"][0]),
    )
    lines: List[List[Dict[str, Any]]] = []
    for ob in sorted_ocr:
        obbox = ob.get("bbox", [0, 0, 0, 0])
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


def _detect_input_seeds(
    raw_ocr_boxes: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
    screen_w: float,
    screen_h: float,
) -> List[Dict[str, Any]]:
    """
    Phase A: гипотезы (input_seed). Seed не atom, не финальный bbox.
    roi_bbox = область для вырезки ROI (OCR-линия + margin); используется только для Canny.
    """
    if not raw_ocr_boxes:
        return []
    discard_bboxes = _get_discard_bboxes(atoms)
    lines = _group_ocr_into_lines(raw_ocr_boxes)
    seeds: List[Dict[str, Any]] = []
    for line in lines:
        if not line:
            continue
        bboxes = [o.get("bbox", [0, 0, 0, 0]) for o in line if len((o.get("bbox") or [])) >= 4]
        if not bboxes:
            continue
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        h = y2 - y1
        if h < MIN_OCR_LINE_HEIGHT:
            continue
        roi_bbox = [
            max(0, x1 - ROI_MARGIN_X_PX),
            max(0, y1 - ROI_MARGIN_Y_PX),
            min(screen_w, x2 + ROI_MARGIN_X_PX),
            min(screen_h, y2 + ROI_MARGIN_Y_PX),
        ]
        if _seed_inside_forbidden(roi_bbox, discard_bboxes):
            continue
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        has_label, label_right_x = _has_label_left([x1, y1, x2, y2], raw_ocr_boxes)
        has_action = _has_action_button_nearby((cx, cy), atoms, raw_ocr_boxes)
        seeds.append({
            "roi_bbox": roi_bbox,
            "ocr_line_bbox": [x1, y1, x2, y2],
            "center": (cx, cy),
            "context": {"has_label": has_label, "label_right_x": label_right_x, "has_action": has_action},
            "source": "ocr",
        })
    return seeds


# --- InputVisualSeedFallback: визуальные seeds без OCR ---


def _detect_visual_fallback_seeds(
    regions: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
    screen_w: float,
    screen_h: float,
) -> List[Dict[str, Any]]:
    """
    Fallback: когда OCR дал 0–1 seed, строим визуальные seeds по регионам (горизонтальные полосы).
    Не создаёт атомы, только разрешает локальный CV-поиск в Phase B.
    """
    if not regions:
        return []
    discard_bboxes = _get_discard_bboxes(atoms)
    seeds: List[Dict[str, Any]] = []
    for r in regions:
        rbbox = r.get("bbox", [0, 0, 0, 0])
        if len(rbbox) < 4:
            continue
        rx1, ry1, rx2, ry2 = rbbox[0], rbbox[1], rbbox[2], rbbox[3]
        rw = rx2 - rx1
        rh = ry2 - ry1
        if rw < FALLBACK_MIN_STRIP_WIDTH or rh < FALLBACK_STRIP_HEIGHT_MIN:
            continue
        y = ry1
        while y + FALLBACK_STRIP_HEIGHT <= ry2:
            strip = [rx1, y, rx2, y + FALLBACK_STRIP_HEIGHT]
            strip_w = rx2 - rx1
            strip_h = FALLBACK_STRIP_HEIGHT
            if strip_w / max(strip_h, 1e-9) < FALLBACK_ASPECT_MIN:
                y += FALLBACK_STRIP_STEP
                continue
            if _seed_inside_forbidden(strip, discard_bboxes):
                y += FALLBACK_STRIP_STEP
                continue
            roi_bbox = [
                max(0, rx1 - ROI_MARGIN_X_PX),
                max(0, y - ROI_MARGIN_Y_PX),
                min(screen_w, rx2 + ROI_MARGIN_X_PX),
                min(screen_h, y + FALLBACK_STRIP_HEIGHT + ROI_MARGIN_Y_PX),
            ]
            cx = (rx1 + rx2) / 2
            cy = y + strip_h / 2
            seeds.append({
                "roi_bbox": roi_bbox,
                "ocr_line_bbox": strip,
                "center": (cx, cy),
                "context": {"has_label": False, "label_right_x": 0, "has_action": False},
                "source": "fallback_visual",
            })
            y += FALLBACK_STRIP_STEP
    return seeds


# --- InputBBoxDeduplicator ---


def _iou_bbox(a: List[float], b: List[float]) -> float:
    inter = _intersection_area(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(1e-9, union)


def _deduplicate_input_bboxes(
    candidates: List[Tuple[List[float], float, Dict[str, bool], str]],
) -> List[Tuple[List[float], float, Dict[str, bool], str]]:
    """
    IoU > 0.6, aspect схож, height ±10% → оставить один (большая площадь, выше confidence).
    candidates: [(bbox, confidence, evidence, source), ...]
    """
    if len(candidates) <= 1:
        return candidates
    keep = set(range(len(candidates)))
    for i in range(len(candidates)):
        if i not in keep:
            continue
        bbox_i, conf_i, _, _ = candidates[i]
        hi = bbox_i[3] - bbox_i[1]
        wi = bbox_i[2] - bbox_i[0]
        aspect_i = wi / max(hi, 1e-9)
        area_i = _bbox_area(bbox_i)
        for j in range(i + 1, len(candidates)):
            if j not in keep:
                continue
            bbox_j, conf_j, _, _ = candidates[j]
            if _iou_bbox(bbox_i, bbox_j) < DEDUP_IOU_MIN:
                continue
            hj = bbox_j[3] - bbox_j[1]
            wj = bbox_j[2] - bbox_j[0]
            aspect_j = wj / max(hj, 1e-9)
            if abs(aspect_i - aspect_j) / max(aspect_i, 1e-9) > DEDUP_ASPECT_RATIO_TOL:
                continue
            if abs(hi - hj) / max(hi, 1e-9) > DEDUP_HEIGHT_RATIO_TOL:
                continue
            area_j = _bbox_area(bbox_j)
            if area_j >= area_i and conf_j >= conf_i:
                keep.discard(i)
                break
            elif area_j <= area_i and conf_j <= conf_i:
                keep.discard(j)
    return [candidates[i] for i in sorted(keep)]


def _vertical_overlap(bbox_a: List[float], bbox_b: List[float]) -> float:
    """Доля перекрытия по Y относительно меньшей высоты (0..1)."""
    if len(bbox_a) < 4 or len(bbox_b) < 4:
        return 0.0
    iy1 = max(bbox_a[1], bbox_b[1])
    iy2 = min(bbox_a[3], bbox_b[3])
    if iy2 <= iy1:
        return 0.0
    ha = bbox_a[3] - bbox_a[1]
    hb = bbox_b[3] - bbox_b[1]
    min_h = min(ha, hb)
    if min_h <= 0:
        return 0.0
    return (iy2 - iy1) / min_h


def _merge_vertical_input_bboxes(
    candidates: List[Tuple[List[float], float, Dict[str, bool], str]],
) -> List[Tuple[List[float], float, Dict[str, bool], str]]:
    """
    Объединение двух bbox одного поля: vertical overlap ≥ 0.6, |center_x| ≤ 15px,
    высоты ±20%, один вложен в другой по X → заменяем парой union bbox.
    """
    if len(candidates) <= 1:
        return candidates
    merged: List[Tuple[List[float], float, Dict[str, bool], str]] = []
    used = set()

    for i in range(len(candidates)):
        if i in used:
            continue
        bbox_i, conf_i, ev_i, src_i = candidates[i]
        hi = bbox_i[3] - bbox_i[1]
        cxi = (bbox_i[0] + bbox_i[2]) / 2
        best_j: Optional[int] = None
        for j in range(i + 1, len(candidates)):
            if j in used:
                continue
            bbox_j, conf_j, ev_j, src_j = candidates[j]
            hj = bbox_j[3] - bbox_j[1]
            cxj = (bbox_j[0] + bbox_j[2]) / 2
            if _vertical_overlap(bbox_i, bbox_j) < MERGE_VERTICAL_OVERLAP_MIN:
                continue
            if abs(cxi - cxj) > MERGE_CENTER_X_MAX_PX:
                continue
            if abs(hi - hj) / max(hi, 1e-9) > MERGE_HEIGHT_RATIO_TOL:
                continue
            # один вложен в другой по X или значимое перекрытие
            ix1 = max(bbox_i[0], bbox_j[0])
            ix2 = min(bbox_i[2], bbox_j[2])
            if ix2 <= ix1:
                continue
            best_j = j
            break
        if best_j is not None:
            j = best_j
            bbox_j, conf_j, ev_j, src_j = candidates[j]
            union_bbox = [
                min(bbox_i[0], bbox_j[0]),
                min(bbox_i[1], bbox_j[1]),
                max(bbox_i[2], bbox_j[2]),
                max(bbox_i[3], bbox_j[3]),
            ]
            best_conf = max(conf_i, conf_j)
            best_ev = ev_i if conf_i >= conf_j else ev_j
            best_src = src_i if conf_i >= conf_j else src_j
            merged.append((union_bbox, best_conf, best_ev, best_src))
            used.add(i)
            used.add(j)
        else:
            merged.append(candidates[i])
    return merged


def _seed_inside_occupied(seed_roi_bbox: List[float], occupied_zones: List[List[float]]) -> bool:
    """True если seed попадает внутрь уже найденного input (доля seed внутри zone ≥ порога)."""
    for zone in occupied_zones:
        if _coverage_in_outer(seed_roi_bbox, zone) >= MERGE_OCCUPIED_SEED_COVERAGE:
            return True
    return False


# --- Phase B: InputBBoxRefinement (Canny в ROI) ---


def _refine_bbox_canny(
    seed: Dict[str, Any],
    image_path: str,
    img_w: int,
    img_h: int,
) -> Optional[List[float]]:
    """
    Phase B: в ROI ищем визуальный контейнер (границы). Без контура — возвращаем None (seed отбрасывается).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.debug("input_candidate_recovery: cv2 not available, skip Phase B")
        return None
    roi = seed.get("roi_bbox", [0, 0, 0, 0])
    if len(roi) < 4:
        return None
    x1, y1, x2, y2 = int(max(0, roi[0])), int(max(0, roi[1])), int(min(img_w, roi[2])), int(min(img_h, roi[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img_h_actual, img_w_actual = img.shape[:2]
    x1 = max(0, min(x1, img_w_actual - 1))
    y1 = max(0, min(y1, img_h_actual - 1))
    x2 = max(x1 + 1, min(x2, img_w_actual))
    y2 = max(y1 + 1, min(y2, img_h_actual))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK, 2
    )
    edges = cv2.Canny(adaptive, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[float, List[float]]] = []
    label_right_x = seed.get("context", {}).get("label_right_x") or 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < CONTOUR_MIN_AREA:
            continue
        rx, ry, rw, rh = cv2.boundingRect(c)
        if rh <= 0:
            continue
        aspect = rw / rh
        if aspect < CONTOUR_ASPECT_MIN or aspect > CONTOUR_ASPECT_MAX:
            continue
        if rh < CONTOUR_HEIGHT_MIN or rh > CONTOUR_HEIGHT_MAX:
            continue
        # Border-only эвристика: input = тонкий бордер, высота 24–60px; card/outline-button отсекаем
        if rh < BORDER_INPUT_HEIGHT_MIN or rh > BORDER_INPUT_HEIGHT_MAX:
            continue
        if aspect < BORDER_ASPECT_MIN_INPUT:
            continue
        global_x1 = x1 + rx
        global_y1 = y1 + ry
        global_bbox = [float(global_x1), float(global_y1), float(global_x1 + rw), float(global_y1 + rh)]
        score = 0.5
        if label_right_x > 0 and abs(global_x1 - label_right_x) < 30:
            score += 0.3
        candidates.append((score, global_bbox))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    refined = candidates[0][1]
    if label_right_x > 0 and refined[0] < label_right_x + 2:
        refined = [max(refined[0], label_right_x + 2), refined[1], refined[2], refined[3]]
        if refined[2] - refined[0] < 30:
            return None
    return refined


def _score_refined_candidate(
    refined_bbox: List[float],
    seed: Dict[str, Any],
    raw_ocr_boxes: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
    screen_w: float,
    screen_h: float,
) -> Tuple[float, Dict[str, bool]]:
    """Confidence и evidence для уже подтверждённого (Phase B) кандидата."""
    evidence: Dict[str, bool] = {"geometry": False, "text_density": False, "alignment": False, "context": False}
    w = refined_bbox[2] - refined_bbox[0]
    h = refined_bbox[3] - refined_bbox[1]
    if h <= 0:
        return 0.0, evidence
    aspect = w / h
    if INPUT_ASPECT_MIN <= aspect <= INPUT_ASPECT_MAX and INPUT_HEIGHT_MIN_PX <= h <= INPUT_HEIGHT_MAX_PX:
        evidence["geometry"] = True
    area = _bbox_area(refined_bbox)
    screen_area = screen_w * screen_h
    if screen_area <= 0 or area > 0.25 * screen_area:
        return 0.0, evidence
    text_area = 0.0
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _coverage_in_outer(obbox, refined_bbox) >= 0.2:
            text_area += _intersection_area(obbox, refined_bbox)
    if area > 0 and text_area / area < 0.4:
        evidence["text_density"] = True
    evidence["alignment"] = True
    if seed.get("context", {}).get("has_label") or seed.get("context", {}).get("has_action"):
        evidence["context"] = True
    score = 0.0
    if evidence["geometry"]:
        score += 0.3
    if evidence["text_density"]:
        score += 0.3
    if evidence["alignment"]:
        score += 0.2
    if evidence["context"]:
        score += 0.25
    return min(1.0, score), evidence


def _make_atom_id(bbox: List[float], prefix: str = "recovery") -> str:
    h = hashlib.sha256(str([round(x, 1) for x in bbox]).encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


def _run_phase_b_on_seeds(
    seeds: List[Dict[str, Any]],
    occupied_input_zones: List[List[float]],
    image_path: str,
    img_w: int,
    img_h: int,
    raw_ocr_boxes: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
    screen_w: float,
    screen_h: float,
) -> Tuple[
    List[Tuple[List[float], float, Dict[str, bool], str]],
    int,
    int,
]:
    """
    Phase B по списку seeds: для каждого — Canny, контур, veto на кнопку, occupied zones.
    Возвращает (candidates, phase_b_no_contour, saved_by_fallback, saved_by_visual_scanner).
    """
    candidates: List[Tuple[List[float], float, Dict[str, bool], str]] = []
    phase_b_no_contour = 0
    saved_by_fallback = 0
    saved_by_visual_scanner = 0
    for seed in seeds:
        if _seed_inside_occupied(seed.get("roi_bbox", [0, 0, 0, 0]), occupied_input_zones):
            continue
        refined = _refine_bbox_canny(seed, image_path, img_w, img_h)
        if refined is None:
            phase_b_no_contour += 1
            continue
        if _refined_in_header(refined, HEADER_ZONE_PX):
            continue
        confidence, evidence = _score_refined_candidate(
            refined, seed, raw_ocr_boxes, atoms, screen_w, screen_h
        )
        if confidence < CONFIDENCE_LOW:
            continue
        if _refined_on_button(refined, atoms):
            continue
        src = seed.get("source", "ocr")
        candidates.append((refined, confidence, evidence, src))
        occupied_input_zones.append(refined)
        if src == "fallback_visual":
            saved_by_fallback += 1
        elif src == "visual_scanner":
            saved_by_visual_scanner += 1
    return candidates, phase_b_no_contour, saved_by_fallback, saved_by_visual_scanner


def _seed_inside_form_region(
    seed: Dict[str, Any],
    form_regions: List[Dict[str, Any]],
) -> bool:
    """Seed (center или roi_bbox) попадает внутрь хотя бы одного form_region."""
    if not form_regions:
        return True
    roi = seed.get("roi_bbox", [0, 0, 0, 0])
    if len(roi) >= 4:
        cx = (roi[0] + roi[2]) / 2
        cy = (roi[1] + roi[3]) / 2
        for fr in form_regions:
            bbox = fr.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
                return True
            if _coverage_in_outer(roi, bbox) >= 0.3:
                return True
    return False


def run_input_candidate_recovery(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    screen_size: Optional[Tuple[float, float]] = None,
    image_path: Optional[str] = None,
    form_regions: Optional[List[Dict[str, Any]]] = None,
    dark_theme: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    FORM-FIRST: input ищется только внутри form_region.
    Phase A: OCR seeds только внутри form_region. Fallback visual seeds только внутри form_region.
    Phase B: Canny только для подтверждения seed. Veto: кнопка, header, occupied.
    """
    log_lines: List[str] = []
    use_form_filter = form_regions is not None
    form_regions = form_regions if form_regions is not None else []
    if use_form_filter and len(form_regions) == 0:
        log_lines.append("input_candidate_recovery: no form_regions, skip (input only inside form)")
        return atoms, log_lines
    if screen_size is None and regions:
        rbboxes = [r.get("bbox", [0, 0, 0, 0]) for r in regions if len((r.get("bbox") or [])) >= 4]
        screen_w = max(b[2] for b in rbboxes) if rbboxes else 1920.0
        screen_h = max(b[3] for b in rbboxes) if rbboxes else 1080.0
    elif screen_size is not None:
        screen_w, screen_h = screen_size[0], screen_size[1]
    else:
        screen_w, screen_h = 1920.0, 1080.0

    ocr_seeds_raw = _detect_input_seeds(raw_ocr_boxes, atoms, screen_w, screen_h) if raw_ocr_boxes else []
    if use_form_filter and form_regions:
        ocr_seeds = [s for s in ocr_seeds_raw if _seed_inside_form_region(s, form_regions)]
        log_lines.append(
            "input_candidate_recovery form-first: %d form_regions, OCR seeds %d -> %d inside form"
            % (len(form_regions), len(ocr_seeds_raw), len(ocr_seeds))
        )
    else:
        ocr_seeds = ocr_seeds_raw
        log_lines.append("input_candidate_recovery Phase A: %d OCR seeds (no form_regions filter)" % len(ocr_seeds))
    n_ocr_seeds = len(ocr_seeds)

    # VisualFieldScanner: геометрический скан внутри form_region (без OCR bootstrap)
    visual_scanner_seeds: List[Dict[str, Any]] = []
    if image_path and form_regions:
        try:
            from src.infrastructure.atoms_v2.visual_field_scanner import run_visual_field_scan
            visual_bboxes, visual_log = run_visual_field_scan(
                str(image_path), form_regions, dark_theme=dark_theme
            )
            log_lines.extend(visual_log)
            for vb in visual_bboxes:
                if len(vb) < 4:
                    continue
                cx = (vb[0] + vb[2]) / 2
                cy = (vb[1] + vb[3]) / 2
                visual_scanner_seeds.append({
                    "roi_bbox": list(vb),
                    "ocr_line_bbox": list(vb),
                    "center": (cx, cy),
                    "context": {"has_label": False, "label_right_x": 0, "has_action": False},
                    "source": "visual_scanner",
                })
            if visual_scanner_seeds:
                log_lines.append(
                    "input_candidate_recovery: %d visual_scanner seeds (geometric, no OCR)" % len(visual_scanner_seeds)
                )
        except Exception as e:
            logger.debug("input_candidate_recovery: visual_field_scanner failed: %s", e)
            log_lines.append("input_candidate_recovery: visual_field_scanner failed: %s" % e)
    # Объединяем OCR seeds и visual_scanner seeds для Phase B
    seeds_for_phase_b = ocr_seeds + visual_scanner_seeds

    if not image_path:
        log_lines.append("input_candidate_recovery: no image_path, skip Phase B (no confirmation)")
        return atoms, log_lines

    img_w, img_h = int(screen_w), int(screen_h)
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            img_h, img_w = img.shape[:2]
    except Exception:
        pass

    occupied_input_zones: List[List[float]] = []
    fallback_seeds_list: List[Dict[str, Any]] = []
    candidates, phase_b_no_contour, saved_by_fallback, saved_by_visual_scanner = _run_phase_b_on_seeds(
        seeds_for_phase_b, occupied_input_zones, str(image_path), img_w, img_h,
        raw_ocr_boxes or [], atoms, screen_w, screen_h,
    )

    if len(candidates) == 0 and n_ocr_seeds > 0 and regions:
        fallback_seeds_list = _detect_visual_fallback_seeds(regions, atoms, screen_w, screen_h)
        if use_form_filter and form_regions:
            fallback_seeds_list = [s for s in fallback_seeds_list if _seed_inside_form_region(s, form_regions)]
        log_lines.append(
            "input_candidate_recovery: Phase B confirmed=0, trigger visual fallback (%d seeds inside form)" % len(fallback_seeds_list)
        )
        c2, nc2, sf2, _ = _run_phase_b_on_seeds(
            fallback_seeds_list, occupied_input_zones, str(image_path), img_w, img_h,
            raw_ocr_boxes or [], atoms, screen_w, screen_h,
        )
        candidates = c2
        phase_b_no_contour += nc2
        saved_by_fallback += sf2

    candidates_before_dedup = len(candidates)
    log_lines.append(
        "input_candidate_recovery Phase B: %d seeds no contour, %d candidates before dedup (veto applied)"
        % (phase_b_no_contour, candidates_before_dedup)
    )

    candidates = _deduplicate_input_bboxes(candidates)
    candidates_after_dedup = len(candidates)
    log_lines.append("input_candidate_recovery dedup: %d candidates after dedup" % candidates_after_dedup)

    prev_n = 0
    while prev_n != len(candidates):
        prev_n = len(candidates)
        candidates = _merge_vertical_input_bboxes(candidates)
    candidates_after_merge = len(candidates)
    log_lines.append("input_candidate_recovery merge: %d candidates after vertical merge" % candidates_after_merge)

    # Layout propagation: внутри form_region при ≥2 полях добавляем кандидаты в ожидаемых позициях
    propagation_count = 0
    if form_regions and image_path and len(candidates) >= 2:
        try:
            from src.infrastructure.atoms_v2.layout_propagation import run_layout_propagation
            field_bboxes = [bbox for (bbox, _, _, _) in candidates]
            propagated, n_prop = run_layout_propagation(form_regions, field_bboxes, str(image_path))
            if propagated:
                candidates = list(candidates) + propagated
                propagation_count = n_prop
                log_lines.append("input_candidate_recovery propagation: %d candidates added (expected positions)" % n_prop)
        except Exception as prop_e:
            logger.debug("input_candidate_recovery: layout_propagation failed: %s", prop_e)
    if propagation_count > 0:
        candidates = _deduplicate_input_bboxes(candidates)
        log_lines.append("input_candidate_recovery after propagation dedup: %d candidates" % len(candidates))

    if dark_theme:
        log_lines.append("input_candidate_recovery: dark_theme=True, higher trust for input inside form")
    existing_ids = {a.get("id", "") for a in atoms if a.get("id")}
    added = 0
    confidence_boost = 1.1 if dark_theme else 1.0
    for refined, confidence, evidence, src in candidates:
        aid = _make_atom_id(refined)
        if aid in existing_ids:
            continue
        existing_ids.add(aid)
        confidence = min(1.0, confidence * confidence_boost)
        h_refined = refined[3] - refined[1]
        atom_type = (
            "textarea_candidate"
            if h_refined >= TYPICAL_INPUT_HEIGHT_PX * TEXTAREA_HEIGHT_RATIO
            else "input_candidate"
        )
        atoms.append({
            "id": aid,
            "type": atom_type,
            "bbox": refined,
            "confidence": confidence,
            "source": "input_candidate_recovery",
            "evidence": evidence,
            "recovery_source": src,
        })
        added += 1
        log_lines.append(
            "input_candidate_recovery: %s confidence=%.2f source=%s evidence=%s" % (aid, confidence, src, evidence)
        )
    log_lines.append(
        "input_candidate_recovery: added %d candidates (Phase B confirmed), saved_by_fallback=%d saved_by_visual_scanner=%d"
        % (added, saved_by_fallback, saved_by_visual_scanner)
    )
    recovery_sources = [src for (_, _, _, src) in candidates]
    fallback_triggered = n_ocr_seeds > 0 and len(fallback_seeds_list) > 0
    log_lines.append(
        "input_candidate_recovery summary: ocr_seeds=%d visual_scanner_seeds=%d fallback_triggered=%s phase_b_no_contour=%d "
        "candidates_before_dedup=%d after_dedup=%d after_merge=%d propagation_added=%d added=%d saved_by_fallback=%d saved_by_visual_scanner=%d recovery_source=%s"
        % (
            n_ocr_seeds,
            len(visual_scanner_seeds),
            "yes" if fallback_triggered else "no",
            phase_b_no_contour,
            candidates_before_dedup,
            candidates_after_dedup,
            candidates_after_merge,
            propagation_count,
            added,
            saved_by_fallback,
            saved_by_visual_scanner,
            recovery_sources,
        )
    )
    return atoms, log_lines
