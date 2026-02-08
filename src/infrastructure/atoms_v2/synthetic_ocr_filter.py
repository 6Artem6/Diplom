"""
OCR-based фильтр и нормализация synthetic-атомов до merge с real.

Правка №1: все OCR-фильтры, normalize, gates применяются ТОЛЬКО к synthetic.
Real не трогаем; real типы никогда не понижаются здесь (работаем только с atoms_synthetic).

Выполняется сразу после Detectron2 synthetic, до merge и stabilize_atoms.
Порядок: 1) OCR-anchored нормализация bbox (только link);
         2) OCR-gate + real-overlap gate для synthetic input (Правка №2: IoU ≥ 0.4);
         3) pre-merge: synthetic input ∩ button → input деградирует;
         4) пустые кнопки → candidate;
         5) pagination/filters группа → candidate.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Padding вокруг OCR при пересчёте tight bbox (px)
TIGHT_BBOX_PADDING_PX = 4
# Synthetic input: допустимый aspect ratio (ширина/высота)
INPUT_ASPECT_MIN = 0.5
INPUT_ASPECT_MAX = 20.0
# Synthetic input допустим ТОЛЬКО при пересечении с real CV bbox (button/container/input) ≥ IoU (Правка №2)
INPUT_REAL_OVERLAP_IOU_MIN = 0.4
# Label: макс. расстояние от bbox (слева/сверху)
INPUT_LABEL_OFFSET_PX = 40
INPUT_LABEL_OVERLAP_AXIS = 0.5
# OCR внутри bbox: мин. доля площади OCR в bbox для учёта
OCR_INSIDE_COVERAGE = 0.2
# Пустая кнопка: макс. расстояние до "рядом" OCR (px)
BUTTON_NEARBY_OCR_MAX_PX = 50
# Pagination/filters: 2+ OCR на одной горизонтали, шаг по X равномерный → container_candidate
PAGINATION_MIN_OCR_IN_ROW = 2
PAGINATION_MAX_Y_VARIANCE_RATIO = 0.3
# Равномерность шага: макс. коэффициент вариации межцентровых расстояний
PAGINATION_MAX_STEP_CV = 0.6
# Подтверждение границ по CV: container_candidate и input допустимы только при наличии CV region в окрестности
CV_REGION_MIN_IOU = 0.1


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


def _coverage_bbox_in_bbox(inner: List[float], outer: List[float]) -> float:
    if len(inner) < 4 or len(outer) < 4:
        return 0.0
    area_inner = _bbox_area(inner)
    if area_inner <= 0:
        return 0.0
    inter = _intersection_area(inner, outer)
    return inter / area_inner


def _iou_bbox_bbox(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    inter = _intersection_area(a, b)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return inter / max(1e-9, union)


def _ocr_boxes_inside_bbox(
    atom_bbox: List[float],
    raw_ocr_boxes: List[Dict[str, Any]],
    min_coverage: float = OCR_INSIDE_COVERAGE,
) -> List[List[float]]:
    """Список bbox OCR, у которых доля площади внутри atom_bbox >= min_coverage."""
    out: List[List[float]] = []
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _coverage_bbox_in_bbox(obbox, atom_bbox) >= min_coverage:
            out.append(obbox)
    return out


def _tight_bbox_around_boxes(boxes: List[List[float]], padding_px: float = TIGHT_BBOX_PADDING_PX) -> List[float]:
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [b[0] for b in boxes if len(b) >= 4]
    ys = [b[1] for b in boxes if len(b) >= 4]
    x2s = [b[2] for b in boxes if len(b) >= 4]
    y2s = [b[3] for b in boxes if len(b) >= 4]
    if not xs:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        max(0.0, min(xs) - padding_px),
        max(0.0, min(ys) - padding_px),
        max(x2s) + padding_px,
        max(y2s) + padding_px,
    ]


def _has_aligned_label(
    atom_bbox: List[float],
    raw_ocr_boxes: List[Dict[str, Any]],
    max_offset_px: float = INPUT_LABEL_OFFSET_PX,
    min_axis_overlap: float = INPUT_LABEL_OVERLAP_AXIS,
) -> bool:
    """OCR слева или сверху с перекрытием по оси и зазором <= max_offset_px."""
    if len(atom_bbox) < 4:
        return False
    x1, y1, x2, y2 = atom_bbox[0], atom_bbox[1], atom_bbox[2], atom_bbox[3]
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        ox1, oy1, ox2, oy2 = obbox[0], obbox[1], obbox[2], obbox[3]
        ocr_h = oy2 - oy1
        ocr_w = ox2 - ox1
        if ocr_h <= 0 or ocr_w <= 0:
            continue
        if ox2 <= x1 and (x1 - ox2) <= max_offset_px:
            overlap_y = min(oy2, y2) - max(oy1, y1)
            if overlap_y >= min_axis_overlap * ocr_h:
                return True
        if oy2 <= y1 and (y1 - oy2) <= max_offset_px:
            overlap_x = min(ox2, x2) - max(ox1, x1)
            if overlap_x >= min_axis_overlap * ocr_w:
                return True
    return False


def _has_ocr_inside(atom_bbox: List[float], raw_ocr_boxes: List[Dict[str, Any]], min_coverage: float = OCR_INSIDE_COVERAGE) -> bool:
    return len(_ocr_boxes_inside_bbox(atom_bbox, raw_ocr_boxes, min_coverage)) > 0


def _input_aspect_ok(bbox: List[float]) -> bool:
    if len(bbox) < 4:
        return False
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if h <= 0:
        return False
    aspect = w / h
    return INPUT_ASPECT_MIN <= aspect <= INPUT_ASPECT_MAX


def _downgrade_to(a: Dict[str, Any], new_type: str) -> None:
    old_type = a.get("type", "")
    a["type"] = new_type
    a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
    logger.debug("synthetic_ocr_filter: %s -> %s id=%s", old_type, new_type, a.get("id"))


# --- 1) OCR-anchored нормализация bbox для synthetic button и link ---


def _normalize_synthetic_bbox_by_ocr(
    atoms_synthetic: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> None:
    """Только link: если внутри bbox есть OCR — tight bbox по OCR + padding. Button не трогаем (OCR — якорь, не форма кнопки)."""
    for a in atoms_synthetic:
        if a.get("source") != "synthetic":
            continue
        if a.get("type") != "link":
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        inside = _ocr_boxes_inside_bbox(bbox, raw_ocr_boxes, min_coverage=OCR_INSIDE_COVERAGE)
        if not inside:
            continue
        tight = _tight_bbox_around_boxes(inside, TIGHT_BBOX_PADDING_PX)
        a["bbox"] = tight
        logger.debug("synthetic_ocr_filter: link bbox normalized to OCR tight id=%s", a.get("id"))


# --- 2) OCR-gate и real-overlap gate для synthetic input ---


def _synthetic_input_has_real_overlap(bbox: List[float], atoms_real: List[Dict[str, Any]], min_iou: float = INPUT_REAL_OVERLAP_IOU_MIN) -> bool:
    """Input — визуальный UI с рамкой: synthetic допустим только при пересечении с real CV bbox (button/container/input)."""
    if len(bbox) < 4:
        return False
    for r in atoms_real:
        if r.get("type") not in ("button", "container", "input"):
            continue
        rbbox = r.get("bbox", [0, 0, 0, 0])
        if len(rbbox) < 4:
            continue
        if _iou_bbox_bbox(bbox, rbbox) >= min_iou:
            return True
    return False


def _ocr_gate_synthetic_input(
    atoms_synthetic: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    atoms_real: List[Dict[str, Any]],
) -> None:
    """Synthetic input допустим только при (OCR внутри или label слева/сверху), допустимом aspect и overlap с real CV bbox. Иначе — layout_candidate."""
    for a in atoms_synthetic:
        if a.get("source") != "synthetic":
            continue
        if a.get("type") != "input":
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            _downgrade_to(a, "layout_candidate")
            continue
        if not _synthetic_input_has_real_overlap(bbox, atoms_real):
            _downgrade_to(a, "layout_candidate")
            logger.debug("synthetic_ocr_filter: input no real CV overlap -> layout_candidate id=%s", a.get("id"))
            continue
        has_inside = _has_ocr_inside(bbox, raw_ocr_boxes)
        has_label = _has_aligned_label(bbox, raw_ocr_boxes)
        if not has_inside and not has_label:
            _downgrade_to(a, "layout_candidate")
            logger.debug("synthetic_ocr_filter: input no OCR/label -> layout_candidate id=%s", a.get("id"))
            continue
        if not _input_aspect_ok(bbox):
            _downgrade_to(a, "layout_candidate")
            logger.debug("synthetic_ocr_filter: input bad aspect -> layout_candidate id=%s", a.get("id"))


# --- 3) Pre-merge: synthetic input ∩ button (real или synthetic) → input проигрывает ---


def _pre_merge_synthetic_input_vs_button(
    atoms_synthetic: List[Dict[str, Any]],
    atoms_real: List[Dict[str, Any]],
) -> None:
    """Synthetic input, пересекающийся с любой button (real или synthetic), деградирует в layout_candidate."""
    buttons_bboxes: List[List[float]] = []
    for r in atoms_real:
        if r.get("type") == "button":
            b = r.get("bbox", [0, 0, 0, 0])
            if len(b) >= 4:
                buttons_bboxes.append(b)
    for s in atoms_synthetic:
        if s.get("source") != "synthetic" or s.get("type") != "button":
            continue
        b = s.get("bbox", [0, 0, 0, 0])
        if len(b) >= 4:
            buttons_bboxes.append(b)

    for a in atoms_synthetic:
        if a.get("source") != "synthetic" or a.get("type") != "input":
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        for bb in buttons_bboxes:
            if _intersection_area(bbox, bb) > 0:
                _downgrade_to(a, "layout_candidate")
                logger.debug("synthetic_ocr_filter: input intersects button -> layout_candidate id=%s", a.get("id"))
                break


# --- 4) Пустые synthetic button (нет OCR внутри и рядом) ---


def _ocr_near_bbox(bbox: List[float], raw_ocr_boxes: List[Dict[str, Any]], max_px: float) -> bool:
    """Есть ли OCR в радиусе max_px от границы bbox (пересечение или близко)."""
    if len(bbox) < 4:
        return False
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        ox1, oy1, ox2, oy2 = obbox[0], obbox[1], obbox[2], obbox[3]
        # расширенная зона: bbox + margin
        if (ox2 < x1 - max_px or ox1 > x2 + max_px or oy2 < y1 - max_px or oy1 > y2 + max_px):
            continue
        if _intersection_area(obbox, bbox) > 0:
            return True
        # центр OCR рядом с bbox
        cx = (ox1 + ox2) / 2
        cy = (oy1 + oy2) / 2
        if (x1 - max_px <= cx <= x2 + max_px and y1 - max_px <= cy <= y2 + max_px):
            return True
    return False


def _filter_empty_synthetic_buttons(
    atoms_synthetic: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> None:
    """Synthetic button без OCR внутри и без OCR рядом → container_candidate."""
    for a in atoms_synthetic:
        if a.get("source") != "synthetic" or a.get("type") != "button":
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            _downgrade_to(a, "container_candidate")
            continue
        if _has_ocr_inside(bbox, raw_ocr_boxes):
            continue
        if _ocr_near_bbox(bbox, raw_ocr_boxes, BUTTON_NEARBY_OCR_MAX_PX):
            continue
        _downgrade_to(a, "container_candidate")
        logger.debug("synthetic_ocr_filter: empty button -> container_candidate id=%s", a.get("id"))


# --- 5) Pagination/filters: несколько OCR в строке, один bbox → не одна кнопка ---


def _is_pagination_row(
    atom_bbox: List[float],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> bool:
    """2+ OCR на одной горизонтальной линии с близкой высотой и равномерным шагом внутри bbox."""
    inside = _ocr_boxes_inside_bbox(atom_bbox, raw_ocr_boxes, min_coverage=0.3)
    if len(inside) < PAGINATION_MIN_OCR_IN_ROW:
        return False
    h = atom_bbox[3] - atom_bbox[1]
    if h <= 0:
        return False
    y_centers = [(b[1] + b[3]) / 2 for b in inside]
    y_var = max(y_centers) - min(y_centers) if y_centers else 0
    if y_var > PAGINATION_MAX_Y_VARIANCE_RATIO * h:
        return False
    # равномерность шага по X
    x_centers = sorted((b[0] + b[2]) / 2 for b in inside)
    if len(x_centers) < 2:
        return True
    steps = [x_centers[i + 1] - x_centers[i] for i in range(len(x_centers) - 1)]
    avg_step = sum(steps) / len(steps)
    if avg_step <= 0:
        return True
    variance = sum((s - avg_step) ** 2 for s in steps) / len(steps)
    std = math.sqrt(variance)
    cv = std / avg_step if avg_step else 0
    if cv > PAGINATION_MAX_STEP_CV:
        return False
    return True


def _pagination_group_downgrade(
    atoms_synthetic: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> None:
    """Synthetic button, покрывающий строку из 2+ равномерных OCR, → container_candidate (группа контролов)."""
    for a in atoms_synthetic:
        if a.get("source") != "synthetic" or a.get("type") != "button":
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        if _is_pagination_row(bbox, raw_ocr_boxes):
            _downgrade_to(a, "container_candidate")
            logger.debug("synthetic_ocr_filter: pagination/filters row -> container_candidate id=%s", a.get("id"))


# --- 6) Подтверждение границ по CV bbox (перед объединением) ---


def _has_region_in_vicinity(bbox: List[float], regions: List[Dict[str, Any]], min_iou: float = CV_REGION_MIN_IOU) -> bool:
    """Есть ли CV region (прямоугольник/скруглённый контур), пересекающийся с bbox с IoU >= min_iou."""
    if len(bbox) < 4 or not regions:
        return False
    area_a = _bbox_area(bbox)
    if area_a <= 0:
        return False
    for r in regions:
        rbbox = r.get("bbox", [0, 0, 0, 0])
        if len(rbbox) < 4:
            continue
        iou = _iou_bbox_bbox(bbox, rbbox)
        if iou >= min_iou:
            return True
    return False


def _filter_synthetic_without_cv_region(
    atoms_synthetic: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> None:
    """Фантомные input/button/link без CV region не допускаются. IoU ≥ 0.1 с region. Фильтровать перед merge."""
    for a in atoms_synthetic:
        if a.get("source") != "synthetic":
            continue
        t = a.get("type", "")
        if t not in ("input", "container_candidate", "button", "link"):
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            if t == "input":
                _downgrade_to(a, "layout_candidate")
            elif t == "button":
                _downgrade_to(a, "container_candidate")
            elif t == "link":
                _downgrade_to(a, "inline_text_candidate")
            else:
                _downgrade_to(a, "layout_candidate")
            continue
        if not _has_region_in_vicinity(bbox, regions):
            if t == "input":
                _downgrade_to(a, "layout_candidate")
            elif t == "button":
                _downgrade_to(a, "container_candidate")
            elif t == "link":
                _downgrade_to(a, "inline_text_candidate")
            else:
                _downgrade_to(a, "layout_candidate")
            logger.debug("synthetic_ocr_filter: no CV region in vicinity -> %s id=%s type=%s", a.get("type"), a.get("id"), t)


# --- Публичный вход ---


def filter_synthetic_atoms_by_ocr(
    atoms_synthetic: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    atoms_real: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> None:
    """
    OCR-based фильтр и нормализация synthetic-атомов. Модифицирует atoms_synthetic in-place.
    Вызывать сразу после получения atoms_synthetic от Detectron2, до merge и stabilize_atoms.

    Порядок: 1) нормализация bbox link по OCR;
             2) OCR-gate + real-overlap для input;
             3) synthetic input ∩ button → input деградирует;
             4) пустые кнопки → candidate;
             5) pagination/filters группа → candidate;
             6) container_candidate и input только при наличии CV region в окрестности.
    """
    if not atoms_synthetic:
        return
    _normalize_synthetic_bbox_by_ocr(atoms_synthetic, raw_ocr_boxes)
    _ocr_gate_synthetic_input(atoms_synthetic, raw_ocr_boxes, atoms_real)
    _pre_merge_synthetic_input_vs_button(atoms_synthetic, atoms_real)
    _filter_empty_synthetic_buttons(atoms_synthetic, raw_ocr_boxes)
    _pagination_group_downgrade(atoms_synthetic, raw_ocr_boxes)
    _filter_synthetic_without_cv_region(atoms_synthetic, regions)
