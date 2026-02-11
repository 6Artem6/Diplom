"""
Уровень 4 — Поиск полей по роли слота (Role-based Field Search).

Для каждого слота ищем только кандидатов под конкретную роль:
- input_slot → визуальные bbox с рамкой, контрастным фоном, типовой высотой
- textarea_slot → другие размеры
- action_slot → отдельная логика (кнопки ≠ поля)

Один слот → максимум один bbox. Поле не появляется без слота.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.models import RowSlots, Slot, SlotAssignment, SlotRole

logger = logging.getLogger(__name__)

# Перекрытие bbox с зоной слота (IoU с слотом) — минимум для кандидата
SLOT_OVERLAP_MIN = 0.3
# input_bbox никогда не режется по OCR; обрезка допустима только при пересечении с соседним слотом > 40%
SLOT_OVERLAP_FOR_TRIM = 0.4
# Соотношение сторон: input вытянут по ширине
INPUT_ASPECT_MIN = 2.0
TEXTAREA_ASPECT_MAX = 4.0
INPUT_HEIGHT_MIN_PX = 20
INPUT_HEIGHT_MAX_PX = 80
TEXTAREA_HEIGHT_MIN_PX = 60


def _bbox_iou(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(1e-9, union)


def _slot_bbox(slot: Slot) -> List[float]:
    return [slot.x_min, slot.y_min, slot.x_max, slot.y_max]


def _score_candidate_for_slot(
    bbox: List[float],
    slot: Slot,
    role: SlotRole,
) -> float:
    """Скор кандидата для данного слота по роли."""
    if len(bbox) < 4:
        return 0.0
    slot_rect = _slot_bbox(slot)
    iou = _bbox_iou(bbox, slot_rect)
    if iou < SLOT_OVERLAP_MIN:
        return 0.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if h <= 0:
        return 0.0
    aspect = w / h
    if role == "input_slot":
        if aspect < INPUT_ASPECT_MIN:
            return 0.0
        if h < INPUT_HEIGHT_MIN_PX or h > INPUT_HEIGHT_MAX_PX:
            return 0.3 * iou
        return iou * (0.5 + 0.5 * min(1.0, (aspect - INPUT_ASPECT_MIN) / 8.0))
    if role == "textarea_slot":
        if h < TEXTAREA_HEIGHT_MIN_PX:
            return 0.0
        if aspect > TEXTAREA_ASPECT_MAX:
            return 0.4 * iou
        return iou
    return 0.0


def _bbox_inside_container(bbox: List[float], container_bbox: List[float]) -> bool:
    """bbox целиком внутри container."""
    if len(bbox) < 4 or len(container_bbox) < 4:
        return False
    return (container_bbox[0] <= bbox[0] and container_bbox[1] <= bbox[1]
            and bbox[2] <= container_bbox[2] and bbox[3] <= container_bbox[3])


def _bbox_overlaps_hint(bbox: List[float], hint: List[float]) -> bool:
    if len(hint) < 4:
        return True
    return _bbox_iou(bbox, hint) >= SLOT_OVERLAP_MIN


def locate_fields_for_slot(
    slot: Slot,
    visual_candidates: List[List[float]],
    image_path: Optional[str] = None,
    dark_theme: bool = False,
    container_bbox: Optional[List[float]] = None,
) -> Optional[SlotAssignment]:
    """
    Для одного слота находит не более одного bbox. bbox обязан быть внутри FormContainer
    и пересекаться с slot.expected_bbox_hint. Кнопки не назначаются input-slot'ам.
    """
    if slot.role not in ("input_slot", "textarea_slot"):
        return SlotAssignment(slot=slot, bbox=None, field_type="input", confidence=0.0)

    slot_rect = slot.expected_bbox_hint if slot.expected_bbox_hint else _slot_bbox(slot)
    in_slot = [
        b for b in visual_candidates
        if len(b) >= 4 and _bbox_iou(b, slot_rect) >= SLOT_OVERLAP_MIN
    ]
    if container_bbox:
        in_slot = [b for b in in_slot if _bbox_inside_container(b, container_bbox)]
    if slot.expected_bbox_hint:
        in_slot = [b for b in in_slot if _bbox_overlaps_hint(b, slot.expected_bbox_hint)]
    if not in_slot:
        return SlotAssignment(slot=slot, bbox=None, field_type="input", confidence=0.0)

    best_bbox: Optional[List[float]] = None
    best_score = 0.0
    for b in in_slot:
        sc = _score_candidate_for_slot(b, slot, slot.role)
        if sc > best_score:
            best_score = sc
            best_bbox = list(b)

    if best_bbox is None or best_score <= 0:
        return SlotAssignment(slot=slot, bbox=None, field_type="input", confidence=0.0)

    field_type = "textarea" if slot.role == "textarea_slot" else "input"
    return SlotAssignment(slot=slot, bbox=best_bbox, field_type=field_type, confidence=min(1.0, best_score))


def run_role_based_locator(
    row_slots: List[RowSlots],
    visual_candidates: List[List[float]],
    image_path: Optional[str] = None,
    dark_theme: bool = False,
    container_bbox: Optional[List[float]] = None,
) -> Tuple[List[SlotAssignment], Dict[str, Any]]:
    """
    Для всех слотов находит не более одного bbox на слот. bbox только внутри container_bbox.
    """
    assignments: List[SlotAssignment] = []
    for rs in row_slots:
        for slot in rs.slots:
            if slot.role not in ("input_slot", "textarea_slot"):
                continue
            sa = locate_fields_for_slot(
                slot, visual_candidates, image_path, dark_theme, container_bbox=container_bbox,
            )
            assignments.append(sa)

    filled = sum(1 for a in assignments if a.bbox is not None)
    diag = {"total_input_slots": len(assignments), "filled": filled}
    return assignments, diag


def visualize_slot_assignments(
    image_path: str,
    assignments: List[SlotAssignment],
    output_path: str,
) -> None:
    """Визуализация связей слот → bbox (уровень 4)."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    from src.infrastructure.debug_draw import line_visible, rectangle_visible

    out = img.copy()
    for a in assignments:
        slot_rect = _slot_bbox(a.slot)
        rectangle_visible(
            out,
            (int(slot_rect[0]), int(slot_rect[1])),
            (int(slot_rect[2]), int(slot_rect[3])),
            (0, 140, 180), 1,
        )
        if a.bbox is not None:
            rectangle_visible(
                out,
                (int(a.bbox[0]), int(a.bbox[1])),
                (int(a.bbox[2]), int(a.bbox[3])),
                (0, 180, 0), 2,
            )
            cx_s = int((slot_rect[0] + slot_rect[2]) / 2)
            cy_s = int((slot_rect[1] + slot_rect[3]) / 2)
            cx_b = int((a.bbox[0] + a.bbox[2]) / 2)
            cy_b = int((a.bbox[1] + a.bbox[3]) / 2)
            line_visible(out, (cx_s, cy_s), (cx_b, cy_b), (0, 180, 180), 1)
    cv2.imwrite(output_path, out)
    logger.debug("role_based_field_locator: saved %s", output_path)
