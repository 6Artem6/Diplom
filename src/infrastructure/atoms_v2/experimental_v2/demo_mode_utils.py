"""
Утилиты demo_mode: сериализация в JSON, валидация пайплайна, визуализация.

Цвета: контейнер — синий, строки — жёлтые, label_slot — голубой,
input_slot — оранжевый, action — фиолетовый.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.models import (
    FormContainer,
    FormGraph,
    FormRow,
    FormSkeleton,
    RowSlots,
    Slot,
    SlotAssignment,
)

logger = logging.getLogger(__name__)

# Цвета для demo_visualization (BGR): тёмные, чтобы внутренняя обводка была видна на светлом фоне; на тёмном видна белая гало
DEMO_COLOR_CONTAINER = (180, 0, 0)       # тёмно-синий
DEMO_COLOR_ROWS = (0, 180, 180)         # тёмно-жёлтый
DEMO_COLOR_LABEL_SLOT = (180, 140, 0)   # тёмно-голубой
DEMO_COLOR_INPUT_SLOT = (0, 100, 200)   # тёмно-оранжевый
DEMO_COLOR_ACTION = (180, 0, 100)       # тёмно-фиолетовый


def _slot_to_dict(s: Slot) -> Dict[str, Any]:
    return {
        "slot_id": s.slot_id,
        "role": s.role,
        "row_index": s.row_index,
        "column_index": s.column_index,
        "x_min": s.x_min, "x_max": s.x_max, "y_min": s.y_min, "y_max": s.y_max,
        "width_hint": s.width_hint, "height_hint": s.height_hint,
        "expected_bbox_hint": list(s.expected_bbox_hint) if s.expected_bbox_hint else [],
    }


def _row_to_dict(r: FormRow) -> Dict[str, Any]:
    return {
        "row_index": r.row_index,
        "y_min": r.y_min, "y_max": r.y_max, "x_min": r.x_min, "x_max": r.x_max,
        "column_count": r.column_count,
        "row_type": r.row_type,
        "input_bbox": list(r.input_bbox) if r.input_bbox else None,
        "label_bbox": list(r.label_bbox) if r.label_bbox else None,
        "vertical_separators": r.vertical_separators,
    }


def save_demo_artifacts(
    debug_output_dir: str,
    container: FormContainer,
    skeleton: FormSkeleton,
    row_slots: List[RowSlots],
    assignments: List[SlotAssignment],
    graph: FormGraph,
) -> None:
    """Сохраняет demo_container.json, demo_rows.json, demo_slots.json, demo_slot_assignments.json, demo_form_graph.json."""
    import os
    os.makedirs(debug_output_dir, exist_ok=True)
    with open(os.path.join(debug_output_dir, "demo_container.json"), "w", encoding="utf-8") as f:
        json.dump({"bbox": list(container.bbox), "confidence": container.confidence}, f, indent=2)
    with open(os.path.join(debug_output_dir, "demo_rows.json"), "w", encoding="utf-8") as f:
        json.dump([_row_to_dict(r) for r in skeleton.rows], f, indent=2, ensure_ascii=False)
    slots_data = []
    for rs in row_slots:
        for s in rs.slots:
            slots_data.append({"row_index": rs.row_index, ** _slot_to_dict(s)})
    with open(os.path.join(debug_output_dir, "demo_slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots_data, f, indent=2)
    assign_data = []
    for a in assignments:
        assign_data.append({
            "slot_id": a.slot.slot_id,
            "role": a.slot.role,
            "bbox": list(a.bbox) if a.bbox else None,
            "field_type": a.field_type,
            "confidence": a.confidence,
        })
    with open(os.path.join(debug_output_dir, "demo_slot_assignments.json"), "w", encoding="utf-8") as f:
        json.dump(assign_data, f, indent=2)
    graph_data = {
        "label_to_input": list(graph.label_to_input),
        "input_to_helper": list(graph.input_to_helper),
        "row_edges": graph.metadata.get("row_edges", []),
    }
    with open(os.path.join(debug_output_dir, "demo_form_graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)


def validate_demo_pipeline(
    container: FormContainer,
    skeleton: FormSkeleton,
    row_slots: List[RowSlots],
    assignments: List[SlotAssignment],
    graph: FormGraph,
) -> Tuple[bool, List[str]]:
    """
    Проверки перед BPG в demo_mode.
    Возвращает (ok, list of error messages).
    """
    errors: List[str] = []
    field_rows = [r for r in skeleton.rows if r.row_type in ("FIELD_VERTICAL", "FIELD_HORIZONTAL", "FIELD_INPUT_ONLY", "FIELD")]
    input_slot_count = sum(1 for a in assignments if a.slot.role in ("input_slot", "textarea_slot") and a.bbox is not None)

    for r in field_rows:
        rs = next((rs for rs in row_slots if rs.row_index == r.row_index), None)
        if not rs:
            errors.append("FIELD row %d has no RowSlots" % r.row_index)
            continue
        input_slots = [s for s in rs.slots if s.role in ("input_slot", "textarea_slot")]
        if len(input_slots) != 1:
            errors.append("FIELD row %d must have exactly 1 input_slot, got %d" % (r.row_index, len(input_slots)))
        if r.input_bbox and len(r.input_bbox) >= 4 and input_slots:
            slot = input_slots[0]
            ix1, iy1, ix2, iy2 = r.input_bbox[0], r.input_bbox[1], r.input_bbox[2], r.input_bbox[3]
            if slot.x_min > ix1 or slot.x_max < ix2 or slot.y_min > iy1 or slot.y_max < iy2:
                errors.append("row %d: input_slot does not fully contain input_bbox" % r.row_index)
        label_slots = [s for s in rs.slots if s.role == "label_slot"]
        input_slots = [s for s in rs.slots if s.role in ("input_slot", "textarea_slot")]
        if label_slots and input_slots:
            ls, ins = label_slots[0], input_slots[0]
            overlap_x = max(0, min(ls.x_max, ins.x_max) - max(ls.x_min, ins.x_min))
            overlap_y = max(0, min(ls.y_max, ins.y_max) - max(ls.y_min, ins.y_min))
            if overlap_x > 0 and overlap_y > 0:
                errors.append("row %d: label_slot overlaps input_slot" % r.row_index)

    if len(field_rows) != input_slot_count:
        errors.append("FIELD row count (%d) != filled input_slot count (%d)" % (len(field_rows), input_slot_count))

    action_rows = [r for r in skeleton.rows if r.row_type == "ACTION"]
    for r in action_rows:
        rs = next((rs for rs in row_slots if rs.row_index == r.row_index), None)
        if rs and any(s.role == "input_slot" for s in rs.slots):
            errors.append("ACTION row %d must not have input_slot" % r.row_index)

    n_rows = len(skeleton.rows)
    row_edges = graph.metadata.get("row_edges", [])
    if n_rows > 1 and not row_edges:
        errors.append("form_graph has no row_edges (expected linear sequence for demo)")

    return len(errors) == 0, errors


def visualize_demo(
    image_path: str,
    container: FormContainer,
    skeleton: FormSkeleton,
    row_slots: List[RowSlots],
    output_path: str,
) -> None:
    """Визуализация demo: контейнер, строки, слоты с двойной обводкой и подписями, видимыми на любом фоне."""
    import cv2
    from src.infrastructure.debug_draw import putText_visible, rectangle_visible

    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    if len(container.bbox) >= 4:
        x1, y1, x2, y2 = int(container.bbox[0]), int(container.bbox[1]), int(container.bbox[2]), int(container.bbox[3])
        rectangle_visible(out, (x1, y1), (x2, y2), DEMO_COLOR_CONTAINER, 2)
    for r in skeleton.rows:
        rectangle_visible(out, (int(r.x_min), int(r.y_min)), (int(r.x_max), int(r.y_max)), DEMO_COLOR_ROWS, 1)
    for rs in row_slots:
        for slot in rs.slots:
            x1, y1 = int(slot.x_min), int(slot.y_min)
            x2, y2 = int(slot.x_max), int(slot.y_max)
            if slot.role == "label_slot":
                color = DEMO_COLOR_LABEL_SLOT
            elif slot.role in ("input_slot", "textarea_slot"):
                color = DEMO_COLOR_INPUT_SLOT
            elif slot.role == "action_slot":
                color = DEMO_COLOR_ACTION
            else:
                color = (80, 80, 80)
            rectangle_visible(out, (x1, y1), (x2, y2), color, 2)
            putText_visible(
                out,
                slot.role.replace("_slot", ""),
                (x1 + 2, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                (0, 0, 0),
                1,
            )
    cv2.imwrite(output_path, out)
    logger.debug("demo_mode: saved %s", output_path)
