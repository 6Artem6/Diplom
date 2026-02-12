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

# Цвета для demo_visualization (BGR): контрастные для разных типов элементов
DEMO_COLOR_CONTAINER = (180, 0, 0)       # синий — контейнер
DEMO_COLOR_ROWS = (0, 180, 180)          # жёлтый — строки (по умолчанию)
DEMO_COLOR_LABEL_SLOT = (255, 180, 0)    # голубой — label
DEMO_COLOR_INPUT_SLOT = (0, 200, 0)      # зелёный — input
DEMO_COLOR_ACTION = (0, 0, 255)          # красный — button/action
DEMO_COLOR_TEXTAREA = (0, 140, 255)      # оранжевый — textarea
DEMO_COLOR_CHECKBOX = (200, 0, 200)      # пурпурный — checkbox/radio
DEMO_COLOR_HEADER = (255, 100, 0)        # голубой яркий — header
DEMO_COLOR_SECTION = (150, 150, 150)     # серый — section

# Цвета для LeafElementDetection (BGR) — контрастные
LEAF_COLOR_BUTTON = (0, 0, 255)          # красный
LEAF_COLOR_INPUT = (0, 200, 0)           # зелёный
LEAF_COLOR_TEXTAREA = (0, 140, 255)      # оранжевый
LEAF_COLOR_CHECKBOX = (200, 0, 200)      # пурпурный
LEAF_COLOR_RADIO = (200, 0, 200)         # пурпурный (как checkbox)
LEAF_COLOR_LABEL = (255, 180, 0)         # голубой
LEAF_COLOR_SECTION = (150, 150, 150)     # серый
LEAF_COLOR_ELEMENT = (100, 100, 100)     # тёмно-серый

LEAF_COLORS = {
    "button": LEAF_COLOR_BUTTON,
    "input": LEAF_COLOR_INPUT,
    "field": LEAF_COLOR_INPUT,           # field → input цвет
    "textarea": LEAF_COLOR_TEXTAREA,
    "checkbox": LEAF_COLOR_CHECKBOX,
    "radio": LEAF_COLOR_RADIO,
    "label": LEAF_COLOR_LABEL,
    "section": LEAF_COLOR_SECTION,
    "element": LEAF_COLOR_ELEMENT,
}

# Маппинг row_type → цвет для визуализации
ROW_TYPE_COLORS = {
    "HEADER": DEMO_COLOR_HEADER,
    "TEXT": DEMO_COLOR_LABEL_SLOT,
    "FIELD": DEMO_COLOR_INPUT_SLOT,
    "FIELD_HORIZONTAL": DEMO_COLOR_INPUT_SLOT,
    "FIELD_VERTICAL": DEMO_COLOR_INPUT_SLOT,
    "FIELD_INPUT_ONLY": (0, 180, 100),   # зелёный светлый
    "TEXTAREA": DEMO_COLOR_TEXTAREA,
    "ACTION": DEMO_COLOR_ACTION,
    "SPACER": DEMO_COLOR_SECTION,
}


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
    result = {
        "row_index": r.row_index,
        "y_min": r.y_min, "y_max": r.y_max, "x_min": r.x_min, "x_max": r.x_max,
        "column_count": r.column_count,
        "row_type": r.row_type,
        "input_bbox": list(r.input_bbox) if r.input_bbox else None,
        "label_bbox": list(r.label_bbox) if r.label_bbox else None,
        "vertical_separators": r.vertical_separators,
    }
    # Добавить leaf_candidates если есть
    if hasattr(r, "metadata") and r.metadata:
        if "leaf_candidates" in r.metadata:
            result["leaf_candidates"] = r.metadata["leaf_candidates"]
        if "ocr_orientation" in r.metadata:
            result["ocr_orientation"] = r.metadata["ocr_orientation"]
    return result


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
    container_data = {"bbox": list(container.bbox), "confidence": container.confidence}
    # Добавить container_leaf если есть
    if hasattr(container, "metadata") and container.metadata:
        if "container_leaf" in container.metadata:
            container_data["container_leaf"] = container.metadata["container_leaf"]
    with open(os.path.join(debug_output_dir, "demo_container.json"), "w", encoding="utf-8") as f:
        json.dump(container_data, f, indent=2)
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
    # LeafElementDetection: показать кандидатов справа от строки
    for r in skeleton.rows:
        if not hasattr(r, "metadata") or not r.metadata:
            continue
        candidates = r.metadata.get("leaf_candidates", [])
        if not candidates:
            continue
        # Позиция текста — справа от строки
        text_x = int(r.x_max) + 5
        text_y = int(r.y_min) + 12
        for i, cand in enumerate(candidates):
            ctype = cand.get("type", "?")
            conf = cand.get("confidence", 0.0)
            color = LEAF_COLORS.get(ctype, (100, 100, 100))
            label = f"[L]{ctype}:{conf:.2f}"
            putText_visible(
                out,
                label,
                (text_x, text_y + i * 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                (255, 255, 255),
                1,
            )
    cv2.imwrite(output_path, out)
    logger.debug("demo_mode: saved %s", output_path)


def visualize_leaf_detection(
    image_path: str,
    skeleton: FormSkeleton,
    output_path: str,
) -> None:
    """
    Отдельная визуализация LeafElementDetection.
    Показывает строки и leaf_candidates с подробной информацией.
    """
    import cv2
    from src.infrastructure.debug_draw import putText_visible, rectangle_visible

    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()

    for r in skeleton.rows:
        # Рисуем границы строки
        rx1, ry1 = int(r.x_min), int(r.y_min)
        rx2, ry2 = int(r.x_max), int(r.y_max)

        # Получаем leaf_candidates
        candidates = []
        leaf_debug = {}
        if hasattr(r, "metadata") and r.metadata:
            candidates = r.metadata.get("leaf_candidates", [])
            leaf_debug = r.metadata.get("leaf_debug", {})

        # Цвет строки зависит от наличия кандидатов
        if candidates:
            # Есть leaf-кандидаты — подсветить строку цветом лучшего кандидата
            best = max(candidates, key=lambda c: c.get("confidence", 0))
            row_color = LEAF_COLORS.get(best.get("type"), DEMO_COLOR_ROWS)
            rectangle_visible(out, (rx1, ry1), (rx2, ry2), row_color, 2)
        else:
            # Нет кандидатов — серая обводка
            rectangle_visible(out, (rx1, ry1), (rx2, ry2), (120, 120, 120), 1)

        # Подпись row_type слева
        putText_visible(
            out,
            f"r{r.row_index}:{r.row_type}",
            (rx1 + 2, ry1 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            (0, 0, 0),
            1,
        )

        # Leaf candidates справа
        text_x = rx2 + 5
        text_y = ry1 + 12
        if candidates:
            for i, cand in enumerate(candidates):
                ctype = cand.get("type", "?")
                conf = cand.get("confidence", 0.0)
                color = LEAF_COLORS.get(ctype, (100, 100, 100))
                label = f"{ctype}:{conf:.2f}"
                putText_visible(
                    out,
                    label,
                    (text_x, text_y + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    (255, 255, 255),
                    1,
                )
        else:
            # Показать причину отсутствия
            detectors = leaf_debug.get("detectors", {})
            reasons = []
            for dtype, info in detectors.items():
                reason = info.get("reason", "")
                if reason and reason != "detected":
                    reasons.append(f"{dtype[:3]}:{reason[:15]}")
            if reasons:
                for i, reason in enumerate(reasons[:3]):  # max 3
                    putText_visible(
                        out,
                        reason,
                        (text_x, text_y + i * 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (100, 100, 100),
                        (255, 255, 255),
                        1,
                    )

    cv2.imwrite(output_path, out)
    logger.debug("leaf_detection: saved %s", output_path)


# Цвета для ContainerLeafDetection
CONTAINER_LEAF_COLOR_INSIDE = (100, 180, 100)   # зелёный (внутри rows)
CONTAINER_LEAF_COLOR_OUTSIDE = (0, 0, 220)      # красный (вне rows — потерянные)


def visualize_container_leaf_detection(
    image_path: str,
    container_bbox: List[float],
    container_leaf_result: Dict[str, Any],
    rows: Optional[List[FormRow]] = None,
    output_path: str = "",
) -> None:
    """
    Визуализация ContainerLeafDetection (Stage 1.1).

    Показывает:
    - Container bbox (синий)
    - Rows (жёлтые пунктиры)
    - Candidates inside_rows (зелёные)
    - Candidates outside_rows (красные — потерянные элементы)
    """
    import cv2
    from src.infrastructure.debug_draw import putText_visible, rectangle_visible

    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()

    # Container bbox
    if len(container_bbox) >= 4:
        cx1, cy1 = int(container_bbox[0]), int(container_bbox[1])
        cx2, cy2 = int(container_bbox[2]), int(container_bbox[3])
        rectangle_visible(out, (cx1, cy1), (cx2, cy2), DEMO_COLOR_CONTAINER, 2)

    # Rows (пунктиром)
    if rows:
        for r in rows:
            rx1, ry1 = int(r.x_min), int(r.y_min)
            rx2, ry2 = int(r.x_max), int(r.y_max)
            # Пунктирная линия эмулируется рисованием сегментов
            for x in range(rx1, rx2, 8):
                cv2.line(out, (x, ry1), (min(x + 4, rx2), ry1), DEMO_COLOR_ROWS, 1)
                cv2.line(out, (x, ry2), (min(x + 4, rx2), ry2), DEMO_COLOR_ROWS, 1)
            for y in range(ry1, ry2, 8):
                cv2.line(out, (rx1, y), (rx1, min(y + 4, ry2)), DEMO_COLOR_ROWS, 1)
                cv2.line(out, (rx2, y), (rx2, min(y + 4, ry2)), DEMO_COLOR_ROWS, 1)

    if not container_leaf_result:
        cv2.imwrite(output_path, out)
        return

    # Inside rows — зелёные
    inside_rows = container_leaf_result.get("inside_rows", [])
    for cand in inside_rows:
        bbox = cand.get("bbox", [])
        if len(bbox) >= 4:
            bx1, by1 = int(bbox[0]), int(bbox[1])
            bx2, by2 = int(bbox[2]), int(bbox[3])
            rectangle_visible(out, (bx1, by1), (bx2, by2), CONTAINER_LEAF_COLOR_INSIDE, 1)
            ctype = cand.get("type", "?")
            conf = cand.get("confidence", 0.0)
            putText_visible(
                out,
                f"[in]{ctype}:{conf:.2f}",
                (bx1 + 2, by1 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                CONTAINER_LEAF_COLOR_INSIDE,
                (255, 255, 255),
                1,
            )

    # Outside rows — красные (потерянные)
    outside_rows = container_leaf_result.get("outside_rows", [])
    for cand in outside_rows:
        bbox = cand.get("bbox", [])
        if len(bbox) >= 4:
            bx1, by1 = int(bbox[0]), int(bbox[1])
            bx2, by2 = int(bbox[2]), int(bbox[3])
            rectangle_visible(out, (bx1, by1), (bx2, by2), CONTAINER_LEAF_COLOR_OUTSIDE, 2)
            ctype = cand.get("type", "?")
            conf = cand.get("confidence", 0.0)
            putText_visible(
                out,
                f"[LOST]{ctype}:{conf:.2f}",
                (bx1 + 2, by1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                CONTAINER_LEAF_COLOR_OUTSIDE,
                (255, 255, 255),
                1,
            )

    # Статистика внизу
    n_all = len(container_leaf_result.get("all_candidates", []))
    n_inside = len(inside_rows)
    n_outside = len(outside_rows)
    stats_text = f"ContainerLeaf: all={n_all} inside={n_inside} LOST={n_outside}"
    putText_visible(
        out,
        stats_text,
        (10, out.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        (0, 0, 0),
        1,
    )

    cv2.imwrite(output_path, out)
    logger.debug("container_leaf_detection: saved %s", output_path)
