"""
Уровень 3 — Определение слотов (ожидаемые зоны).

Для каждой строки скелета определяет абстрактные слоты: label_slot, input_slot, helper_slot, action_slot.
Слоты — ожидаемые зоны с примерной шириной/высотой/положением. Слот существует даже если поле не найдено.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.models import FormRow, FormSkeleton, RowSlots, Slot, SlotRole

logger = logging.getLogger(__name__)

# Доли ширины строки на слот (vertical: label слева, input справа)
LABEL_WIDTH_RATIO = 0.25
INPUT_WIDTH_RATIO = 0.65
HELPER_WIDTH_RATIO = 0.10
# I5: инварианты слота
MIN_INPUT_WIDTH_RATIO = 0.4
MIN_INPUT_RIGHT_RATIO = 0.9
# Высота слота по умолчанию = высота строки
# Зона label: слева до LABEL_MAX_LEFT_RATIO*container_width, сверху до LABEL_ABOVE_ROW_RATIO*row_height
LABEL_MAX_LEFT_RATIO = 0.35
LABEL_ABOVE_ROW_RATIO = 0.5
# Helper: top(helper) ∈ [bottom(input), bottom(input)+HELPER_BELOW_INPUT_RATIO*input_height]; не удалять до финала
HELPER_BELOW_INPUT_RATIO = 1.2
ACTION_WORDS = frozenset({"save", "submit", "search", "send", "add", "ok", "login", "cancel", "apply", "отправить", "сохранить", "войти"})


def _slot_id() -> str:
    return "slot_%s" % uuid.uuid4().hex[:8]


def infer_slots_for_row(
    row: FormRow,
    skeleton: FormSkeleton,
    ocr_in_row: Optional[List[Dict[str, Any]]] = None,
) -> RowSlots:
    """
    Для одной строки скелета определяет слоты по layout_type и опционально OCR (вспомогательно).
    ACTION-строка: только action_slot (кнопка не считается input-row).
    TEXTAREA-строка: label + textarea_slot, без разбиения по grid.
    """
    ocr_in_row = ocr_in_row or []
    row_w = row.x_max - row.x_min
    row_h = row.y_max - row.y_min
    slots: List[Slot] = []

    row_type = getattr(row, "row_type", "FIELD_HORIZONTAL")

    if row_type in ("TEXT", "HEADER"):
        return RowSlots(row_index=row.row_index, row_bbox=[row.x_min, row.y_min, row.x_max, row.y_max], slots=[])

    if row_type == "ACTION":
        slots.append(Slot(
            slot_id=_slot_id(),
            role="action_slot",
            row_index=row.row_index,
            column_index=0,
            x_min=row.x_min,
            x_max=row.x_max,
            y_min=row.y_min,
            y_max=row.y_max,
            width_hint=row_w,
            height_hint=row_h,
            expected_bbox_hint=[row.x_min, row.y_min, row.x_max, row.y_max],
            metadata={"row_type": "ACTION"},
        ))
        return RowSlots(row_index=row.row_index, row_bbox=[row.x_min, row.y_min, row.x_max, row.y_max], slots=slots)

    input_bboxes = getattr(row, "input_bboxes", None)
    if input_bboxes and len(input_bboxes) >= 2:
        role = "textarea_slot" if row_type == "TEXTAREA" else "input_slot"
        for col_idx, bbox in enumerate(input_bboxes):
            if len(bbox) >= 4:
                cx1, cy1, cx2, cy2 = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                cx1, cy1, cx2, cy2 = row.x_min, row.y_min, row.x_max, row.y_max
            slots.append(Slot(
                slot_id=_slot_id(),
                role=role,
                row_index=row.row_index,
                column_index=col_idx,
                x_min=cx1,
                x_max=cx2,
                y_min=cy1,
                y_max=cy2,
                width_hint=cx2 - cx1,
                height_hint=cy2 - cy1,
                expected_bbox_hint=[cx1, cy1, cx2, cy2],
                metadata={"layout": "grid_visual", "row_type": row_type},
            ))
        label_bbox = getattr(row, "label_bbox", None)
        if label_bbox and len(label_bbox) >= 4:
            slots.insert(0, Slot(
                slot_id=_slot_id(),
                role="label_slot",
                row_index=row.row_index,
                column_index=0,
                x_min=label_bbox[0],
                x_max=label_bbox[2],
                y_min=label_bbox[1],
                y_max=label_bbox[3],
                width_hint=label_bbox[2] - label_bbox[0],
                height_hint=label_bbox[3] - label_bbox[1],
                expected_bbox_hint=list(label_bbox),
                metadata={"layout": "grid_visual"},
            ))
        helper_bbox = getattr(row, "helper_bbox", None)
        if helper_bbox and len(helper_bbox) >= 4:
            slots.append(Slot(
                slot_id=_slot_id(),
                role="helper_slot",
                row_index=row.row_index,
                column_index=len(input_bboxes),
                x_min=helper_bbox[0],
                x_max=helper_bbox[2],
                y_min=helper_bbox[1],
                y_max=helper_bbox[3],
                width_hint=helper_bbox[2] - helper_bbox[0],
                height_hint=helper_bbox[3] - helper_bbox[1],
                expected_bbox_hint=list(helper_bbox),
                metadata={"layout": "grid_visual"},
            ))
        return RowSlots(row_index=row.row_index, row_bbox=[row.x_min, row.y_min, row.x_max, row.y_max], slots=slots)

    if row_type == "FIELD_INPUT_ONLY":
        hint = getattr(row, "input_bbox", None)
        if not hint or len(hint) < 4:
            hint = [row.x_min, row.y_min, row.x_max, row.y_max]
        slots.append(Slot(
            slot_id=_slot_id(),
            role="input_slot",
            row_index=row.row_index,
            column_index=0,
            x_min=hint[0],
            x_max=hint[2],
            y_min=hint[1],
            y_max=hint[3],
            width_hint=hint[2] - hint[0],
            height_hint=hint[3] - hint[1],
            expected_bbox_hint=list(hint),
            metadata={"row_type": "FIELD_INPUT_ONLY"},
        ))
        return RowSlots(row_index=row.row_index, row_bbox=[row.x_min, row.y_min, row.x_max, row.y_max], slots=slots)

    split_y = getattr(row, "vertical_split_y", None)
    if row_type == "FIELD_VERTICAL" and (split_y is not None or getattr(row, "input_bbox", None)):
        label_bbox = getattr(row, "label_bbox", None)
        input_bbox = getattr(row, "input_bbox", None)
        y_label_end = split_y if split_y is not None and row.y_min < split_y < row.y_max else (row.y_min + row_h * 0.3 if row_h else row.y_min)
        if label_bbox and len(label_bbox) >= 4:
            y_label_end = label_bbox[3]
        if input_bbox and len(input_bbox) >= 4:
            ix_min, iy_min, ix_max, iy_max = input_bbox[0], input_bbox[1], input_bbox[2], input_bbox[3]
        else:
            ix_min, iy_min, ix_max, iy_max = row.x_min, row.y_min, row.x_max, row.y_max
        slots.append(Slot(
            slot_id=_slot_id(),
            role="label_slot",
            row_index=row.row_index,
            column_index=0,
            x_min=row.x_min,
            x_max=row.x_max,
            y_min=row.y_min,
            y_max=y_label_end,
            width_hint=row_w,
            height_hint=y_label_end - row.y_min,
            expected_bbox_hint=[row.x_min, row.y_min, row.x_max, y_label_end],
            metadata={"row_type": "FIELD_VERTICAL"},
        ))
        slots.append(Slot(
            slot_id=_slot_id(),
            role="input_slot",
            row_index=row.row_index,
            column_index=0,
            x_min=ix_min,
            x_max=ix_max,
            y_min=iy_min,
            y_max=iy_max,
            width_hint=ix_max - ix_min,
            height_hint=iy_max - iy_min,
            expected_bbox_hint=[ix_min, iy_min, ix_max, iy_max],
            metadata={"row_type": "FIELD_VERTICAL"},
        ))
        helper_bbox = getattr(row, "helper_bbox", None)
        if helper_bbox and len(helper_bbox) >= 4:
            slots.append(Slot(
                slot_id=_slot_id(),
                role="helper_slot",
                row_index=row.row_index,
                column_index=0,
                x_min=helper_bbox[0],
                x_max=helper_bbox[2],
                y_min=helper_bbox[1],
                y_max=helper_bbox[3],
                width_hint=helper_bbox[2] - helper_bbox[0],
                height_hint=helper_bbox[3] - helper_bbox[1],
                expected_bbox_hint=list(helper_bbox),
                metadata={"row_type": "FIELD_VERTICAL"},
            ))
        return RowSlots(row_index=row.row_index, row_bbox=[row.x_min, row.y_min, row.x_max, row.y_max], slots=slots)

    if row.column_count > 1:
        # Grid: слоты по input_bboxes (вертикальные разделители из CV) или по skeleton.column_boundaries
        if getattr(row, "input_bboxes", None) and len(row.input_bboxes) >= row.column_count:
            for col_idx, bbox in enumerate(row.input_bboxes):
                if col_idx >= row.column_count:
                    break
                if len(bbox) >= 4:
                    cx1, cy1, cx2, cy2 = bbox[0], bbox[1], bbox[2], bbox[3]
                else:
                    cx1, cy1, cx2, cy2 = row.x_min, row.y_min, row.x_max, row.y_max
                role = "textarea_slot" if row_type == "TEXTAREA" else "input_slot"
                slots.append(Slot(
                    slot_id=_slot_id(),
                    role=role,
                    row_index=row.row_index,
                    column_index=col_idx,
                    x_min=cx1,
                    x_max=cx2,
                    y_min=cy1,
                    y_max=cy2,
                    width_hint=cx2 - cx1,
                    height_hint=cy2 - cy1,
                    expected_bbox_hint=[cx1, cy1, cx2, cy2],
                    metadata={"layout": "grid", "row_type": row_type, "vertical_separators": getattr(row, "vertical_separators", None)},
                ))
        elif skeleton.column_boundaries:
            for col_idx, (cx1, cx2) in enumerate(skeleton.column_boundaries):
                if col_idx >= row.column_count:
                    break
                hint = [cx1, row.y_min, cx2, row.y_max]
                role = "textarea_slot" if row_type == "TEXTAREA" else "input_slot"
                slots.append(Slot(
                    slot_id=_slot_id(),
                    role=role,
                    row_index=row.row_index,
                    column_index=col_idx,
                    x_min=cx1,
                    x_max=cx2,
                    y_min=row.y_min,
                    y_max=row.y_max,
                    width_hint=cx2 - cx1,
                    height_hint=row_h,
                    expected_bbox_hint=hint,
                    metadata={"layout": "grid", "row_type": row_type},
                ))
    elif row_type == "FIELD_HORIZONTAL" and getattr(row, "right_label_bbox", None) and len(row.right_label_bbox) >= 4 and getattr(row, "input_bbox", None) and len(row.input_bbox) >= 4:
        # Деление вертикальными границами: input слева, label справа
        ib = row.input_bbox
        rlb = row.right_label_bbox
        slots.append(Slot(
            slot_id=_slot_id(),
            role="input_slot",
            row_index=row.row_index,
            column_index=0,
            x_min=ib[0],
            x_max=ib[2],
            y_min=ib[1],
            y_max=ib[3],
            width_hint=ib[2] - ib[0],
            height_hint=ib[3] - ib[1],
            expected_bbox_hint=list(ib),
            metadata={"row_type": "FIELD_HORIZONTAL", "label_right": True},
        ))
        slots.append(Slot(
            slot_id=_slot_id(),
            role="label_slot",
            row_index=row.row_index,
            column_index=1,
            x_min=rlb[0],
            x_max=rlb[2],
            y_min=rlb[1],
            y_max=rlb[3],
            width_hint=rlb[2] - rlb[0],
            height_hint=rlb[3] - rlb[1],
            expected_bbox_hint=list(rlb),
            metadata={"row_type": "FIELD_HORIZONTAL", "label_right": True},
        ))
        helper_bbox = getattr(row, "helper_bbox", None)
        if helper_bbox and len(helper_bbox) >= 4:
            slots.append(Slot(
                slot_id=_slot_id(),
                role="helper_slot",
                row_index=row.row_index,
                column_index=2,
                x_min=helper_bbox[0],
                x_max=helper_bbox[2],
                y_min=helper_bbox[1],
                y_max=helper_bbox[3],
                width_hint=helper_bbox[2] - helper_bbox[0],
                height_hint=helper_bbox[3] - helper_bbox[1],
                expected_bbox_hint=list(helper_bbox),
                metadata={"row_type": "FIELD_HORIZONTAL"},
            ))
    else:
        # Vertical: сначала input_slot (по input_bbox или row), затем label в зоне слева до LABEL_MAX_LEFT_RATIO*container
        helper_bbox = getattr(row, "helper_bbox", None)
        action_bbox = getattr(row, "action_bbox", None)
        has_action = any(
            (t.get("text") or "").strip().lower() in ACTION_WORDS
            for t in ocr_in_row
        ) or (action_bbox is not None and len(action_bbox) >= 4)
        has_helper = helper_bbox and len(helper_bbox) >= 4
        container_w = 0.0
        if skeleton and getattr(skeleton, "form_region", None) and getattr(skeleton.form_region, "bbox", None) and len(skeleton.form_region.bbox) >= 4:
            container_w = skeleton.form_region.bbox[2] - skeleton.form_region.bbox[0]
        input_bbox = getattr(row, "input_bbox", None)
        if input_bbox and len(input_bbox) >= 4:
            ix_min, iy_min, ix_max, iy_max = input_bbox[0], input_bbox[1], input_bbox[2], input_bbox[3]
            label_x_max = min(ix_min, row.x_min + container_w * LABEL_MAX_LEFT_RATIO) if container_w > 0 else ix_min
            label_x_max = max(label_x_max, row.x_min)
            lw = label_x_max - row.x_min
            hint_label = [row.x_min, row.y_min, label_x_max, row.y_max]
            slots.append(Slot(
                slot_id=_slot_id(),
                role="label_slot",
                row_index=row.row_index,
                column_index=0,
                x_min=row.x_min,
                x_max=label_x_max,
                y_min=row.y_min,
                y_max=row.y_max,
                width_hint=lw,
                height_hint=row_h,
                expected_bbox_hint=hint_label,
                metadata={},
            ))
            slot_role: SlotRole = "textarea_slot" if row_type == "TEXTAREA" else "input_slot"
            slots.append(Slot(
                slot_id=_slot_id(),
                role=slot_role,
                row_index=row.row_index,
                column_index=1,
                x_min=ix_min,
                x_max=ix_max,
                y_min=iy_min,
                y_max=iy_max,
                width_hint=ix_max - ix_min,
                height_hint=iy_max - iy_min,
                expected_bbox_hint=list(input_bbox),
                metadata={"row_type": row_type},
            ))
            x_cur = ix_max
        else:
            x_cur = row.x_min
            lw = row_w * LABEL_WIDTH_RATIO
            hint_label = [x_cur, row.y_min, x_cur + lw, row.y_max]
            slots.append(Slot(
                slot_id=_slot_id(),
                role="label_slot",
                row_index=row.row_index,
                column_index=0,
                x_min=x_cur,
                x_max=x_cur + lw,
                y_min=row.y_min,
                y_max=row.y_max,
                width_hint=lw,
                height_hint=row_h,
                expected_bbox_hint=hint_label,
                metadata={},
            ))
            x_cur += lw
            if not has_action and not has_helper:
                input_x_max = row.x_max
            else:
                input_x_max = x_cur + row_w * INPUT_WIDTH_RATIO
            min_input_right = row.x_min + row_w * MIN_INPUT_RIGHT_RATIO
            input_x_max = max(input_x_max, min_input_right, x_cur + row_w * MIN_INPUT_WIDTH_RATIO)
            input_x_max = min(input_x_max, row.x_max)
            iw = input_x_max - x_cur
            slot_role = "textarea_slot" if row_type == "TEXTAREA" else "input_slot"
            slots.append(Slot(
                slot_id=_slot_id(),
                role=slot_role,
                row_index=row.row_index,
                column_index=1,
                x_min=x_cur,
                x_max=input_x_max,
                y_min=row.y_min,
                y_max=row.y_max,
                width_hint=iw,
                height_hint=row_h,
                expected_bbox_hint=[x_cur, row.y_min, input_x_max, row.y_max],
                metadata={"row_type": row_type},
            ))
            x_cur = input_x_max
        if has_action:
            if action_bbox and len(action_bbox) >= 4:
                ax1, ay1, ax2, ay2 = action_bbox[0], action_bbox[1], action_bbox[2], action_bbox[3]
                slots.append(Slot(
                    slot_id=_slot_id(),
                    role="action_slot",
                    row_index=row.row_index,
                    column_index=2,
                    x_min=ax1,
                    x_max=ax2,
                    y_min=ay1,
                    y_max=ay2,
                    width_hint=ax2 - ax1,
                    height_hint=ay2 - ay1,
                    expected_bbox_hint=list(action_bbox),
                    metadata={"from_visual": True},
                ))
            else:
                slots.append(Slot(
                    slot_id=_slot_id(),
                    role="action_slot",
                    row_index=row.row_index,
                    column_index=2,
                    x_min=x_cur,
                    x_max=row.x_max,
                    y_min=row.y_min,
                    y_max=row.y_max,
                    width_hint=row.x_max - x_cur,
                    height_hint=row_h,
                    expected_bbox_hint=[x_cur, row.y_min, row.x_max, row.y_max],
                    metadata={},
                ))
        elif has_helper:
            slots.append(Slot(
                slot_id=_slot_id(),
                role="helper_slot",
                row_index=row.row_index,
                column_index=2,
                x_min=helper_bbox[0],
                x_max=helper_bbox[2],
                y_min=helper_bbox[1],
                y_max=helper_bbox[3],
                width_hint=helper_bbox[2] - helper_bbox[0],
                height_hint=helper_bbox[3] - helper_bbox[1],
                expected_bbox_hint=list(helper_bbox),
                metadata={},
            ))

    return RowSlots(row_index=row.row_index, row_bbox=[row.x_min, row.y_min, row.x_max, row.y_max], slots=slots)


def build_slot_layout(
    skeleton: FormSkeleton,
    ocr_boxes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[RowSlots], Dict[str, Any]]:
    """
    Строит слоты для всех строк скелета.
    """
    ocr_boxes = ocr_boxes or []
    form_bbox = skeleton.form_region.bbox
    if len(form_bbox) < 4:
        return [], {"error": "invalid_skeleton"}

    row_slots: List[RowSlots] = []
    for row in skeleton.rows:
        ocr_in_row = [
            ob for ob in ocr_boxes
            if len((ob.get("bbox") or [])) >= 4
            and row.y_min <= (ob["bbox"][1] + ob["bbox"][3]) / 2 <= row.y_max
            and row.x_min <= (ob["bbox"][0] + ob["bbox"][2]) / 2 <= row.x_max
        ]
        rs = infer_slots_for_row(row, skeleton, ocr_in_row)
        row_slots.append(rs)

    diag = {"n_rows": len(row_slots), "total_slots": sum(len(rs.slots) for rs in row_slots)}
    return row_slots, diag


def visualize_slot_layout(
    image_path: str,
    row_slots: List[RowSlots],
    output_path: str,
) -> None:
    """Визуализация слотов по строкам (уровень 3)."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    colors = {
        "label_slot": (100, 140, 0),
        "input_slot": (0, 140, 180),
        "textarea_slot": (0, 180, 140),
        "helper_slot": (140, 0, 140),
        "action_slot": (0, 100, 180),
    }
    from src.infrastructure.debug_draw import putText_visible, rectangle_visible

    for rs in row_slots:
        for slot in rs.slots:
            x1, y1 = int(slot.x_min), int(slot.y_min)
            x2, y2 = int(slot.x_max), int(slot.y_max)
            color = colors.get(slot.role, (80, 80, 80))
            rectangle_visible(out, (x1, y1), (x2, y2), color, 1)
            putText_visible(
                out, slot.role.replace("_slot", ""), (x1 + 2, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), (0, 0, 0), 1,
            )
    cv2.imwrite(output_path, out)
    logger.debug("slot_layout_inference: saved %s", output_path)
