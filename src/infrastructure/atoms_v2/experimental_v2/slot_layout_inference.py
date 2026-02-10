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
# Высота слота по умолчанию = высота строки
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

    row_type = getattr(row, "row_type", "FIELD")

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

    if skeleton.column_boundaries and row.column_count > 1:
        # Grid: каждый столбец — input_slot (TEXTAREA не разбиваем: column_count=1 задано выше)
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
    else:
        # Vertical: label_slot (слева) + input_slot (основная часть) + опционально helper / action
        x_cur = row.x_min
        # Label
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
        # Input или textarea (textarea не ломает grid — уже одна строка)
        iw = row_w * INPUT_WIDTH_RATIO
        hint_input = [x_cur, row.y_min, x_cur + iw, row.y_max]
        slot_role: SlotRole = "textarea_slot" if row_type == "TEXTAREA" else "input_slot"
        slots.append(Slot(
            slot_id=_slot_id(),
            role=slot_role,
            row_index=row.row_index,
            column_index=1,
            x_min=x_cur,
            x_max=x_cur + iw,
            y_min=row.y_min,
            y_max=row.y_max,
            width_hint=iw,
            height_hint=row_h,
            expected_bbox_hint=hint_input,
            metadata={"row_type": row_type},
        ))
        x_cur += iw
        # Helper или action (если в OCR есть кнопка в этой строке)
        has_action = any(
            (t.get("text") or "").strip().lower() in ACTION_WORDS
            for t in ocr_in_row
        )
        hint_action = [x_cur, row.y_min, row.x_max, row.y_max]
        if has_action:
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
                expected_bbox_hint=hint_action,
                metadata={},
            ))
        else:
            slots.append(Slot(
                slot_id=_slot_id(),
                role="helper_slot",
                row_index=row.row_index,
                column_index=2,
                x_min=x_cur,
                x_max=row.x_max,
                y_min=row.y_min,
                y_max=row.y_max,
                width_hint=row.x_max - x_cur,
                height_hint=row_h,
                expected_bbox_hint=hint_action,
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
        "label_slot": (200, 200, 100),
        "input_slot": (100, 200, 255),
        "textarea_slot": (100, 255, 200),
        "helper_slot": (200, 100, 200),
        "action_slot": (255, 150, 100),
    }
    for rs in row_slots:
        for slot in rs.slots:
            x1, y1 = int(slot.x_min), int(slot.y_min)
            x2, y2 = int(slot.x_max), int(slot.y_max)
            color = colors.get(slot.role, (128, 128, 128))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
            cv2.putText(out, slot.role.replace("_slot", ""), (x1 + 2, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.imwrite(output_path, out)
    logger.debug("slot_layout_inference: saved %s", output_path)
