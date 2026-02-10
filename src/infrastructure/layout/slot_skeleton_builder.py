"""
Level 3 — SlotSkeletonBuilder.

Тонкая обёртка над form_inner_layout + slot_layout_inference:
строит slot-скелет формы (label/input/helper/button) внутри контейнера.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.infrastructure.atoms_v2.experimental_v2.form_inner_layout import (
    build_form_inner_layout,
)
from src.infrastructure.atoms_v2.experimental_v2.slot_layout_inference import (
    build_slot_layout,
)
from src.infrastructure.atoms_v2.experimental_v2.models import (
    FormContainer,
    FormRow,
    FormColumn,
    RowSlots,
    Slot,
)


@dataclass
class SlotSkeleton:
    """
    Скелет формы: строки, колонки и слоты, привязанные к контейнеру.
    """

    container_id: str
    rows: List[FormRow]
    columns: List[FormColumn]
    row_slots: List[RowSlots]


class SlotSkeletonBuilder:
    """
    Level 3.
    Обёртка над существующими form_inner_layout + slot_layout_inference.
    """

    def build(
        self,
        image_path: str,
        container_id: str,
        container_bbox: List[float],
        raw_ocr_boxes: List[Dict[str, Any]],
    ) -> SlotSkeleton:
        """
        Строит SlotSkeleton для одного контейнера.
        """
        fc = FormContainer(bbox=list(container_bbox), confidence=1.0, metadata={"container_id": container_id})
        ocr_inside: List[Dict[str, Any]] = [
            ob
            for ob in raw_ocr_boxes
            if len((ob.get("bbox") or [])) >= 4
            and fc.bbox[0] <= (ob["bbox"][0] + ob["bbox"][2]) / 2 <= fc.bbox[2]
            and fc.bbox[1] <= (ob["bbox"][1] + ob["bbox"][3]) / 2 <= fc.bbox[3]
        ]

        skeleton, _diag = build_form_inner_layout(
            image_path=image_path,
            container=fc,
            ocr_inside=ocr_inside,
        )
        if not skeleton:
            return SlotSkeleton(container_id=container_id, rows=[], columns=[], row_slots=[])

        row_slots, _slot_diag = build_slot_layout(
            skeleton,
            raw_ocr_boxes,
        )
        columns: List[FormColumn] = skeleton.columns or []
        return SlotSkeleton(
            container_id=container_id,
            rows=skeleton.rows,
            columns=columns,
            row_slots=row_slots,
        )

