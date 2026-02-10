"""
Level 4 — ElementLocator.

Тонкая обёртка над role_based_field_locator: специализация элементов только
внутри slot.expected_bbox_hint и container_bbox.
"""

from __future__ import annotations

from typing import List, Optional

from src.infrastructure.atoms_v2.experimental_v2.role_based_field_locator import (
    locate_fields_for_slot,
)
from src.infrastructure.atoms_v2.experimental_v2.models import Slot, SlotAssignment


class ElementLocator:
    """
    Level 4.
    Специализация элементов ТОЛЬКО внутри slot + container.

    Оборачивает существующую функцию locate_fields_for_slot без изменения логики.
    """

    def locate(
        self,
        visual_candidates: List[List[float]],
        slot: Slot,
        container_bbox: Optional[List[float]] = None,
    ) -> SlotAssignment:
        """
        Возвращает SlotAssignment для одного слота.
        """
        return locate_fields_for_slot(
            slot=slot,
            visual_candidates=visual_candidates,
            image_path=None,
            dark_theme=False,
            container_bbox=container_bbox,
        )

