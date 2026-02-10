"""
Level 2 — StructuralContainerDetector.

Тонкая обёртка над form_container_detector: контейнеры формы как структурные элементы
внутри семантического региона.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.infrastructure.atoms_v2.experimental_v2.form_container_detector import (
    detect_form_containers,
)
from src.infrastructure.atoms_v2.experimental_v2.models import FormContainer


@dataclass
class Container:
    """
    Структурный контейнер (тонкий wrapper над FormContainer) с привязкой к region_id.
    """

    id: str
    region_id: str
    type: str
    bbox: List[float]
    confidence: float
    metadata: Dict[str, Any]


class StructuralContainerDetector:
    """
    Level 2.
    Детекция структурных контейнеров внутри semantic region.

    Логика детекции берётся из detect_form_containers; этот класс только
    добавляет привязку к region_id и типу "form" без изменения поведения.
    """

    def detect(
        self,
        image_path: str,
        region_id: str,
        detectron_regions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Container]:
        """
        Возвращает список Container, оборачивающих FormContainer, с указанным region_id.
        """
        form_containers, _diag = detect_form_containers(
            image_path=image_path,
            detectron_regions=detectron_regions,
        )
        containers: List[Container] = []
        for idx, fc in enumerate(form_containers):
            containers.append(
                Container(
                    id=f"container_{region_id}_{idx}",
                    region_id=region_id,
                    type="form",
                    bbox=fc.bbox,
                    confidence=fc.confidence,
                    metadata=dict(fc.metadata or {}),
                )
            )
        return containers

