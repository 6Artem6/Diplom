"""
Level 1 — SemanticRegionDetector.

Тонкая обёртка над существующим semantic_region_builder: фиксирует Semantic Regions
как отдельный архитектурный уровень (Page → Semantic Regions → ...).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.semantic_region_builder import (
    build_semantic_regions,
)
from src.infrastructure.atoms_v2.experimental_v2.models import Region as SemanticRegion


class SemanticRegionDetector:
    """
    Level 1.
    Thin wrapper над existing semantic_region_builder.build_semantic_regions.

    Важно: не добавляет новой логики, только проксирует вызов и возвращает список SemanticRegion.
    """

    def detect(
        self,
        image_path: str,
        segments: Optional[List[Any]] = None,
        page_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> List[SemanticRegion]:
        """
        Вызывает build_semantic_regions и возвращает только список SemanticRegion.
        segments/page_diagnostics — опциональные подсказки для реализации.
        """
        segments = segments or []
        regions, _diag = build_semantic_regions(
            image_path=image_path,
            segments=segments,
            page_diagnostics=page_diagnostics or {},
        )
        return regions

