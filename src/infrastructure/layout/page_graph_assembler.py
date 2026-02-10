"""
Level 5 — PageGraphAssembler.

Тонкая обёртка над ui_graph: собирает граф страницы из regions/containers/slots/elements.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.infrastructure.ui_graph.build import build_ui_graph
from src.infrastructure.ui_graph.graph import UIGraph as PageGraph


class PageGraphAssembler:
    """
    Level 5.
    Обёртка над существующим ui_graph.build.build_ui_graph.

    Важно: не изменяет структуру графа, только фиксирует его как PageGraph.
    """

    def assemble(
        self,
        atoms: List[Dict[str, Any]],
        raw_ocr_boxes: List[Dict[str, Any]],
        regions: List[Dict[str, Any]],
    ) -> PageGraph:
        """
        Собирает PageGraph (UIGraph) из atoms + OCR + CV-регионов.
        """
        return build_ui_graph(atoms, raw_ocr_boxes, regions)

