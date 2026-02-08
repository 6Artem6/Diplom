"""
UI-граф: узлы и рёбра. Ориентированный многослойный граф.

Узлы: AtomNode, OCRNode, RegionNode.
Рёбра: CONTAINS, ADJACENT, ALIGNED_ROW, ALIGNED_COL, LABELED_BY, PART_OF.
Граф не создаёт bbox, не меняет CV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EdgeType(str, Enum):
    CONTAINS = "contains"             # region → atom, atom → OCR
    ADJACENT = "adjacent"             # atom ↔ atom
    ALIGNED_ROW = "aligned_row"       # atom ↔ atom
    ALIGNED_COL = "aligned_col"       # atom ↔ atom
    LABELED_BY = "labeled_by"         # atom ↔ OCR (снаружи: слева/сверху)
    LABELED_BY_INTERNAL = "labeled_by_internal"  # OCR внутри atom bbox ≥ 30% — label
    PART_OF = "part_of"               # atom → region


@dataclass
class AtomNode:
    """Узел атома CV (Detectron2)."""
    id: str
    type: str  # CV-гипотеза: button, input, link, ...
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    source: str = "real"  # real | synthetic


@dataclass
class OCRNode:
    """Узел OCR-бокса."""
    id: str
    text: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float = 0.0


@dataclass
class RegionNode:
    """Узел CV-региона (прямоугольный/скруглённый контур)."""
    id: str
    bbox: List[float]  # [x1, y1, x2, y2]
    shape_type: str = "rect"  # rect | rounded


@dataclass
class Edge:
    """Ориентированное ребро: src_id, dst_id, edge_type. Узлы идентифицируются по id + префиксу (atom_, ocr_, region_)."""
    src_id: str
    dst_id: str
    edge_type: EdgeType
    payload: Optional[Dict[str, Any]] = None  # опционально: overlap, distance, etc.


@dataclass
class UIGraph:
    """
    UI-граф: узлы по типам, рёбра списком.
    Хранит структуру данных; построение рёбер — в edges.py, признаки — в features.py.
    """
    atoms: Dict[str, AtomNode] = field(default_factory=dict)
    ocr_nodes: Dict[str, OCRNode] = field(default_factory=dict)
    regions: Dict[str, RegionNode] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_atom(self, node: AtomNode) -> None:
        self.atoms[node.id] = node

    def add_ocr(self, node: OCRNode) -> None:
        self.ocr_nodes[node.id] = node

    def add_region(self, node: RegionNode) -> None:
        self.regions[node.id] = node

    def add_edge(self, src_id: str, dst_id: str, edge_type: EdgeType, payload: Optional[Dict[str, Any]] = None) -> None:
        self.edges.append(Edge(src_id=src_id, dst_id=dst_id, edge_type=edge_type, payload=payload))

    def edges_from(self, src_id: str, edge_type: Optional[EdgeType] = None) -> List[Edge]:
        out = [e for e in self.edges if e.src_id == src_id]
        if edge_type is not None:
            out = [e for e in out if e.edge_type == edge_type]
        return out

    def edges_to(self, dst_id: str, edge_type: Optional[EdgeType] = None) -> List[Edge]:
        out = [e for e in self.edges if e.dst_id == dst_id]
        if edge_type is not None:
            out = [e for e in out if e.edge_type == edge_type]
        return out

    def neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None, direction: str = "both") -> List[str]:
        """direction: out | in | both."""
        ids: List[str] = []
        if direction in ("out", "both"):
            for e in self.edges_from(node_id, edge_type):
                ids.append(e.dst_id)
        if direction in ("in", "both"):
            for e in self.edges_to(node_id, edge_type):
                ids.append(e.src_id)
        return list(dict.fromkeys(ids))
