"""
UI-граф: структурный слой между CV и финальной семантикой.

Граф агрегирует контекст (atoms, OCR, regions) и даёт признаки для классификации ролей.
Не создаёт bbox, не меняет CV, не заменяет модели.
ui_role ≠ atom.type: atom.type — CV-гипотеза, ui_role — итоговая семантика.
"""

from src.infrastructure.ui_graph.graph import (
    UIGraph,
    AtomNode,
    OCRNode,
    RegionNode,
    EdgeType,
)
from src.infrastructure.ui_graph.build import (
    build_ui_graph,
    extract_features,
    classify_roles,
    apply_roles_to_atoms,
    run_ui_graph_pipeline,
)
from src.infrastructure.ui_graph.roles import UIRole

__all__ = [
    "UIGraph",
    "AtomNode",
    "OCRNode",
    "RegionNode",
    "EdgeType",
    "UIRole",
    "build_ui_graph",
    "extract_features",
    "classify_roles",
    "apply_roles_to_atoms",
    "run_ui_graph_pipeline",
]
