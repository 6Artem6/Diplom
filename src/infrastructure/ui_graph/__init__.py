"""
UI-граф: структурный слой между CV и финальной семантикой.

v3: semantic_validation — единственный источник ролей. ui_graph только структура (read-only семантика).
Граф агрегирует контекст (atoms, OCR, regions). Не назначает ui_role в v3 — только копирует semantic_role.
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
    run_ui_graph_pipeline_v3,
    run_ui_graph_pipeline,
    classify_roles,
    apply_roles_to_atoms,
)
from src.infrastructure.ui_graph.features import extract_features
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
    "run_ui_graph_pipeline_v3",
    "run_ui_graph_pipeline",
    "classify_roles",
    "apply_roles_to_atoms",
]
