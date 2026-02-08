"""
Debug UI-графа: per-atom лог, статистика, визуализация графа (nodes + edges).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.ui_graph.graph import UIGraph, EdgeType
from src.infrastructure.ui_graph.roles import UIRole

logger = logging.getLogger(__name__)


def debug_per_atom_log(
    graph: UIGraph,
    role_predictions: Dict[str, Tuple[UIRole, float]],
    features_by_atom: Dict[str, Dict[str, float]],
) -> List[str]:
    """
    Строки для лога: atom_id | atom.type | ui_role | key_features.
    """
    lines: List[str] = []
    for aid, atom in graph.atoms.items():
        pred = role_predictions.get(aid)
        role_str = pred[0].value if pred else "—"
        conf_str = f"{pred[1]:.2f}" if pred else "—"
        feats = features_by_atom.get(aid, {})
        key_feats = " ".join(
            f"{k}={feats.get(k, 0):.2f}" for k in ("aspect_ratio", "num_adjacent", "num_aligned_row", "has_label", "row_group_size")
        )
        lines.append(f"ui_graph atom_id={aid} | atom.type={atom.type} | ui_role={role_str} conf={conf_str} | {key_feats}")
    return lines


def debug_stats(
    atoms_before: List[Dict[str, Any]],
    final_atoms: List[Dict[str, Any]],
    role_predictions: Dict[str, Tuple[UIRole, float]],
) -> Dict[str, Any]:
    """
    Статистика: сколько button -> control_group, input -> noise, и т.д.
    """
    override_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    for aid, (role, _) in role_predictions.items():
        role_counts[role.value] = role_counts.get(role.value, 0) + 1
    atom_by_id = {a.get("id"): a for a in atoms_before if a.get("id")}
    for aid, (role, _) in role_predictions.items():
        a = atom_by_id.get(aid)
        if not a:
            continue
        t = a.get("type", "")
        if t != role.value:
            key = f"{t}_to_{role.value}"
            override_counts[key] = override_counts.get(key, 0) + 1
    return {
        "before_count": len(atoms_before),
        "after_count": len(final_atoms),
        "role_counts": role_counts,
        "override_counts": override_counts,
    }


def debug_graph_summary(graph: UIGraph) -> List[str]:
    """Краткое описание графа: число узлов и рёбер по типам."""
    lines = [
        f"ui_graph nodes: atoms={len(graph.atoms)} ocr={len(graph.ocr_nodes)} regions={len(graph.regions)}",
        f"ui_graph edges: total={len(graph.edges)}",
    ]
    by_type: Dict[EdgeType, int] = {}
    for e in graph.edges:
        by_type[e.edge_type] = by_type.get(e.edge_type, 0) + 1
    for et, count in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"  {et.value}={count}")
    return lines
