"""
Вычисляемые признаки (feature extraction) для каждого AtomNode.

Геометрия: aspect_ratio, area, bbox_coverage_ocr, relative_size_to_region.
Контекст: num_adjacent, num_aligned_row, num_aligned_col, num_inputs_nearby, num_buttons_nearby, is_inside_region, region_density.
Текст: has_label, has_action_word, text_length.
Паттерны: row_group_size, column_group_size, uniform_spacing_score, mixed_types_in_row.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from src.infrastructure.ui_graph.graph import UIGraph, EdgeType, AtomNode


ACTION_WORDS: Set[str] = {
    "ok", "save", "send", "submit", "login", "search", "next", "back", "cancel", "add", "delete",
    "edit", "create", "update", "apply", "confirm", "close", "done", "find", "open", "continue",
    "retry", "reset", "clear", "copy", "paste", "download", "upload", "signin", "signup", "filter",
    "sort", "refresh", "reload", "register",
}


def _bbox_area(bbox: List[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _intersection_area(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _coverage_in_outer(inner: List[float], outer: List[float]) -> float:
    area_inner = _bbox_area(inner)
    if area_inner <= 0:
        return 0.0
    return _intersection_area(inner, outer) / area_inner


def _extract_geometry(atom: AtomNode, graph: UIGraph) -> Dict[str, float]:
    bbox = atom.bbox
    if len(bbox) < 4:
        return {
            "aspect_ratio": 0.0, "area": 0.0, "bbox_coverage_ocr": 0.0, "relative_size_to_region": 0.0,
            "ocr_inside_count": 0.0, "ocr_inside_mean_conf": 0.0, "ocr_inside_text_len": 0.0,
        }
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    aspect_ratio = w / max(1e-9, h)
    area = _bbox_area(bbox)
    ocr_inside = 0.0
    ocr_confs: List[float] = []
    ocr_lens: List[int] = []
    for e in graph.edges_from(atom.id, EdgeType.CONTAINS):
        if e.dst_id in graph.ocr_nodes:
            ocr_node = graph.ocr_nodes[e.dst_id]
            ocr_inside += _intersection_area(ocr_node.bbox, bbox)
            ocr_confs.append(ocr_node.confidence)
            ocr_lens.append(len((ocr_node.text or "").strip()))
    bbox_coverage_ocr = ocr_inside / max(area, 1e-9)
    ocr_inside_count = float(len(ocr_confs))
    ocr_inside_mean_conf = sum(ocr_confs) / len(ocr_confs) if ocr_confs else 0.0
    ocr_inside_text_len = sum(ocr_lens) if ocr_lens else 0
    relative_size_to_region = 0.0
    for e in graph.edges_to(atom.id, EdgeType.PART_OF):
        if e.src_id in graph.regions:
            reg = graph.regions[e.src_id]
            reg_area = _bbox_area(reg.bbox)
            if reg_area > 0:
                relative_size_to_region = area / reg_area
                break
    return {
        "aspect_ratio": aspect_ratio,
        "area": area,
        "bbox_coverage_ocr": min(1.0, bbox_coverage_ocr),
        "relative_size_to_region": min(1.0, relative_size_to_region),
        "ocr_inside_count": ocr_inside_count,
        "ocr_inside_mean_conf": ocr_inside_mean_conf,
        "ocr_inside_text_len": float(ocr_inside_text_len),
    }


def _extract_context(atom: AtomNode, graph: UIGraph) -> Dict[str, Any]:
    adj = graph.neighbors(atom.id, EdgeType.ADJACENT)
    row = graph.neighbors(atom.id, EdgeType.ALIGNED_ROW)
    col = graph.neighbors(atom.id, EdgeType.ALIGNED_COL)
    num_inputs_nearby = sum(1 for nid in adj if nid in graph.atoms and graph.atoms[nid].type == "input")
    num_buttons_nearby = sum(1 for nid in adj if nid in graph.atoms and graph.atoms[nid].type == "button")
    is_inside_region = any(e.edge_type == EdgeType.PART_OF for e in graph.edges_from(atom.id))
    region_density = 0.0
    for e in graph.edges_from(atom.id):
        if e.edge_type == EdgeType.PART_OF and e.dst_id in graph.regions:
            rid = e.dst_id
            atoms_in_r = [aid for aid, a in graph.atoms.items() if any(ee.dst_id == rid for ee in graph.edges_from(aid) if ee.edge_type == EdgeType.PART_OF)]
            reg = graph.regions[rid]
            reg_area = _bbox_area(reg.bbox)
            region_density = len(atoms_in_r) / max(reg_area / 10000.0, 1e-9)
            break
    return {
        "num_adjacent": len(adj),
        "num_aligned_row": len(row),
        "num_aligned_col": len(col),
        "num_inputs_nearby": num_inputs_nearby,
        "num_buttons_nearby": num_buttons_nearby,
        "is_inside_region": 1.0 if is_inside_region else 0.0,
        "region_density": region_density,
    }


def _extract_text(atom: AtomNode, graph: UIGraph) -> Dict[str, Any]:
    has_label = any(
        e.edge_type in (EdgeType.LABELED_BY, EdgeType.LABELED_BY_INTERNAL)
        for e in graph.edges_from(atom.id)
    ) or any(
        e.edge_type in (EdgeType.LABELED_BY, EdgeType.LABELED_BY_INTERNAL)
        for e in graph.edges_to(atom.id)
    )
    texts: List[str] = []
    for e in graph.edges_from(atom.id):
        if e.edge_type == EdgeType.CONTAINS and e.dst_id in graph.ocr_nodes:
            texts.append(graph.ocr_nodes[e.dst_id].text)
        if e.edge_type == EdgeType.LABELED_BY_INTERNAL and e.dst_id in graph.ocr_nodes:
            texts.append(graph.ocr_nodes[e.dst_id].text)
    for e in graph.edges_to(atom.id):
        if e.edge_type == EdgeType.LABELED_BY and e.src_id in graph.ocr_nodes:
            texts.append(graph.ocr_nodes[e.src_id].text)
        if e.edge_type == EdgeType.LABELED_BY_INTERNAL and e.src_id in graph.ocr_nodes:
            texts.append(graph.ocr_nodes[e.src_id].text)
    text_joined = " ".join(t for t in texts if t).strip()
    tokens = set(re.sub(r"[^\w\s]", " ", text_joined.lower()).split())
    has_action_word = 1.0 if (tokens & ACTION_WORDS) else 0.0
    return {
        "has_label": 1.0 if has_label else 0.0,
        "has_action_word": has_action_word,
        "text_length": len(text_joined),
    }


def _extract_patterns(atom: AtomNode, graph: UIGraph) -> Dict[str, Any]:
    row_neighbors = graph.neighbors(atom.id, EdgeType.ALIGNED_ROW)
    col_neighbors = graph.neighbors(atom.id, EdgeType.ALIGNED_COL)
    row_group_size = 1 + len(row_neighbors)
    col_group_size = 1 + len(col_neighbors)
    bbox = atom.bbox
    if len(bbox) < 4:
        return {
            "row_group_size": row_group_size,
            "column_group_size": col_group_size,
            "uniform_spacing_score": 0.0,
            "mixed_types_in_row": 0.0,
        }
    row_atoms = [graph.atoms[nid] for nid in row_neighbors if nid in graph.atoms]
    row_atoms.append(atom)
    cxs = [(a.bbox[0] + a.bbox[2]) / 2 for a in row_atoms if len(a.bbox) >= 4]
    cxs.sort()
    if len(cxs) < 2:
        uniform_spacing_score = 0.0
    else:
        steps = [cxs[i + 1] - cxs[i] for i in range(len(cxs) - 1)]
        avg = sum(steps) / len(steps)
        var = sum((s - avg) ** 2 for s in steps) / len(steps)
        cv = (var ** 0.5 / avg) if avg else 1.0
        uniform_spacing_score = max(0.0, 1.0 - cv)
    types_in_row = {a.type for a in row_atoms}
    mixed_types_in_row = 1.0 if len(types_in_row) >= 2 and ("button" in types_in_row or "input" in types_in_row) else 0.0
    return {
        "row_group_size": row_group_size,
        "column_group_size": col_group_size,
        "uniform_spacing_score": uniform_spacing_score,
        "mixed_types_in_row": mixed_types_in_row,
    }


def extract_features(graph: UIGraph) -> Dict[str, Dict[str, float]]:
    """
    Для каждого AtomNode возвращает feature vector (плоский dict).
    Ключ — atom_id.
    """
    out: Dict[str, Dict[str, float]] = {}
    for aid, atom in graph.atoms.items():
        geo = _extract_geometry(atom, graph)
        ctx = _extract_context(atom, graph)
        txt = _extract_text(atom, graph)
        pat = _extract_patterns(atom, graph)
        flat: Dict[str, float] = {}
        flat.update(geo)
        flat.update(ctx)
        flat.update(txt)
        flat.update(pat)
        out[aid] = flat
    return out
