"""
Построение рёбер UI-графа по правилам геометрии.

CONTAINS: region → atom (IoU ≥ 0.2), atom → OCR (OCR внутри atom bbox ≥ 20% area OCR).
ADJACENT: расстояние между bbox ≤ max(20px, 0.1*min(w,h)), без сильного overlap.
ALIGNED_ROW: |cy1-cy2| ≤ 0.2*max(h1,h2), overlap по X ≥ 30%.
ALIGNED_COL: |cx1-cx2| ≤ 0.2*max(w1,w2), overlap по Y ≥ 30%.
LABELED_BY: OCR слева/сверху от atom, расстояние ≤ 40px, overlap по ортогональной оси ≥ 50%.
PART_OF: atom → region (IoU ≥ 0.2).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.infrastructure.ui_graph.graph import UIGraph, EdgeType, AtomNode, OCRNode, RegionNode


# Пороги
CONTAINS_ATOM_REGION_IOU_MIN = 0.2
CONTAINS_OCR_IN_ATOM_COVERAGE_MIN = 0.2
ADJACENT_DIST_PX = 20
ADJACENT_DIST_RATIO = 0.1
ADJACENT_OVERLAP_MAX = 0.5  # сильный overlap — не adjacent
ALIGNED_ROW_Y_TOL_RATIO = 0.2
ALIGNED_ROW_X_OVERLAP_MIN = 0.3
ALIGNED_COL_X_TOL_RATIO = 0.2
ALIGNED_COL_Y_OVERLAP_MIN = 0.3
LABELED_BY_OFFSET_PX = 40
LABELED_BY_ORTH_OVERLAP_MIN = 0.5
PART_OF_IOU_MIN = 0.2


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


def _iou(a: List[float], b: List[float]) -> float:
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    inter = _intersection_area(a, b)
    union = area_a + area_b - inter
    return inter / max(1e-9, union)


def _center(bbox: List[float]) -> Tuple[float, float]:
    if len(bbox) < 4:
        return 0.0, 0.0
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _distance_bbox(a: List[float], b: List[float]) -> float:
    cx1, cy1 = _center(a)
    cx2, cy2 = _center(b)
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def _overlap_ratio_x(a: List[float], b: List[float]) -> float:
    """Доля перекрытия по X (относительно min ширины)."""
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix1 = max(a[0], b[0])
    ix2 = min(a[2], b[2])
    if ix2 <= ix1:
        return 0.0
    w_a = a[2] - a[0]
    w_b = b[2] - b[0]
    w_min = min(w_a, w_b)
    return (ix2 - ix1) / max(1e-9, w_min)


def _overlap_ratio_y(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    iy1 = max(a[1], b[1])
    iy2 = min(a[3], b[3])
    if iy2 <= iy1:
        return 0.0
    h_a = a[3] - a[1]
    h_b = b[3] - b[1]
    h_min = min(h_a, h_b)
    return (iy2 - iy1) / max(1e-9, h_min)


def _coverage_in_outer(inner: List[float], outer: List[float]) -> float:
    """Доля inner, попадающая в outer (intersection/area(inner))."""
    area_inner = _bbox_area(inner)
    if area_inner <= 0:
        return 0.0
    return _intersection_area(inner, outer) / area_inner


def _build_part_of_and_contains_region_atom(graph: UIGraph) -> None:
    """PART_OF: atom → region (IoU ≥ PART_OF_IOU_MIN). CONTAINS: region → atom (то же)."""
    for aid, atom in graph.atoms.items():
        abbox = atom.bbox
        if len(abbox) < 4:
            continue
        for rid, region in graph.regions.items():
            rbbox = region.bbox
            if len(rbbox) < 4:
                continue
            iou = _iou(abbox, rbbox)
            if iou >= PART_OF_IOU_MIN:
                graph.add_edge(aid, rid, EdgeType.PART_OF, {"iou": iou})
                graph.add_edge(rid, aid, EdgeType.CONTAINS, {"iou": iou})


def _build_contains_atom_ocr(graph: UIGraph) -> None:
    """CONTAINS: atom → OCR, если OCR внутри atom bbox ≥ CONTAINS_OCR_IN_ATOM_COVERAGE_MIN (доля площади OCR)."""
    for aid, atom in graph.atoms.items():
        abbox = atom.bbox
        if len(abbox) < 4:
            continue
        for oid, ocr in graph.ocr_nodes.items():
            obbox = ocr.bbox
            if len(obbox) < 4:
                continue
            cov = _coverage_in_outer(obbox, abbox)
            if cov >= CONTAINS_OCR_IN_ATOM_COVERAGE_MIN:
                graph.add_edge(aid, oid, EdgeType.CONTAINS, {"coverage": cov})


def _build_adjacent(graph: UIGraph) -> None:
    """ADJACENT: atom ↔ atom, расстояние ≤ max(20px, 0.1*min(w,h)), без сильного overlap."""
    atoms_list = list(graph.atoms.items())
    for i in range(len(atoms_list)):
        aid, a = atoms_list[i]
        abbox = a.bbox
        if len(abbox) < 4:
            continue
        wa, ha = abbox[2] - abbox[0], abbox[3] - abbox[1]
        dist_thresh = max(ADJACENT_DIST_PX, ADJACENT_DIST_RATIO * min(wa, ha))
        for j in range(i + 1, len(atoms_list)):
            bid, b = atoms_list[j]
            bbbox = b.bbox
            if len(bbbox) < 4:
                continue
            inter = _intersection_area(abbox, bbbox)
            area_a = _bbox_area(abbox)
            area_b = _bbox_area(bbbox)
            if area_a <= 0 or area_b <= 0:
                continue
            if inter / min(area_a, area_b) > ADJACENT_OVERLAP_MAX:
                continue
            dist = _distance_bbox(abbox, bbbox)
            if dist <= dist_thresh:
                graph.add_edge(aid, bid, EdgeType.ADJACENT, {"distance": dist})
                graph.add_edge(bid, aid, EdgeType.ADJACENT, {"distance": dist})


def _build_aligned_row(graph: UIGraph) -> None:
    """ALIGNED_ROW: atom ↔ atom, |cy1-cy2| ≤ 0.2*max(h1,h2), overlap по X ≥ 30%."""
    atoms_list = list(graph.atoms.items())
    for i in range(len(atoms_list)):
        aid, a = atoms_list[i]
        abbox = a.bbox
        if len(abbox) < 4:
            continue
        cy_a = (abbox[1] + abbox[3]) / 2
        ha = abbox[3] - abbox[1]
        for j in range(i + 1, len(atoms_list)):
            bid, b = atoms_list[j]
            bbbox = b.bbox
            if len(bbbox) < 4:
                continue
            cy_b = (bbbox[1] + bbbox[3]) / 2
            hb = bbbox[3] - bbbox[1]
            if abs(cy_a - cy_b) > ALIGNED_ROW_Y_TOL_RATIO * max(ha, hb):
                continue
            ov_x = _overlap_ratio_x(abbox, bbbox)
            if ov_x >= ALIGNED_ROW_X_OVERLAP_MIN:
                graph.add_edge(aid, bid, EdgeType.ALIGNED_ROW, {"overlap_x": ov_x})
                graph.add_edge(bid, aid, EdgeType.ALIGNED_ROW, {"overlap_x": ov_x})


def _build_aligned_col(graph: UIGraph) -> None:
    """ALIGNED_COL: atom ↔ atom, |cx1-cx2| ≤ 0.2*max(w1,w2), overlap по Y ≥ 30%."""
    atoms_list = list(graph.atoms.items())
    for i in range(len(atoms_list)):
        aid, a = atoms_list[i]
        abbox = a.bbox
        if len(abbox) < 4:
            continue
        cx_a = (abbox[0] + abbox[2]) / 2
        wa = abbox[2] - abbox[0]
        for j in range(i + 1, len(atoms_list)):
            bid, b = atoms_list[j]
            bbbox = b.bbox
            if len(bbbox) < 4:
                continue
            cx_b = (bbbox[0] + bbbox[2]) / 2
            wb = bbbox[2] - bbbox[0]
            if abs(cx_a - cx_b) > ALIGNED_COL_X_TOL_RATIO * max(wa, wb):
                continue
            ov_y = _overlap_ratio_y(abbox, bbbox)
            if ov_y >= ALIGNED_COL_Y_OVERLAP_MIN:
                graph.add_edge(aid, bid, EdgeType.ALIGNED_COL, {"overlap_y": ov_y})
                graph.add_edge(bid, aid, EdgeType.ALIGNED_COL, {"overlap_y": ov_y})


def _build_labeled_by(graph: UIGraph) -> None:
    """LABELED_BY: OCR слева или сверху от atom, расстояние ≤ 40px, overlap по ортогональной оси ≥ 50%."""
    for aid, atom in graph.atoms.items():
        abbox = atom.bbox
        if len(abbox) < 4:
            continue
        for oid, ocr in graph.ocr_nodes.items():
            obbox = ocr.bbox
            if len(obbox) < 4:
                continue
            # OCR слева от atom: ocr.x2 <= atom.x1, расстояние (atom.x1 - ocr.x2) <= 40px
            left_ok = obbox[2] <= abbox[0] and (abbox[0] - obbox[2]) <= LABELED_BY_OFFSET_PX
            overlap_y_left = _overlap_ratio_y(obbox, abbox) if left_ok else 0.0
            # OCR сверху от atom: ocr.y2 <= atom.y1, расстояние (atom.y1 - ocr.y2) <= 40px
            top_ok = obbox[3] <= abbox[1] and (abbox[1] - obbox[3]) <= LABELED_BY_OFFSET_PX
            overlap_x_top = _overlap_ratio_x(obbox, abbox) if top_ok else 0.0
            dist = _distance_bbox(abbox, obbox)
            if dist > LABELED_BY_OFFSET_PX:
                continue
            if left_ok and overlap_y_left >= LABELED_BY_ORTH_OVERLAP_MIN:
                graph.add_edge(aid, oid, EdgeType.LABELED_BY, {"side": "left", "overlap_y": overlap_y_left})
                graph.add_edge(oid, aid, EdgeType.LABELED_BY, {"side": "left", "overlap_y": overlap_y_left})
            elif top_ok and overlap_x_top >= LABELED_BY_ORTH_OVERLAP_MIN:
                graph.add_edge(aid, oid, EdgeType.LABELED_BY, {"side": "top", "overlap_x": overlap_x_top})
                graph.add_edge(oid, aid, EdgeType.LABELED_BY, {"side": "top", "overlap_x": overlap_x_top})


LABELED_BY_INTERNAL_COVERAGE_MIN = 0.3  # OCR покрывает ≥30% atom bbox → label внутри


def _build_labeled_by_internal(graph: UIGraph) -> None:
    """LABELED_BY_INTERNAL: OCR bbox покрывает ≥30% atom bbox — это label внутри, не просто текст."""
    for aid, atom in graph.atoms.items():
        abbox = atom.bbox
        if len(abbox) < 4:
            continue
        area_atom = _bbox_area(abbox)
        if area_atom <= 0:
            continue
        for oid, ocr in graph.ocr_nodes.items():
            obbox = ocr.bbox
            if len(obbox) < 4:
                continue
            inter = _intersection_area(obbox, abbox)
            cov = inter / area_atom
            if cov >= LABELED_BY_INTERNAL_COVERAGE_MIN:
                graph.add_edge(aid, oid, EdgeType.LABELED_BY_INTERNAL, {"coverage": cov})
                graph.add_edge(oid, aid, EdgeType.LABELED_BY_INTERNAL, {"coverage": cov})


def build_edges(graph: UIGraph) -> None:
    """Строит все рёбра графа по правилам. Модифицирует graph in-place."""
    _build_part_of_and_contains_region_atom(graph)
    _build_contains_atom_ocr(graph)
    _build_adjacent(graph)
    _build_aligned_row(graph)
    _build_aligned_col(graph)
    _build_labeled_by(graph)
    _build_labeled_by_internal(graph)
