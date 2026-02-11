"""
Уровень 5 — Сборка графа формы.

Связывает slot → bbox, label → input, input → helper.
Финальный источник истины — граф формы, а не список полей.
"""

from __future__ import annotations

import hashlib
import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.models import (
    FormGraph,
    FormSkeleton,
    RowSlots,
    SlotAssignment,
)

logger = logging.getLogger(__name__)

# Граф строится по центрам input_slot. Edge(rowA→rowB) если vertical_distance < 1.8*median_row_height и overlap_x > 0.3
FORM_GRAPH_ROW_EDGE_VERTICAL_RATIO = 1.8
FORM_GRAPH_ROW_EDGE_OVERLAP_X = 0.3


def _row_edges_from_input_slot_centers(
    row_slots: List[RowSlots],
    assignments: List[SlotAssignment],
) -> List[Tuple[int, int]]:
    """Рёбра между строками по центрам input_slot (не по средней линии строки)."""
    slot_to_bbox: Dict[str, List[float]] = {}
    for a in assignments:
        if a.slot.slot_id and a.bbox is not None and len(a.bbox) >= 4:
            slot_to_bbox[a.slot.slot_id] = a.bbox
    rows_with_inputs: List[Tuple[int, float, float, float, float]] = []
    for rs in row_slots:
        for slot in rs.slots:
            if slot.role not in ("input_slot", "textarea_slot"):
                continue
            bbox = slot_to_bbox.get(slot.slot_id) or (slot.expected_bbox_hint if slot.expected_bbox_hint and len(slot.expected_bbox_hint) >= 4 else None)
            if bbox is None:
                continue
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            rows_with_inputs.append((rs.row_index, cx, cy, w, rs.row_bbox[2] - rs.row_bbox[0] if len(rs.row_bbox) >= 4 else w))
            break
    if len(rows_with_inputs) < 2:
        return []
    heights = [abs(r[4]) for r in rows_with_inputs if r[4] > 0]
    median_row_h = statistics.median(heights) if heights else 80.0
    max_vert = median_row_h * FORM_GRAPH_ROW_EDGE_VERTICAL_RATIO
    edges: List[Tuple[int, int]] = []
    for i in range(len(rows_with_inputs)):
        ri, cxi, cyi, wi, _ = rows_with_inputs[i]
        for j in range(i + 1, len(rows_with_inputs)):
            rj, cxj, cyj, wj, _ = rows_with_inputs[j]
            vert_dist = abs(cyj - cyi)
            if vert_dist >= max_vert:
                continue
            overlap_x = max(0, min(cxi + wi / 2, cxj + wj / 2) - max(cxi - wi / 2, cxj - wj / 2)) / max(wi, wj, 1e-9)
            if overlap_x >= FORM_GRAPH_ROW_EDGE_OVERLAP_X:
                edges.append((ri, rj))
    return edges


def assemble_form_graph(
    skeleton: FormSkeleton,
    row_slots: List[RowSlots],
    assignments: List[SlotAssignment],
) -> FormGraph:
    """
    Собирает граф формы: слоты, назначения slot→bbox, связи label→input, input→helper.
    """
    label_to_input: List[Tuple[str, str]] = []
    input_to_helper: List[Tuple[str, str]] = []

    slot_by_id: Dict[str, Any] = {}
    for rs in row_slots:
        for slot in rs.slots:
            slot_by_id[slot.slot_id] = slot

    for rs in row_slots:
        label_slots = [s for s in rs.slots if s.role == "label_slot"]
        input_slots = [s for s in rs.slots if s.role == "input_slot"]
        helper_slots = [s for s in rs.slots if s.role == "helper_slot"]
        for il in input_slots:
            if label_slots:
                label_to_input.append((label_slots[0].slot_id, il.slot_id))
            if helper_slots:
                input_to_helper.append((il.slot_id, helper_slots[0].slot_id))

    row_edges = _row_edges_from_input_slot_centers(row_slots, assignments)
    metadata: Dict[str, Any] = {"source": "experimental_multilevel_v2", "row_edges": row_edges}

    return FormGraph(
        skeleton=skeleton,
        row_slots=row_slots,
        assignments=assignments,
        label_to_input=label_to_input,
        input_to_helper=input_to_helper,
        metadata=metadata,
    )


def form_graph_to_atoms(
    graph: FormGraph,
    existing_ids: Optional[set] = None,
    recovery_source: str = "experimental_multilevel_v2",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Преобразует граф формы в список атомов. Только назначения с непустым bbox.
    """
    existing_ids = existing_ids or set()
    atoms: List[Dict[str, Any]] = []
    log_lines: List[str] = []
    prefix = "fcf_" if recovery_source == "form_container_first" else "exp_v2_"

    for a in graph.assignments:
        if a.bbox is None:
            continue
        bbox = a.bbox
        atom_type = "textarea_candidate" if a.field_type == "textarea" else "input_candidate"
        aid = "%s%s" % (prefix, hashlib.sha256(str([round(x, 1) for x in bbox]).encode()).hexdigest()[:12])
        if aid in existing_ids:
            continue
        existing_ids.add(aid)
        atoms.append({
            "id": aid,
            "type": atom_type,
            "bbox": list(bbox),
            "confidence": a.confidence,
            "source": "input_candidate_recovery",
            "recovery_source": recovery_source,
            "evidence": {
                "source": recovery_source,
                "slot_id": a.slot.slot_id,
                "slot_role": a.slot.role,
            },
        })

    n_with_bbox = sum(1 for x in graph.assignments if x.bbox)
    log_lines.append("%s: graph -> %d atoms (from %d assignments with bbox)" % (recovery_source, len(atoms), n_with_bbox))
    return atoms, log_lines


def visualize_form_graph(
    image_path: str,
    graph: FormGraph,
    output_path: str,
) -> None:
    """Визуализация итогового графа: слоты, bbox, связи label→input (уровень 5)."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    slot_by_id = {}
    for rs in graph.row_slots:
        for s in rs.slots:
            slot_by_id[s.slot_id] = s
    from src.infrastructure.debug_draw import line_visible, rectangle_visible

    for a in graph.assignments:
        if a.bbox is not None:
            x1, y1, x2, y2 = int(a.bbox[0]), int(a.bbox[1]), int(a.bbox[2]), int(a.bbox[3])
            rectangle_visible(out, (x1, y1), (x2, y2), (0, 180, 0), 2)
    for lid, iid in graph.label_to_input:
        if lid in slot_by_id and iid in slot_by_id:
            sl = slot_by_id[lid]
            si = slot_by_id[iid]
            c1 = (int((sl.x_min + sl.x_max) / 2), int((sl.y_min + sl.y_max) / 2))
            c2 = (int((si.x_min + si.x_max) / 2), int((si.y_min + si.y_max) / 2))
            line_visible(out, c1, c2, (0, 140, 200), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_graph_assembler: saved %s", output_path)
