"""
Уровень 5 — Сборка графа формы.

Связывает slot → bbox, label → input, input → helper.
Финальный источник истины — граф формы, а не список полей.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.models import (
    FormGraph,
    FormSkeleton,
    RowSlots,
    SlotAssignment,
)

logger = logging.getLogger(__name__)


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

    return FormGraph(
        skeleton=skeleton,
        row_slots=row_slots,
        assignments=assignments,
        label_to_input=label_to_input,
        input_to_helper=input_to_helper,
        metadata={"source": "experimental_multilevel_v2"},
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
    for a in graph.assignments:
        if a.bbox is not None:
            x1, y1, x2, y2 = int(a.bbox[0]), int(a.bbox[1]), int(a.bbox[2]), int(a.bbox[3])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lid, iid in graph.label_to_input:
        if lid in slot_by_id and iid in slot_by_id:
            sl = slot_by_id[lid]
            si = slot_by_id[iid]
            c1 = ((sl.x_min + sl.x_max) / 2, (sl.y_min + sl.y_max) / 2)
            c2 = ((si.x_min + si.x_max) / 2, (si.y_min + si.y_max) / 2)
            cv2.line(out, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (255, 200, 0), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_graph_assembler: saved %s", output_path)
