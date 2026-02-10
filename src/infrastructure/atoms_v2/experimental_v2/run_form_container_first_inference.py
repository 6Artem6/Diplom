"""
Точка входа Form Container First (ТЗ).

Инвариант №0: ни строка, ни слот, ни поле не существуют вне FormContainer.bbox.
Уровни: FormContainerDetector → FormInnerLayout (RowDetector, ColumnDetector) → SlotDetector
→ FieldLocator (внутри контейнера) → FormGraph.
Debug: container_bbox.png, rows.png, slots.png, slot_assignments.png.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.form_container_detector import (
    detect_form_containers,
    get_best_container,
    visualize_container,
)
from src.infrastructure.atoms_v2.experimental_v2.form_graph_assembler import (
    assemble_form_graph,
    form_graph_to_atoms,
    visualize_form_graph,
)
from src.infrastructure.atoms_v2.experimental_v2.form_inner_layout import (
    build_form_inner_layout,
    visualize_rows,
    visualize_rows_with_types,
    visualize_skipped_rows,
    visualize_textarea_rows,
)
from src.infrastructure.atoms_v2.experimental_v2.models import FormContainer, FormGraph
from src.infrastructure.atoms_v2.experimental_v2.role_based_field_locator import (
    run_role_based_locator,
    visualize_slot_assignments,
)
from src.infrastructure.atoms_v2.experimental_v2.slot_layout_inference import (
    build_slot_layout,
    visualize_slot_layout,
)

logger = logging.getLogger(__name__)


def run_form_container_first_inference(
    image_path: str,
    raw_ocr_boxes: Optional[List[Dict[str, Any]]] = None,
    dark_theme: bool = False,
    debug_output_dir: Optional[str] = None,
    existing_atom_ids: Optional[set] = None,
    detectron_regions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str], Optional[FormGraph]]:
    """
    Пайплайн «сначала контейнер формы». Все уровни работают только внутри FormContainer.bbox.

    Возвращает (atoms, log_lines, form_graph).
    Debug (если debug_output_dir задан): container_bbox.png, rows.png, slots.png, slot_assignments.png.
    """
    log_lines: List[str] = []
    raw_ocr_boxes = raw_ocr_boxes or []
    existing_atom_ids = existing_atom_ids or set()

    # Level 0 — FormContainerDetector (геометрический контейнер, не «остаток»)
    containers, cont_diag = detect_form_containers(image_path, detectron_regions)
    if cont_diag.get("error"):
        log_lines.append("form_container_first: level0 failed %s" % cont_diag.get("error"))
        return [], log_lines, None

    container = get_best_container(containers)
    if not container:
        log_lines.append("form_container_first: no FormContainer found")
        return [], log_lines, None

    log_lines.append("form_container_first: 1 container confidence=%.2f" % container.confidence)

    if debug_output_dir:
        os.makedirs(debug_output_dir, exist_ok=True)
        visualize_container(image_path, container, os.path.join(debug_output_dir, "container_bbox.png"))

    # OCR внутри контейнера
    def _ocr_inside(box: List[float]) -> List[Dict[str, Any]]:
        if len(box) < 4:
            return []
        return [
            ob for ob in raw_ocr_boxes
            if len((ob.get("bbox") or [])) >= 4
            and box[0] <= (ob["bbox"][0] + ob["bbox"][2]) / 2 <= box[2]
            and box[1] <= (ob["bbox"][1] + ob["bbox"][3]) / 2 <= box[3]
        ]

    ocr_inside = _ocr_inside(container.bbox)

    # Level 1 — FormInnerLayout строго внутри container.bbox
    skeleton, layout_diag = build_form_inner_layout(image_path, container, ocr_inside)
    if not skeleton:
        log_lines.append("form_container_first: FormInnerLayout failed")
        return [], log_lines, None

    log_lines.append("form_container_first: rows=%d layout=%s" % (layout_diag.get("n_rows", 0), layout_diag.get("layout_type", "")))

    if debug_output_dir:
        visualize_rows(image_path, container, skeleton.rows, os.path.join(debug_output_dir, "rows.png"))
        visualize_rows_with_types(
            image_path, skeleton.rows, os.path.join(debug_output_dir, "rows_with_types.png")
        )
        visualize_skipped_rows(
            image_path,
            layout_diag.get("skipped_bboxes", []),
            os.path.join(debug_output_dir, "skipped_rows.png"),
        )
        visualize_textarea_rows(
            image_path,
            skeleton.rows,
            layout_diag.get("textarea_row_indices", []),
            os.path.join(debug_output_dir, "textarea_rows.png"),
        )

    # Level 2 — SlotDetector
    row_slots, slot_diag = build_slot_layout(skeleton, raw_ocr_boxes)
    log_lines.append("form_container_first: total_slots=%d" % slot_diag.get("total_slots", 0))

    if debug_output_dir:
        visualize_slot_layout(image_path, row_slots, os.path.join(debug_output_dir, "slots.png"))

    # Визуальные кандидаты только внутри контейнера
    try:
        from src.infrastructure.atoms_v2.visual_field_scanner import run_visual_field_scan
        form_regions_for_scan = [{"bbox": container.bbox}]
        visual_candidates, _ = run_visual_field_scan(image_path, form_regions_for_scan, dark_theme=dark_theme)
    except Exception as e:
        log_lines.append("form_container_first: visual_scan failed %s" % e)
        visual_candidates = []

    # Level 3 — FieldLocator: bbox только внутри container, пересечение с expected_bbox_hint
    assignments, loc_diag = run_role_based_locator(
        row_slots,
        visual_candidates,
        image_path,
        dark_theme,
        container_bbox=container.bbox,
    )
    log_lines.append("form_container_first: filled=%d/%d" % (loc_diag.get("filled", 0), loc_diag.get("total_input_slots", 0)))

    if debug_output_dir:
        visualize_slot_assignments(image_path, assignments, os.path.join(debug_output_dir, "slot_assignments.png"))

    # Level 4 — FormGraph
    graph = assemble_form_graph(skeleton, row_slots, assignments)
    graph.metadata["source"] = "form_container_first"

    if debug_output_dir:
        visualize_form_graph(image_path, graph, os.path.join(debug_output_dir, "form_graph.png"))

    atoms, atom_log = form_graph_to_atoms(graph, existing_atom_ids, recovery_source="form_container_first")
    log_lines.extend(atom_log)

    return atoms, log_lines, graph
