"""
Точка входа experimental_multilevel_v2.

Последовательность: Level 0 → 1 → 2 → 3 → 4 → 5.
Сохраняет диагностические визуализации по уровням, если задан debug_output_dir.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.form_graph_assembler import (
    assemble_form_graph,
    form_graph_to_atoms,
    visualize_form_graph,
)
from src.infrastructure.atoms_v2.experimental_v2.form_skeleton_builder import (
    build_form_skeleton,
    visualize_form_skeleton,
)
from src.infrastructure.atoms_v2.experimental_v2.models import FormGraph, Region
from src.infrastructure.atoms_v2.experimental_v2.page_orientation_context import (
    build_page_orientation_context,
    visualize_page_orientation,
)
from src.infrastructure.atoms_v2.experimental_v2.role_based_field_locator import (
    run_role_based_locator,
    visualize_slot_assignments,
)
from src.infrastructure.atoms_v2.experimental_v2.semantic_region_builder import (
    build_semantic_regions,
    visualize_semantic_regions,
)
from src.infrastructure.atoms_v2.experimental_v2.slot_layout_inference import (
    build_slot_layout,
    visualize_slot_layout,
)

logger = logging.getLogger(__name__)


def run_experimental_multilevel_v2(
    image_path: str,
    raw_ocr_boxes: Optional[List[Dict[str, Any]]] = None,
    dark_theme: bool = False,
    debug_output_dir: Optional[str] = None,
    existing_atom_ids: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], List[str], Optional[FormGraph]]:
    """
    Запускает экспериментальный пайплайн v2: от вершины к листьям.

    Возвращает (atoms, log_lines, form_graph).
    Если debug_output_dir задан, сохраняет визуализации по уровням:
      - level0_page_orientation.png
      - level1_semantic_regions.png
      - level2_form_skeleton.png
      - level3_slots.png
      - level4_slot_bbox.png
      - level5_form_graph.png
    """
    log_lines: List[str] = []
    raw_ocr_boxes = raw_ocr_boxes or []
    existing_atom_ids = existing_atom_ids or set()

    # Level 0 — глобальная ориентация
    segments, page_diag = build_page_orientation_context(image_path)
    if page_diag.get("error"):
        log_lines.append("experimental_v2: level0 failed %s" % page_diag.get("error"))
        return [], log_lines, None
    log_lines.append("experimental_v2: level0 segments=%d" % len(segments))

    if debug_output_dir:
        os.makedirs(debug_output_dir, exist_ok=True)
        visualize_page_orientation(image_path, segments, os.path.join(debug_output_dir, "level0_page_orientation.png"))

    # Level 1 — семантические регионы
    regions, sem_diag = build_semantic_regions(image_path, segments, page_diag)
    form_areas = [r for r in regions if r.region_type == "form_area"]
    if not form_areas:
        log_lines.append("experimental_v2: level1 no form_area")
        if debug_output_dir:
            visualize_semantic_regions(image_path, regions, os.path.join(debug_output_dir, "level1_semantic_regions.png"))
        return [], log_lines, None

    if debug_output_dir:
        visualize_semantic_regions(image_path, regions, os.path.join(debug_output_dir, "level1_semantic_regions.png"))

    form_region = form_areas[0]

    # Level 2 — скелет формы внутри form_area
    skeleton, skel_diag = build_form_skeleton(image_path, form_region, segments, raw_ocr_boxes)
    if skeleton is None:
        log_lines.append("experimental_v2: level2 skeleton failed")
        return [], log_lines, None
    log_lines.append("experimental_v2: level2 rows=%d layout=%s" % (skel_diag.get("n_rows", 0), skel_diag.get("layout_type", "")))

    if debug_output_dir:
        visualize_form_skeleton(image_path, skeleton, os.path.join(debug_output_dir, "level2_form_skeleton.png"))

    # Level 3 — слоты по строкам
    row_slots, slot_diag = build_slot_layout(skeleton, raw_ocr_boxes)
    log_lines.append("experimental_v2: level3 total_slots=%d" % slot_diag.get("total_slots", 0))

    if debug_output_dir:
        visualize_slot_layout(image_path, row_slots, os.path.join(debug_output_dir, "level3_slots.png"))

    # Визуальные кандидаты только внутри form_area (без Detectron2)
    try:
        from src.infrastructure.atoms_v2.visual_field_scanner import run_visual_field_scan
        form_regions_for_scan = [{"bbox": form_region.bbox}]
        visual_candidates, _ = run_visual_field_scan(image_path, form_regions_for_scan, dark_theme=dark_theme)
    except Exception as e:
        log_lines.append("experimental_v2: visual_scan failed %s" % e)
        visual_candidates = []

    # Level 4 — поиск по роли слота
    assignments, loc_diag = run_role_based_locator(row_slots, visual_candidates, image_path, dark_theme)
    log_lines.append("experimental_v2: level4 filled=%d/%d" % (loc_diag.get("filled", 0), loc_diag.get("total_input_slots", 0)))

    if debug_output_dir:
        visualize_slot_assignments(image_path, assignments, os.path.join(debug_output_dir, "level4_slot_bbox.png"))

    # Level 5 — граф формы
    graph = assemble_form_graph(skeleton, row_slots, assignments)
    if debug_output_dir:
        visualize_form_graph(image_path, graph, os.path.join(debug_output_dir, "level5_form_graph.png"))

    atoms, atom_log = form_graph_to_atoms(graph, existing_atom_ids)
    log_lines.extend(atom_log)

    return atoms, log_lines, graph
