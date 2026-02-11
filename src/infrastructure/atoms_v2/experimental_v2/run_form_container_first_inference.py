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
    validate_container_with_ocr,
    validate_container_with_visual,
    visualize_rows,
    visualize_rows_with_types,
    visualize_skipped_rows,
    visualize_textarea_rows,
    visualize_rows_debug,
)
from src.infrastructure.atoms_v2.experimental_v2.models import FormContainer, FormGraph, FormSkeleton, RowSlots
from src.infrastructure.atoms_v2.experimental_v2.role_based_field_locator import (
    run_role_based_locator,
    visualize_slot_assignments,
)
from src.infrastructure.atoms_v2.experimental_v2.slot_layout_inference import (
    build_slot_layout,
    visualize_slot_layout,
)
from src.infrastructure.atoms_v2.experimental_v2.demo_mode_utils import (
    save_demo_artifacts,
    validate_demo_pipeline,
    visualize_demo,
)

logger = logging.getLogger(__name__)

MIN_CONTAINER_AREA_INVARIANT = 320 * 120


def _mark_no_container(debug_output_dir: str, reason: str) -> None:
    """Пишет маркер в debug-каталог: пайплайн остановлен. Скриншоты не трогаем — остаются от прошлого успешного запуска."""
    try:
        path = os.path.join(debug_output_dir, "no_container_found.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Form Container First: пайплайн остановлен.\n")
            f.write("Reason: %s\n" % reason)
            f.write("Скриншоты в папке (если есть) — от предыдущего успешного запуска.\n")
    except Exception as e:
        logger.debug("form_container_first: could not write no_container_found.txt: %s", e)


def _remove_failure_marker(debug_output_dir: str) -> None:
    """Удаляет маркер неудачи при успешном прохождении пайплайна."""
    try:
        path = os.path.join(debug_output_dir, "no_container_found.txt")
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


MIN_INPUT_WIDTH_RATIO_INVARIANT = 0.4


def _validate_form_invariants(
    container: FormContainer,
    skeleton: FormSkeleton,
    row_slots: List[RowSlots],
) -> Tuple[List[str], bool]:
    """
    Проверка инвариантов после этапов. Возвращает (список нарушений, invariant_violation).
    Не бросает исключение — только логирование и флаг.
    """
    violations: List[str] = []
    if len(container.bbox) >= 4:
        area = (container.bbox[2] - container.bbox[0]) * (container.bbox[3] - container.bbox[1])
        if area < MIN_CONTAINER_AREA_INVARIANT:
            violations.append("I1: container.area < min_area (%.0f < %.0f)" % (area, MIN_CONTAINER_AREA_INVARIANT))
    for r in skeleton.rows:
        row_h = r.y_max - r.y_min
        if row_h < 32:
            violations.append("I3: row %d height < 32px (%.1f)" % (r.row_index, row_h))
        if r.row_type not in ("TEXTAREA", "ACTION", "HEADER", "TEXT") and row_h > 120:
            violations.append("I3: row %d height > 120px non-textarea (%.1f)" % (r.row_index, row_h))
    for rs in row_slots:
        row_bbox = rs.row_bbox
        row_w = row_bbox[2] - row_bbox[0] if len(row_bbox) >= 4 else 0
        for slot in rs.slots:
            if slot.role in ("input_slot", "textarea_slot"):
                w = slot.x_max - slot.x_min
                if row_w > 0 and w < row_w * MIN_INPUT_WIDTH_RATIO_INVARIANT:
                    violations.append(
                        "I5: input_slot row %d width %.1f < 0.4*row (%.1f)" % (rs.row_index, w, row_w * 0.4)
                    )
                break
    for v in violations:
        logger.debug("invariant_violation: %s", v)
    return violations, len(violations) > 0


def run_form_container_first_inference(
    image_path: str,
    raw_ocr_boxes: Optional[List[Dict[str, Any]]] = None,
    dark_theme: bool = False,
    debug_output_dir: Optional[str] = None,
    existing_atom_ids: Optional[set] = None,
    detectron_regions: Optional[List[Dict[str, Any]]] = None,
    demo_mode: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str], Optional[FormGraph]]:
    """
    Пайплайн «сначала контейнер формы». Все уровни работают только внутри FormContainer.bbox.

    demo_mode: идеальная форма — контейнер по площади, одна строка = один input, FIELD_VERTICAL, grid выключен,
    placeholder игнорируется; сохраняются demo_*.json и demo_visualization.png; перед BPG вызывается validate_demo_pipeline.
    Возвращает (atoms, log_lines, form_graph).
    Debug (если debug_output_dir задан): container_bbox.png, rows.png, slots.png, slot_assignments.png.
    """
    log_lines: List[str] = []
    raw_ocr_boxes = raw_ocr_boxes or []
    existing_atom_ids = existing_atom_ids or set()

    # Level 0 — FormContainerDetector (в demo_mode берём самый большой bbox)
    containers, cont_diag = detect_form_containers(image_path, detectron_regions)
    if cont_diag.get("error"):
        log_lines.append("form_container_first: level0 failed %s" % cont_diag.get("error"))
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            _mark_no_container(debug_output_dir, "level0: %s" % cont_diag.get("error"))
        return [], log_lines, None

    container = get_best_container(containers, demo_mode=demo_mode)
    if not container:
        diag_str = "candidates=%s after_dedup=%s after_geometry=%s" % (
            cont_diag.get("candidates", "?"),
            cont_diag.get("after_dedup", "?"),
            cont_diag.get("after_geometry", "?"),
        )
        log_lines.append("form_container_first: no FormContainer found (%s)" % diag_str)
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            _mark_no_container(debug_output_dir, diag_str)
        return [], log_lines, None

    log_lines.append("form_container_first: 1 container confidence=%.2f" % container.confidence)

    if debug_output_dir:
        os.makedirs(debug_output_dir, exist_ok=True)
        _remove_failure_marker(debug_output_dir)
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

    if not validate_container_with_ocr(container.bbox, ocr_inside):
        log_lines.append("form_container_first: container rejected (I1: need >= 2 OCR in different Y)")
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            _mark_no_container(debug_output_dir, "I1: need >= 2 OCR in different Y")
        return [], log_lines, None

    # Визуальные кандидаты до layout: строки строятся из CV, OCR только для label/helper
    visual_candidates: List[List[float]] = []
    try:
        from src.infrastructure.atoms_v2.visual_field_scanner import run_visual_field_scan
        form_regions_for_scan = [{"bbox": container.bbox}]
        visual_candidates, _ = run_visual_field_scan(image_path, form_regions_for_scan, dark_theme=dark_theme)
    except Exception as e:
        log_lines.append("form_container_first: visual_scan failed %s" % e)

    if not demo_mode and visual_candidates and not validate_container_with_visual(container.bbox, visual_candidates):
        log_lines.append("form_container_first: container rejected by visual invariants (min rows/inputs)")
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            _mark_no_container(debug_output_dir, "visual invariants: min rows/inputs not met")
        return [], log_lines, None

    # Level 1 — FormInnerLayout (в demo_mode: одна строка = один input, FIELD_VERTICAL, без grid)
    skeleton, layout_diag = build_form_inner_layout(
        image_path, container, ocr_inside,
        visual_candidates=visual_candidates or None,
        demo_mode=demo_mode,
    )
    if not skeleton:
        log_lines.append("form_container_first: FormInnerLayout failed")
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            _mark_no_container(debug_output_dir, "FormInnerLayout failed (no skeleton)")
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
        visualize_rows_debug(
            image_path,
            container,
            layout_diag.get("rows_debug", []),
            os.path.join(debug_output_dir, "rows_debug.png"),
        )

    # Level 2 — SlotDetector
    row_slots, slot_diag = build_slot_layout(skeleton, raw_ocr_boxes)
    log_lines.append("form_container_first: total_slots=%d" % slot_diag.get("total_slots", 0))

    # OCR-ориентация (после slots): vertical / horizontal / placeholder в row.metadata["ocr_orientation"]
    if raw_ocr_boxes:
        from src.infrastructure.atoms_v2.experimental_v2.form_invariants_patch import (
            _ocr_in_row,
            analyze_ocr_orientation,
        )
        for row in skeleton.rows:
            ocr_in_row = [ob for ob in raw_ocr_boxes if _ocr_in_row(ob, row)]
            analyze_ocr_orientation(row, ocr_in_row)

    if debug_output_dir:
        visualize_slot_layout(image_path, row_slots, os.path.join(debug_output_dir, "slots.png"))

    # Level 3 — FieldLocator (visual_candidates уже получены выше для layout): bbox только внутри container, пересечение с expected_bbox_hint
    assignments, loc_diag = run_role_based_locator(
        row_slots,
        visual_candidates,
        image_path,
        dark_theme,
        container_bbox=container.bbox,
    )
    from src.infrastructure.atoms_v2.experimental_v2.form_invariants_patch import (
        correct_slot_assignment_bboxes,
        log_assignment_outside_row_invariant,
    )
    correct_slot_assignment_bboxes(assignments, skeleton, raw_ocr_boxes)
    log_assignment_outside_row_invariant(assignments, skeleton)
    log_lines.append("form_container_first: filled=%d/%d" % (loc_diag.get("filled", 0), loc_diag.get("total_input_slots", 0)))

    if debug_output_dir:
        visualize_slot_assignments(image_path, assignments, os.path.join(debug_output_dir, "slot_assignments.png"))

    # Level 4 — FormGraph
    graph = assemble_form_graph(skeleton, row_slots, assignments)
    graph.metadata["source"] = "form_container_first"

    inv_violations, inv_violation = _validate_form_invariants(container, skeleton, row_slots)
    graph.metadata["invariant_violations"] = inv_violations
    graph.metadata["invariant_violation"] = inv_violation
    if inv_violations:
        log_lines.append("form_container_first: invariant_violations=%d %s" % (len(inv_violations), inv_violations[:3]))

    if demo_mode and debug_output_dir:
        save_demo_artifacts(debug_output_dir, container, skeleton, row_slots, assignments, graph)
        visualize_demo(
            image_path, container, skeleton, row_slots,
            os.path.join(debug_output_dir, "demo_visualization.png"),
        )
        ok, demo_errors = validate_demo_pipeline(container, skeleton, row_slots, assignments, graph)
        if not ok:
            for e in demo_errors:
                log_lines.append("demo_validation: %s" % e)
                logger.warning("demo_validation: %s", e)
            log_lines.append("form_container_first: demo_validation failed, BPG build stopped")
            return [], log_lines, None

    if debug_output_dir:
        visualize_form_graph(image_path, graph, os.path.join(debug_output_dir, "form_graph.png"))

    atoms, atom_log = form_graph_to_atoms(graph, existing_atom_ids, recovery_source="form_container_first")
    log_lines.extend(atom_log)

    return atoms, log_lines, graph
