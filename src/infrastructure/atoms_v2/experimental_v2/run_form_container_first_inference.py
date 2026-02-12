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
    visualize_leaf_detection,
    visualize_container_leaf_detection,
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


def _diagnose_zero_atoms(
    debug_output_dir: str,
    container: FormContainer,
    skeleton: FormSkeleton,
    row_slots: List[RowSlots],
    assignments: List[Any],
    visual_elements: Optional[List[Dict[str, Any]]],
    visual_candidates: List[List[float]],
    demo_errors: List[str],
    log_lines: List[str],
) -> None:
    """
    Диагностика причин atoms=0.
    Выводит детальный отчёт с причинами и статистикой.
    """
    import os
    import json
    
    diag_lines = ["=== ATOMS=0 DIAGNOSTICS ===\n"]
    
    # 1. Контейнер
    if len(container.bbox) >= 4:
        container_w = container.bbox[2] - container.bbox[0]
        container_h = container.bbox[3] - container.bbox[1]
        diag_lines.append(f"Container: {container_w:.0f}x{container_h:.0f}, area={container_w*container_h:.0f}")
    
    # 2. Строки и слоты
    diag_lines.append(f"\nRows: {len(skeleton.rows)}")
    for row in skeleton.rows:
        rw = row.x_max - row.x_min
        rh = row.y_max - row.y_min
        has_input = "YES" if row.input_bbox else "NO"
        diag_lines.append(f"  Row {row.row_index}: type={row.row_type}, size={rw:.0f}x{rh:.0f}, input={has_input}")
    
    diag_lines.append(f"\nSlots: {len(row_slots)}")
    for rs in row_slots:
        diag_lines.append(f"  Row {rs.row_index}: {len(rs.slots)} slots")
        for slot in rs.slots:
            diag_lines.append(f"    Slot {slot.slot_id}: role={slot.role}")
    
    # 3. Assignments
    diag_lines.append(f"\nAssignments: {len(assignments)}")
    filled = 0
    for a in assignments:
        if hasattr(a, 'bbox') and a.bbox:
            filled += 1
            diag_lines.append(f"  Slot {a.slot_id}: bbox present, width={a.bbox[2]-a.bbox[0]:.0f}")
        else:
            diag_lines.append(f"  Slot {a.slot_id}: NO bbox")
    diag_lines.append(f"Filled slots: {filled}/{len(assignments)}")
    
    # 4. Visual elements
    if visual_elements:
        diag_lines.append(f"\nVisual elements: {len(visual_elements)}")
        type_counts = {}
        for e in visual_elements:
            etype = e.get("element_type", "unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1
        diag_lines.append(f"By type: {type_counts}")
        
        containers = [e for e in visual_elements if e.get("is_container")]
        diag_lines.append(f"Containers (excluded): {len(containers)}")
    
    # 5. Visual candidates
    diag_lines.append(f"\nVisual candidates: {len(visual_candidates)}")
    if visual_candidates:
        for i, vc in enumerate(visual_candidates[:10]):
            w = vc[2] - vc[0]
            h = vc[3] - vc[1]
            container_w = container.bbox[2] - container.bbox[0] if len(container.bbox) >= 4 else 1000
            width_pct = w / container_w * 100
            diag_lines.append(f"  [{i}]: {w:.0f}x{h:.0f} ({width_pct:.1f}% width)")
    
    # 6. Demo errors
    diag_lines.append(f"\nDemo validation errors: {len(demo_errors)}")
    for e in demo_errors:
        diag_lines.append(f"  - {e}")
    
    # 7. Анализ причин
    diag_lines.append("\n=== LIKELY CAUSES ===")
    
    # Проверка ширины элементов
    narrow_elements = 0
    for a in assignments:
        if hasattr(a, 'bbox') and a.bbox:
            w = a.bbox[2] - a.bbox[0]
            container_w = container.bbox[2] - container.bbox[0] if len(container.bbox) >= 4 else 1000
            if w / container_w < MIN_INPUT_WIDTH_RATIO_INVARIANT:
                narrow_elements += 1
    if narrow_elements > 0:
        diag_lines.append(f"- {narrow_elements} elements narrower than {MIN_INPUT_WIDTH_RATIO_INVARIANT*100:.0f}% container width")
    
    # Проверка перекрытия slots
    overlap_errors = [e for e in demo_errors if "overlap" in e.lower()]
    if overlap_errors:
        diag_lines.append(f"- {len(overlap_errors)} slot overlap errors")
    
    # Проверка несоответствия типов
    type_errors = [e for e in demo_errors if "FIELD row count" in e or "type" in e.lower()]
    if type_errors:
        diag_lines.append(f"- {len(type_errors)} type mismatch errors")
    
    # Сохраняем диагностику
    if debug_output_dir:
        os.makedirs(debug_output_dir, exist_ok=True)
        diag_path = os.path.join(debug_output_dir, "atoms_zero_diagnostics.txt")
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write("\n".join(diag_lines))
        
        # Также сохраняем как JSON для программного анализа
        diag_json = {
            "container_bbox": list(container.bbox) if container.bbox else [],
            "rows_count": len(skeleton.rows),
            "slots_count": len(row_slots),
            "assignments_count": len(assignments),
            "filled_assignments": filled,
            "visual_elements_count": len(visual_elements) if visual_elements else 0,
            "visual_candidates_count": len(visual_candidates),
            "demo_errors": demo_errors,
        }
        with open(os.path.join(debug_output_dir, "atoms_zero_diagnostics.json"), "w") as f:
            json.dump(diag_json, f, indent=2)
    
    # Логируем ключевые причины
    log_lines.append("atoms=0 diagnosis: %d rows, %d slots, %d filled, %d visual_elements, %d candidates" % (
        len(skeleton.rows), len(row_slots), filled, 
        len(visual_elements) if visual_elements else 0, len(visual_candidates)
    ))


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

    # В demo_mode без OCR пропускаем валидацию (для тестирования на скриншотах без OCR-данных)
    if not demo_mode and not validate_container_with_ocr(container.bbox, ocr_inside):
        log_lines.append("form_container_first: container rejected (I1: need >= 2 OCR in different Y)")
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            _mark_no_container(debug_output_dir, "I1: need >= 2 OCR in different Y")
        return [], log_lines, None
    elif demo_mode and not ocr_inside:
        log_lines.append("form_container_first: demo_mode without OCR, skipping OCR validation")

    # Визуальные кандидаты до layout: строки строятся из CV, OCR только для label/helper
    visual_candidates: List[List[float]] = []
    try:
        from src.infrastructure.atoms_v2.visual_field_scanner import run_visual_field_scan
        form_regions_for_scan = [{"bbox": container.bbox}]
        visual_candidates, _ = run_visual_field_scan(image_path, form_regions_for_scan, dark_theme=dark_theme)
    except Exception as e:
        log_lines.append("form_container_first: visual_scan failed %s" % e)

    # Fallback: используем расширенный детектор всех элементов с пост-обработкой
    visual_elements: Optional[List[Dict[str, Any]]] = None
    try:
        from src.infrastructure.atoms_v2.visual_field_scanner import (
            scan_all_visual_elements,
            postprocess_visual_elements,
            detect_checkbox_radio_priority,
        )
        
        # 1. Приоритетная детекция checkbox/radio
        checkbox_radio = detect_checkbox_radio_priority(image_path, container.bbox, dark_theme=dark_theme)
        
        # 2. Основная детекция всех элементов
        all_elements, elem_stats = scan_all_visual_elements(image_path, container.bbox, dark_theme=dark_theme)
        
        # 3. Объединяем checkbox/radio с остальными (checkbox/radio имеют приоритет)
        combined = checkbox_radio + [e for e in all_elements if e.get("element_type") not in ("checkbox", "radio")]
        
        # 4. Пост-обработка: разделение контейнеров и leaf, NMS
        processed_elements, postprocess_logs = postprocess_visual_elements(
            combined, container.bbox, median_text_height=20.0
        )
        log_lines.extend(postprocess_logs)
        
        visual_elements = processed_elements
        
        # Извлекаем bbox из обработанных элементов
        # ВАЖНО: для валидации контейнера включаем ВСЕ элементы (включая is_container)
        # Исключение контейнеров делается только для построения строк
        extra_candidates = [e["bbox"] for e in processed_elements if e.get("bbox")]
        
        # Также сохраняем leaf-only кандидаты для layout
        leaf_candidates = [
            e["bbox"] for e in processed_elements 
            if e.get("bbox") and not e.get("is_container")
        ]
        
        log_lines.append(
            "form_container_first: scan_all found %d, checkbox_radio=%d, processed=%d, candidates=%d (leaf=%d)"
            % (len(all_elements), len(checkbox_radio), len(processed_elements), len(extra_candidates), len(leaf_candidates))
        )
        
        # Статистика по типам
        type_counts = {}
        for e in processed_elements:
            etype = e.get("element_type", "unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1
        log_lines.append(f"form_container_first: element_types={type_counts}")
        
        # Объединяем с visual_candidates (убираем дубликаты по IoU)
        for ec in extra_candidates:
            is_dup = False
            for vc in visual_candidates:
                ix1 = max(ec[0], vc[0])
                iy1 = max(ec[1], vc[1])
                ix2 = min(ec[2], vc[2])
                iy2 = min(ec[3], vc[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area_a = (ec[2] - ec[0]) * (ec[3] - ec[1])
                    area_b = (vc[2] - vc[0]) * (vc[3] - vc[1])
                    iou = inter / max(1, area_a + area_b - inter)
                    if iou > 0.4:  # Более строгий порог
                        is_dup = True
                        break
            if not is_dup:
                visual_candidates.append(ec)
        log_lines.append("form_container_first: merged visual_candidates=%d" % len(visual_candidates))
        
    except Exception as e:
        log_lines.append("form_container_first: scan_all_visual_elements failed %s" % e)

    if not demo_mode and visual_candidates and not validate_container_with_visual(container.bbox, visual_candidates):
        log_lines.append("form_container_first: container rejected by visual invariants (min rows/inputs)")
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            _mark_no_container(debug_output_dir, "visual invariants: min rows/inputs not met")
        return [], log_lines, None

    # Stage 1.1 — ContainerLeafDetection (диагностика ДО row segmentation)
    from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import (
        run_container_leaf_detection,
        update_container_leaf_with_rows,
    )
    container_leaf_result = run_container_leaf_detection(
        container.bbox, image_path, ocr_inside,
        rows=None,  # rows ещё не построены
        median_input_height=None,
    )

    # Level 1 — FormInnerLayout (в demo_mode: одна строка = один input, FIELD_VERTICAL, без grid)
    skeleton, layout_diag = build_form_inner_layout(
        image_path, container, ocr_inside,
        visual_candidates=visual_candidates or None,
        visual_elements=visual_elements,
        demo_mode=demo_mode,
    )
    if not skeleton:
        log_lines.append("form_container_first: FormInnerLayout failed")
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            _mark_no_container(debug_output_dir, "FormInnerLayout failed (no skeleton)")
        return [], log_lines, None

    log_lines.append("form_container_first: rows=%d layout=%s" % (layout_diag.get("n_rows", 0), layout_diag.get("layout_type", "")))

    # Stage 1.1 — обновить ContainerLeafDetection после построения rows
    container_leaf_result = update_container_leaf_with_rows(container_leaf_result, skeleton.rows)
    container.metadata["container_leaf"] = container_leaf_result
    if container_leaf_result.get("outside_rows"):
        log_lines.append(
            "form_container_first: container_leaf outside_rows=%d" % len(container_leaf_result["outside_rows"])
        )

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
        # LeafElementDetection visualization (Stage 1 diagnostic)
        visualize_leaf_detection(
            image_path,
            skeleton,
            os.path.join(debug_output_dir, "leaf_detection.png"),
        )
        # ContainerLeafDetection visualization (Stage 1.1 diagnostic)
        visualize_container_leaf_detection(
            image_path,
            container.bbox,
            container_leaf_result,
            skeleton.rows,
            os.path.join(debug_output_dir, "container_leaf_detection.png"),
        )
        
        # Усиленная визуализация всех элементов
        if visual_elements:
            try:
                from src.infrastructure.atoms_v2.experimental_v2.enhanced_visualization import (
                    visualize_elements_enhanced,
                    visualize_nesting,
                    visualize_overlaps,
                )
                from src.infrastructure.atoms_v2.experimental_v2.detection_diagnostics import (
                    diagnose_visual_candidates,
                    print_diagnostics,
                )
                
                # Добавляем row_type из skeleton к элементам для визуализации
                elements_for_vis = []
                for e in visual_elements:
                    e_copy = e.copy()
                    # Находим соответствующую строку по bbox
                    eb = e.get("bbox", [])
                    if len(eb) >= 4:
                        for row in skeleton.rows:
                            if row.input_bbox and len(row.input_bbox) >= 4:
                                ib = row.input_bbox
                                # Проверяем совпадение
                                if (abs(eb[0] - ib[0]) < 10 and abs(eb[1] - ib[1]) < 10 and
                                    abs(eb[2] - ib[2]) < 10 and abs(eb[3] - ib[3]) < 10):
                                    e_copy["row_type"] = row.row_type
                                    break
                    elements_for_vis.append(e_copy)
                
                visualize_elements_enhanced(
                    image_path,
                    elements_for_vis,
                    os.path.join(debug_output_dir, "elements_enhanced.png"),
                    title=f"Elements: {len(elements_for_vis)}",
                )
                
                # Диагностика
                diag = diagnose_visual_candidates(visual_elements, container.bbox)
                diag_text = print_diagnostics(diag, os.path.basename(image_path))
                with open(os.path.join(debug_output_dir, "detection_diagnostics.txt"), "w") as f:
                    f.write(diag_text)
                
                # Визуализация вложенности
                if diag.nested_elements:
                    visualize_nesting(
                        image_path,
                        visual_elements,
                        diag.nested_elements,
                        os.path.join(debug_output_dir, "elements_nesting.png"),
                    )
                
                # Визуализация пересечений
                if diag.overlapping_pairs:
                    visualize_overlaps(
                        image_path,
                        visual_elements,
                        diag.overlapping_pairs,
                        os.path.join(debug_output_dir, "elements_overlaps.png"),
                    )
                    
            except Exception as e:
                log_lines.append(f"form_container_first: enhanced visualization failed: {e}")

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
        # LeafElementDetection visualization — отдельное изображение
        visualize_leaf_detection(
            image_path, skeleton,
            os.path.join(debug_output_dir, "leaf_detection.png"),
        )
        # ContainerLeafDetection visualization (Stage 1.1)
        visualize_container_leaf_detection(
            image_path,
            container.bbox,
            container_leaf_result,
            skeleton.rows,
            os.path.join(debug_output_dir, "container_leaf_detection.png"),
        )
        ok, demo_errors = validate_demo_pipeline(container, skeleton, row_slots, assignments, graph)
        if not ok:
            for e in demo_errors:
                log_lines.append("demo_validation: %s" % e)
                logger.warning("demo_validation: %s", e)
            
            # Диагностика atoms=0
            _diagnose_zero_atoms(
                debug_output_dir, container, skeleton, row_slots, assignments, 
                visual_elements, visual_candidates, demo_errors, log_lines
            )
            
            log_lines.append("form_container_first: demo_validation failed, BPG build stopped")
            return [], log_lines, None

    if debug_output_dir:
        visualize_form_graph(image_path, graph, os.path.join(debug_output_dir, "form_graph.png"))

    atoms, atom_log = form_graph_to_atoms(graph, existing_atom_ids, recovery_source="form_container_first")
    log_lines.extend(atom_log)

    return atoms, log_lines, graph
