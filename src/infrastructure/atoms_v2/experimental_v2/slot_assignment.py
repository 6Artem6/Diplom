"""
S4 — Slot Assignment (State Machine Architecture)

Назначение семантических ролей элементам внутри строк:
- LABEL: текстовая метка поля
- INPUT: поле ввода
- ACTION: кнопка действия

ИНВАРИАНТЫ:
- CONTAINER не получает semantic role (пропускается!)
- Используются ТОЛЬКО относительные размеры
- max_label_distance = 1.2 * median_input_width
- Label связывается если vertical_overlap >= 30% AND horizontal_distance <= max_label_distance
- Не модифицирует geometry!
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .structural_segmentation import FormRow, RowElement
from .visual_geometry_extractor import GeometryContext, ElementTypes
from .ocr_extractor import LanguageInfo

logger = logging.getLogger(__name__)


# =============================================================================
# RELATIVE THRESHOLDS (NO ABSOLUTE PX!)
# =============================================================================

class SlotThresholds:
    """All thresholds relative to container or median. NO ABSOLUTE PX!"""
    
    # Max distance label-to-input (relative to median_input_width!)
    LABEL_DISTANCE_MAX_RATIO = 1.2  # max 1.2 * median_input_width
    
    # Min vertical overlap for label binding
    LABEL_VERTICAL_OVERLAP_MIN = 0.3  # min 30% vertical overlap
    
    # Min input width (relative to container_width)
    INPUT_WIDTH_MIN = 0.15  # min 15% of container width
    
    # Label width bounds (relative to container_width)
    LABEL_WIDTH_MAX = 0.4   # max 40% of container width
    
    # Button width bounds (relative to container_width)
    BUTTON_WIDTH_MIN = 0.08  # min 8% of container width
    BUTTON_WIDTH_MAX = 0.7   # max 70% of container width
    
    # Textarea minimum height (relative to median_input_height)
    TEXTAREA_HEIGHT_MIN = 1.5  # min 150% of median input height


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SlotAssignment:
    """Semantic role assignment for an element."""
    element: RowElement
    slot: str  # LABEL, INPUT, TEXTAREA, ACTION, CHECKBOX, RADIO, HEADER, UNKNOWN
    confidence: float
    
    # Binding info
    bound_to: Optional['SlotAssignment'] = None  # e.g., LABEL bound to INPUT


@dataclass
class RowSlots:
    """Slots for a single row."""
    row: FormRow
    assignments: List[SlotAssignment]
    
    @property
    def labels(self) -> List[SlotAssignment]:
        return [a for a in self.assignments if a.slot == "LABEL"]
    
    @property
    def inputs(self) -> List[SlotAssignment]:
        return [a for a in self.assignments if a.slot in ("INPUT", "TEXTAREA")]
    
    @property
    def actions(self) -> List[SlotAssignment]:
        return [a for a in self.assignments if a.slot == "ACTION"]


@dataclass
class S4Result:
    """Result of S4 — Slot Assignment."""
    row_slots: List[RowSlots]
    diagnostics: Dict[str, Any]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_vertical_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute vertical overlap ratio between two bboxes.
    
    Returns overlap / min(height1, height2)
    """
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0
    
    y1_min, y1_max = bbox1[1], bbox1[3]
    y2_min, y2_max = bbox2[1], bbox2[3]
    
    overlap_min = max(y1_min, y2_min)
    overlap_max = min(y1_max, y2_max)
    
    if overlap_max <= overlap_min:
        return 0.0
    
    overlap_height = overlap_max - overlap_min
    min_height = min(y1_max - y1_min, y2_max - y2_min)
    
    return overlap_height / max(1, min_height)


def compute_horizontal_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute horizontal distance between two bboxes.
    
    Returns distance from right edge of bbox1 to left edge of bbox2.
    Negative if bbox1 is to the right of bbox2.
    """
    if len(bbox1) < 4 or len(bbox2) < 4:
        return float('inf')
    
    right1 = bbox1[2]
    left2 = bbox2[0]
    
    return left2 - right1


# =============================================================================
# SLOT ASSIGNMENT FUNCTIONS
# =============================================================================

def assign_slot_to_element(
    elem: RowElement,
    ctx: GeometryContext,
    row_type: str,
    is_leftmost: bool,
    is_rightmost: bool,
) -> Tuple[str, float]:
    """
    Assign semantic slot to element.
    
    CRITICAL: S4 does NOT reclassify element types!
    Uses ONLY element_type from S1. No repeated heuristics.
    
    IMPORTANT: CONTAINER type is skipped (returns None-like tuple)
    
    Returns: (slot, confidence)
    """
    T = SlotThresholds
    
    elem_type = elem.element_type
    bbox = elem.bbox
    
    # ===========================================
    # CONTAINER — no semantic role!
    # ===========================================
    if elem_type == ElementTypes.CONTAINER or elem_type == "container":
        return "CONTAINER", 0.0  # will be skipped
    
    # ===========================================
    # DECORATION — no semantic role!
    # ===========================================
    if elem_type == ElementTypes.DECORATION or elem_type == "decoration":
        return "DECORATION", 0.0  # will be skipped
    
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    if width <= 0 or height <= 0:
        return "UNKNOWN", 0.0
    
    rel_width = width / ctx.container_width
    
    # ===========================================
    # DIRECT TYPE MAPPING — no reclassification!
    # S4 trusts S1 classification completely
    # ===========================================
    
    # Checkbox/Radio
    if elem_type in (ElementTypes.CHECKBOX, ElementTypes.RADIO, "checkbox", "radio"):
        slot = elem_type.upper() if isinstance(elem_type, str) else elem_type
        return slot, 0.9
    
    # Action/Button
    if elem_type in (ElementTypes.ACTION, "action", "button"):
        return "ACTION", 0.85
    
    # Textarea (trust S1, no height-based reclassification!)
    if elem_type in (ElementTypes.TEXTAREA, "textarea"):
        return "TEXTAREA", 0.8
    
    # Input (trust S1, no reclassification to TEXTAREA!)
    if elem_type in (ElementTypes.INPUT, "input"):
        return "INPUT", 0.8
    
    # Label
    if elem_type in (ElementTypes.LABEL, "label"):
        return "LABEL", 0.7
    
    # ===========================================
    # Text-based elements (startswith "text")
    # ===========================================
    if isinstance(elem_type, str) and elem_type.startswith("text"):
        # Label if leftmost and narrow enough
        if is_leftmost and rel_width <= T.LABEL_WIDTH_MAX:
            return "LABEL", 0.7
        # Or if explicitly a label hint from OCR
        if elem.ocr_block and elem.ocr_block.is_label_hint:
            return "LABEL", 0.75
        return "LABEL", 0.5
    
    # ===========================================
    # Header row
    # ===========================================
    if row_type == "header":
        return "HEADER", 0.6
    
    # ===========================================
    # Fallback based on position (only for UNKNOWN types)
    # ===========================================
    if is_leftmost and rel_width <= T.LABEL_WIDTH_MAX:
        return "LABEL", 0.4
    
    if rel_width >= T.INPUT_WIDTH_MIN:
        return "INPUT", 0.4
    
    return "UNKNOWN", 0.2


def bind_labels_to_inputs(
    assignments: List[SlotAssignment],
    ctx: GeometryContext,
) -> None:
    """
    Bind LABEL slots to nearest INPUT/TEXTAREA/CHECKBOX/RADIO.
    
    Supports two layouts:
    1. LEFT-OF-INPUT: label to the left of input, same row
       - vertical_overlap >= 30%
       - horizontal_distance <= 1.2 * median_input_width
    
    2. ABOVE-INPUT: label above input, different row
       - label.bottom <= input.top
       - horizontal_overlap >= 30% (left edges roughly aligned)
       - vertical_distance <= median_input_height
    
    Chooses minimum distance if multiple candidates.
    """
    T = SlotThresholds
    max_h_distance = ctx.median_input_width * T.LABEL_DISTANCE_MAX_RATIO
    max_v_distance = ctx.median_input_height * 1.2  # allow ~1.2 input heights gap
    
    labels = [a for a in assignments if a.slot == "LABEL"]
    inputs = [a for a in assignments if a.slot in ("INPUT", "TEXTAREA", "CHECKBOX", "RADIO")]
    
    for label in labels:
        label_bbox = label.element.bbox
        label_cy = (label_bbox[1] + label_bbox[3]) / 2
        label_cx = (label_bbox[0] + label_bbox[2]) / 2
        
        best_input = None
        best_distance = float('inf')
        
        for inp in inputs:
            inp_bbox = inp.element.bbox
            inp_cy = (inp_bbox[1] + inp_bbox[3]) / 2
            
            # ===========================================
            # CASE 1: LEFT-OF-INPUT (same row)
            # ===========================================
            v_overlap = compute_vertical_overlap(label_bbox, inp_bbox)
            if v_overlap >= T.LABEL_VERTICAL_OVERLAP_MIN:
                h_distance = compute_horizontal_distance(label_bbox, inp_bbox)
                
                # Label should be to the left of input
                if 0 <= h_distance <= max_h_distance:
                    v_distance = abs(label_cy - inp_cy)
                    total = h_distance + v_distance * 0.2  # prioritize horizontal
                    
                    if total < best_distance:
                        best_distance = total
                        best_input = inp
                    continue
            
            # ===========================================
            # CASE 2: ABOVE-INPUT (different row)
            # ===========================================
            label_bottom = label_bbox[3]
            inp_top = inp_bbox[1]
            
            # Label must be above input
            if label_bottom <= inp_top:
                v_gap = inp_top - label_bottom
                
                # Vertical gap must be reasonable
                if v_gap <= max_v_distance:
                    # Check horizontal alignment (left edges should be close)
                    label_left = label_bbox[0]
                    inp_left = inp_bbox[0]
                    h_offset = abs(label_left - inp_left)
                    
                    # Allow some horizontal offset (up to input width)
                    inp_width = inp_bbox[2] - inp_bbox[0]
                    if h_offset <= inp_width:
                        # Calculate total distance (vertical is primary for above-input)
                        total = v_gap + h_offset * 0.3
                        
                        if total < best_distance:
                            best_distance = total
                            best_input = inp
        
        if best_input:
            label.bound_to = best_input


def assign_row_slots(
    row: FormRow,
    ctx: GeometryContext,
) -> RowSlots:
    """
    Assign slots to all elements in a row.
    
    CONTAINER elements are skipped (no semantic role).
    """
    if not row.elements:
        return RowSlots(row=row, assignments=[])
    
    assignments = []
    
    # Sort elements by X position
    sorted_elems = sorted(row.elements, key=lambda e: e.bbox[0])
    
    # Filter out containers for position calculation
    non_container_elems = [e for e in sorted_elems 
                          if e.element_type not in (ElementTypes.CONTAINER, "container",
                                                    ElementTypes.DECORATION, "decoration")]
    
    for i, elem in enumerate(sorted_elems):
        # Check if this is leftmost/rightmost among non-containers
        is_leftmost = (elem in non_container_elems and 
                      non_container_elems.index(elem) == 0) if non_container_elems else False
        is_rightmost = (elem in non_container_elems and 
                       non_container_elems.index(elem) == len(non_container_elems) - 1) if non_container_elems else False
        
        slot, confidence = assign_slot_to_element(
            elem, ctx, row.row_type, is_leftmost, is_rightmost
        )
        
        # Skip CONTAINER and DECORATION — no semantic role
        if slot in ("CONTAINER", "DECORATION"):
            continue
        
        assignments.append(SlotAssignment(
            element=elem,
            slot=slot,
            confidence=confidence,
        ))
    
    # Bind labels to inputs
    bind_labels_to_inputs(assignments, ctx)
    
    return RowSlots(row=row, assignments=assignments)


# =============================================================================
# CONTAINER CHILDREN EXPANSION
# =============================================================================

from .visual_geometry_extractor import VisualElement

def expand_container_row(
    row: FormRow,
    all_elements: List[VisualElement],
) -> Optional[FormRow]:
    """
    If row contains ONLY container(s), replace with container's children.
    
    Returns new FormRow with children, or None if no expansion needed.
    """
    if not row.elements:
        return None
    
    # Check if ALL elements are containers
    all_containers = all(
        e.element_type in (ElementTypes.CONTAINER, "container")
        for e in row.elements
    )
    
    if not all_containers:
        return None
    
    # Build element ID to element mapping
    elem_by_id = {elem.element_id: elem for elem in all_elements}
    
    # Collect all children from all containers
    new_elements = []
    for row_elem in row.elements:
        if not row_elem.visual_element:
            continue
        
        container = row_elem.visual_element
        child_ids = getattr(container, 'child_ids', [])
        
        for child_id in child_ids:
            child_elem = elem_by_id.get(child_id)
            if child_elem:
                new_elements.append(RowElement(
                    visual_element=child_elem,
                    ocr_block=None,  # OCR binding handled separately
                ))
    
    if not new_elements:
        return None
    
    # Create new row with children
    return FormRow(
        row_index=row.row_index,
        elements=new_elements,
        y_center=row.y_center,
        row_type=row.row_type,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def assign_slots(
    rows: List[FormRow],
    context: GeometryContext,
    language: LanguageInfo,
    all_elements: Optional[List[VisualElement]] = None,
) -> S4Result:
    """
    S4 — Slot Assignment.
    
    Assigns semantic roles to elements within rows.
    
    INVARIANTS:
    - CONTAINER does not get semantic role (skipped)
    - Does NOT modify element geometry!
    - If row contains ONLY containers, replace with children
    
    Args:
        rows: строки из S3
        context: контекст с размерами
        language: информация о языке
        all_elements: все visual_elements для expansion containers
    
    Returns:
        S4Result с row_slots
    """
    diagnostics: Dict[str, Any] = {
        "total_rows": len(rows),
        "slot_counts": {},
        "bindings": 0,
        "containers_skipped": 0,
        "containers_expanded": 0,
    }
    
    row_slots = []
    all_elements = all_elements or []
    
    for row in rows:
        # Check if row needs container expansion
        expanded_row = expand_container_row(row, all_elements) if all_elements else None
        
        if expanded_row:
            # Row was container-only, use children instead
            diagnostics["containers_expanded"] += 1
            rs = assign_row_slots(expanded_row, context)
        else:
            # Normal processing
            containers_in_row = sum(1 for e in row.elements 
                                   if e.element_type in (ElementTypes.CONTAINER, "container",
                                                         ElementTypes.DECORATION, "decoration"))
            diagnostics["containers_skipped"] += containers_in_row
            rs = assign_row_slots(row, context)
        
        row_slots.append(rs)
        
        # Count slots
        for a in rs.assignments:
            diagnostics["slot_counts"][a.slot] = diagnostics["slot_counts"].get(a.slot, 0) + 1
            if a.bound_to:
                diagnostics["bindings"] += 1
    
    logger.info(f"S4 completed: {len(row_slots)} rows, slots={diagnostics['slot_counts']}, "
               f"bindings={diagnostics['bindings']}, containers_skipped={diagnostics['containers_skipped']}, "
               f"containers_expanded={diagnostics['containers_expanded']}")
    
    return S4Result(
        row_slots=row_slots,
        diagnostics=diagnostics,
    )


def get_form_atoms(s4_result: S4Result) -> List[Dict[str, Any]]:
    """
    Convert slot assignments to form atoms for compatibility.
    
    Returns list of dicts with element info.
    UNKNOWN slots are excluded.
    """
    atoms = []
    
    for rs in s4_result.row_slots:
        for a in rs.assignments:
            if a.slot == "UNKNOWN":
                continue
            
            elem = a.element
            
            atom = {
                "slot": a.slot,
                "confidence": a.confidence,
                "bbox": elem.bbox,
                "row_index": rs.row.row_index,
                "row_type": rs.row.row_type,
            }
            
            # Add text if from OCR
            if elem.ocr_block:
                atom["text"] = elem.ocr_block.text
            
            # Add visual element info
            if elem.visual_element:
                atom["element_type"] = elem.visual_element.element_type
                atom["has_border"] = elem.visual_element.has_border
                if elem.visual_element.is_checked is not None:
                    atom["is_checked"] = elem.visual_element.is_checked
            
            # Add binding info
            if a.bound_to:
                atom["bound_to_slot"] = a.bound_to.slot
                atom["bound_to_bbox"] = a.bound_to.element.bbox
            
            atoms.append(atom)
    
    return atoms
