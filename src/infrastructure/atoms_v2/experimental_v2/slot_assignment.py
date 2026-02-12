"""
S4 — Slot Assignment (State Machine Architecture)

Назначение семантических ролей элементам внутри строк:
- LABEL: текстовая метка поля
- INPUT: поле ввода
- ACTION: кнопка действия

Использует ТОЛЬКО относительные размеры:
- max_label_distance = container_width * RELATIVE_LABEL_DISTANCE
- min_input_width = container_width * RELATIVE_INPUT_WIDTH_MIN

Не модифицирует geometry!
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .structural_segmentation import FormRow, RowElement
from .visual_geometry_extractor import GeometryContext
from .ocr_extractor import LanguageInfo

logger = logging.getLogger(__name__)


# =============================================================================
# RELATIVE THRESHOLDS (no absolute px!)
# =============================================================================

class SlotThresholds:
    """All thresholds relative to container or median."""
    
    # Max distance label-to-input (relative to container_width)
    LABEL_DISTANCE_MAX = 0.3  # max 30% of container width
    
    # Min input width (relative to container_width)
    INPUT_WIDTH_MIN = 0.15  # min 15% of container width
    
    # Label width bounds (relative to container_width)
    LABEL_WIDTH_MAX = 0.4   # max 40% of container width
    
    # Button width bounds (relative to container_width)
    BUTTON_WIDTH_MIN = 0.1  # min 10% of container width
    BUTTON_WIDTH_MAX = 0.6  # max 60% of container width
    
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
    
    Returns: (slot, confidence)
    """
    T = SlotThresholds
    
    elem_type = elem.element_type
    bbox = elem.bbox
    
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    rel_width = width / ctx.container_width
    rel_height = height / ctx.median_input_height
    
    # Priority 1: Checkbox/Radio (from visual type)
    if elem_type in ("checkbox", "radio"):
        return elem_type.upper(), 0.9
    
    # Priority 2: Button/Action (from visual type or row context)
    if elem_type == "button" or row_type == "button_row":
        if T.BUTTON_WIDTH_MIN <= rel_width <= T.BUTTON_WIDTH_MAX:
            return "ACTION", 0.85
    
    # Priority 3: Textarea (tall input)
    if elem_type == "textarea" or (elem_type == "input" and rel_height >= T.TEXTAREA_HEIGHT_MIN):
        if rel_width >= T.INPUT_WIDTH_MIN:
            return "TEXTAREA", 0.8
    
    # Priority 4: Input (visual type or width)
    if elem_type == "input":
        if rel_width >= T.INPUT_WIDTH_MIN:
            return "INPUT", 0.8
    
    # Priority 5: Text-based classification
    if elem_type.startswith("text"):
        # Label if leftmost and narrow enough
        if is_leftmost and rel_width <= T.LABEL_WIDTH_MAX:
            return "LABEL", 0.7
        # Or if explicitly a label hint
        if elem.ocr_block and elem.ocr_block.is_label_hint:
            return "LABEL", 0.75
        return "LABEL", 0.5
    
    # Priority 6: Header row
    if row_type == "header":
        return "HEADER", 0.6
    
    # Fallback based on position
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
    Bind LABEL slots to nearest INPUT/TEXTAREA.
    
    Uses relative max distance threshold.
    """
    T = SlotThresholds
    max_distance = ctx.container_width * T.LABEL_DISTANCE_MAX
    
    labels = [a for a in assignments if a.slot == "LABEL"]
    inputs = [a for a in assignments if a.slot in ("INPUT", "TEXTAREA", "CHECKBOX", "RADIO")]
    
    for label in labels:
        label_bbox = label.element.bbox
        label_right = label_bbox[2]
        
        best_input = None
        best_distance = float('inf')
        
        for inp in inputs:
            inp_bbox = inp.element.bbox
            inp_left = inp_bbox[0]
            
            # Label should be to the left of input (or above)
            # Horizontal distance
            h_distance = inp_left - label_right
            
            if h_distance < 0:
                # Label is to the right — not typical binding
                continue
            
            if h_distance > max_distance:
                continue
            
            # Vertical alignment check
            label_cy = (label_bbox[1] + label_bbox[3]) / 2
            inp_cy = (inp_bbox[1] + inp_bbox[3]) / 2
            v_distance = abs(label_cy - inp_cy)
            
            # Total distance (favor horizontal proximity)
            total = h_distance + v_distance * 0.3
            
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
    """
    if not row.elements:
        return RowSlots(row=row, assignments=[])
    
    assignments = []
    
    # Sort elements by X position
    sorted_elems = sorted(row.elements, key=lambda e: e.bbox[0])
    
    for i, elem in enumerate(sorted_elems):
        is_leftmost = (i == 0)
        is_rightmost = (i == len(sorted_elems) - 1)
        
        slot, confidence = assign_slot_to_element(
            elem, ctx, row.row_type, is_leftmost, is_rightmost
        )
        
        assignments.append(SlotAssignment(
            element=elem,
            slot=slot,
            confidence=confidence,
        ))
    
    # Bind labels to inputs
    bind_labels_to_inputs(assignments, ctx)
    
    return RowSlots(row=row, assignments=assignments)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def assign_slots(
    rows: List[FormRow],
    context: GeometryContext,
    language: LanguageInfo,
) -> S4Result:
    """
    S4 — Slot Assignment.
    
    Assigns semantic roles to elements within rows.
    Does NOT modify element geometry!
    
    Args:
        rows: строки из S3
        context: контекст с размерами
        language: информация о языке
    
    Returns:
        S4Result с row_slots
    """
    diagnostics: Dict[str, Any] = {
        "total_rows": len(rows),
        "slot_counts": {},
        "bindings": 0,
    }
    
    row_slots = []
    
    for row in rows:
        rs = assign_row_slots(row, context)
        row_slots.append(rs)
        
        # Count slots
        for a in rs.assignments:
            diagnostics["slot_counts"][a.slot] = diagnostics["slot_counts"].get(a.slot, 0) + 1
            if a.bound_to:
                diagnostics["bindings"] += 1
    
    logger.info(f"S4 completed: {len(row_slots)} rows, slots={diagnostics['slot_counts']}, bindings={diagnostics['bindings']}")
    
    return S4Result(
        row_slots=row_slots,
        diagnostics=diagnostics,
    )


def get_form_atoms(s4_result: S4Result) -> List[Dict[str, Any]]:
    """
    Convert slot assignments to form atoms for compatibility.
    
    Returns list of dicts with element info.
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
