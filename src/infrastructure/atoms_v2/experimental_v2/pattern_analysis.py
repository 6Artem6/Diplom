"""
S5 — Structure Pattern Analysis (State Machine Architecture)

Детекция повторяющихся UI паттернов:
- checkbox groups (несколько checkbox в колонке)
- radio groups (взаимоисключающие radio)
- field pairs (label + input повторяются)
- button groups (несколько кнопок в ряд)

Добавляет pattern metadata, НЕ модифицирует geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .slot_assignment import RowSlots, SlotAssignment, S4Result
from .visual_geometry_extractor import GeometryContext

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class UIPattern:
    """Detected UI pattern."""
    pattern_type: str  # checkbox_group, radio_group, field_pair, button_group
    elements: List[SlotAssignment]  # elements participating in pattern
    row_indices: List[int]  # rows containing pattern
    confidence: float
    
    # Pattern-specific metadata
    options_count: int = 0  # for checkbox/radio groups
    is_vertical: bool = False  # vertical or horizontal arrangement
    is_required: bool = False  # if any element marked required


@dataclass
class S5Result:
    """Result of S5 — Structure Pattern Analysis."""
    patterns: List[UIPattern]
    element_to_pattern: Dict[int, UIPattern]  # element id -> pattern
    diagnostics: Dict[str, Any]


# =============================================================================
# PATTERN DETECTION FUNCTIONS
# =============================================================================

def detect_checkbox_groups(
    row_slots: List[RowSlots],
    ctx: GeometryContext,
) -> List[UIPattern]:
    """
    Detect checkbox group patterns.
    
    Checkbox group: 2+ checkboxes in vertical arrangement (same X, different Y)
    or horizontal arrangement (same Y, different X)
    """
    patterns = []
    
    # Collect all checkboxes
    checkboxes: List[Tuple[SlotAssignment, int]] = []  # (assignment, row_index)
    
    for rs in row_slots:
        for a in rs.assignments:
            if a.slot == "CHECKBOX":
                checkboxes.append((a, rs.row.row_index))
    
    if len(checkboxes) < 2:
        return []
    
    # Cluster by X position (vertical groups)
    x_tolerance = ctx.container_width * 0.1
    
    vertical_groups: Dict[float, List[Tuple[SlotAssignment, int]]] = {}
    
    for cb, row_idx in checkboxes:
        bbox = cb.element.bbox
        cx = (bbox[0] + bbox[2]) / 2
        
        # Find matching X cluster
        matched = False
        for group_x in vertical_groups:
            if abs(cx - group_x) <= x_tolerance:
                vertical_groups[group_x].append((cb, row_idx))
                matched = True
                break
        
        if not matched:
            vertical_groups[cx] = [(cb, row_idx)]
    
    # Create patterns from groups with 2+ elements
    for group_x, members in vertical_groups.items():
        if len(members) >= 2:
            elements = [m[0] for m in members]
            row_indices = list(set(m[1] for m in members))
            
            patterns.append(UIPattern(
                pattern_type="checkbox_group",
                elements=elements,
                row_indices=row_indices,
                confidence=0.8,
                options_count=len(elements),
                is_vertical=True,
            ))
    
    # Also check horizontal groups (same row)
    for rs in row_slots:
        row_checkboxes = [a for a in rs.assignments if a.slot == "CHECKBOX"]
        if len(row_checkboxes) >= 2:
            # Check if not already in vertical group
            already_grouped = any(
                any(cb in p.elements for cb in row_checkboxes)
                for p in patterns
            )
            if not already_grouped:
                patterns.append(UIPattern(
                    pattern_type="checkbox_group",
                    elements=row_checkboxes,
                    row_indices=[rs.row.row_index],
                    confidence=0.75,
                    options_count=len(row_checkboxes),
                    is_vertical=False,
                ))
    
    return patterns


def detect_radio_groups(
    row_slots: List[RowSlots],
    ctx: GeometryContext,
) -> List[UIPattern]:
    """
    Detect radio button group patterns.
    
    Radio group: 2+ radio buttons, usually with exclusive selection.
    """
    patterns = []
    
    # Collect all radios
    radios: List[Tuple[SlotAssignment, int]] = []
    
    for rs in row_slots:
        for a in rs.assignments:
            if a.slot == "RADIO":
                radios.append((a, rs.row.row_index))
    
    if len(radios) < 2:
        return []
    
    # Similar logic to checkbox groups
    x_tolerance = ctx.container_width * 0.1
    
    vertical_groups: Dict[float, List[Tuple[SlotAssignment, int]]] = {}
    
    for rb, row_idx in radios:
        bbox = rb.element.bbox
        cx = (bbox[0] + bbox[2]) / 2
        
        matched = False
        for group_x in vertical_groups:
            if abs(cx - group_x) <= x_tolerance:
                vertical_groups[group_x].append((rb, row_idx))
                matched = True
                break
        
        if not matched:
            vertical_groups[cx] = [(rb, row_idx)]
    
    for group_x, members in vertical_groups.items():
        if len(members) >= 2:
            elements = [m[0] for m in members]
            row_indices = list(set(m[1] for m in members))
            
            patterns.append(UIPattern(
                pattern_type="radio_group",
                elements=elements,
                row_indices=row_indices,
                confidence=0.85,
                options_count=len(elements),
                is_vertical=True,
            ))
    
    # Horizontal groups
    for rs in row_slots:
        row_radios = [a for a in rs.assignments if a.slot == "RADIO"]
        if len(row_radios) >= 2:
            already_grouped = any(
                any(rb in p.elements for rb in row_radios)
                for p in patterns
            )
            if not already_grouped:
                patterns.append(UIPattern(
                    pattern_type="radio_group",
                    elements=row_radios,
                    row_indices=[rs.row.row_index],
                    confidence=0.8,
                    options_count=len(row_radios),
                    is_vertical=False,
                ))
    
    return patterns


def detect_field_pairs(
    row_slots: List[RowSlots],
) -> List[UIPattern]:
    """
    Detect label-input pair patterns.
    
    Field pair: LABEL bound to INPUT/TEXTAREA in same row.
    """
    patterns = []
    
    for rs in row_slots:
        # Find bound pairs
        for a in rs.assignments:
            if a.slot == "LABEL" and a.bound_to:
                if a.bound_to.slot in ("INPUT", "TEXTAREA"):
                    # Check for required marker
                    is_required = False
                    if a.element.ocr_block:
                        text = a.element.ocr_block.text
                        is_required = "*" in text or "обязательно" in text.lower() or "required" in text.lower()
                    
                    patterns.append(UIPattern(
                        pattern_type="field_pair",
                        elements=[a, a.bound_to],
                        row_indices=[rs.row.row_index],
                        confidence=0.9,
                        is_required=is_required,
                    ))
    
    return patterns


def detect_button_groups(
    row_slots: List[RowSlots],
) -> List[UIPattern]:
    """
    Detect button group patterns.
    
    Button group: 2+ ACTION buttons in same row.
    """
    patterns = []
    
    for rs in row_slots:
        buttons = [a for a in rs.assignments if a.slot == "ACTION"]
        
        if len(buttons) >= 2:
            patterns.append(UIPattern(
                pattern_type="button_group",
                elements=buttons,
                row_indices=[rs.row.row_index],
                confidence=0.85,
                options_count=len(buttons),
                is_vertical=False,
            ))
    
    return patterns


def detect_repeating_structure(
    row_slots: List[RowSlots],
) -> List[UIPattern]:
    """
    Detect repeating row structures.
    
    If multiple rows have same slot sequence, they form a pattern.
    """
    patterns = []
    
    # Group rows by slot sequence
    sequence_groups: Dict[str, List[RowSlots]] = {}
    
    for rs in row_slots:
        # Create slot sequence signature
        slots = tuple(a.slot for a in sorted(rs.assignments, key=lambda x: x.element.bbox[0]))
        if len(slots) < 2:
            continue
        
        sig = "-".join(slots)
        if sig not in sequence_groups:
            sequence_groups[sig] = []
        sequence_groups[sig].append(rs)
    
    # Find repeated sequences
    for sig, rows in sequence_groups.items():
        if len(rows) >= 2:
            all_elements = []
            all_row_indices = []
            
            for rs in rows:
                all_elements.extend(rs.assignments)
                all_row_indices.append(rs.row.row_index)
            
            patterns.append(UIPattern(
                pattern_type="repeating_row",
                elements=all_elements,
                row_indices=all_row_indices,
                confidence=0.7,
                options_count=len(rows),
                is_vertical=True,
            ))
    
    return patterns


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def analyze_patterns(
    s4_result: S4Result,
    context: GeometryContext,
) -> S5Result:
    """
    S5 — Structure Pattern Analysis.
    
    Detects repeating UI patterns. Does NOT modify geometry.
    
    Args:
        s4_result: результат S4 с slot assignments
        context: контекст с размерами
    
    Returns:
        S5Result с patterns
    """
    diagnostics: Dict[str, Any] = {
        "total_patterns": 0,
        "by_type": {},
    }
    
    row_slots = s4_result.row_slots
    
    all_patterns = []
    
    # Detect different pattern types
    checkbox_groups = detect_checkbox_groups(row_slots, context)
    all_patterns.extend(checkbox_groups)
    
    radio_groups = detect_radio_groups(row_slots, context)
    all_patterns.extend(radio_groups)
    
    field_pairs = detect_field_pairs(row_slots)
    all_patterns.extend(field_pairs)
    
    button_groups = detect_button_groups(row_slots)
    all_patterns.extend(button_groups)
    
    repeating = detect_repeating_structure(row_slots)
    all_patterns.extend(repeating)
    
    # Build element -> pattern mapping
    element_to_pattern: Dict[int, UIPattern] = {}
    for pattern in all_patterns:
        for elem in pattern.elements:
            element_to_pattern[id(elem)] = pattern
    
    # Update diagnostics
    diagnostics["total_patterns"] = len(all_patterns)
    for p in all_patterns:
        diagnostics["by_type"][p.pattern_type] = diagnostics["by_type"].get(p.pattern_type, 0) + 1
    
    logger.info(f"S5 completed: {len(all_patterns)} patterns, types={diagnostics['by_type']}")
    
    return S5Result(
        patterns=all_patterns,
        element_to_pattern=element_to_pattern,
        diagnostics=diagnostics,
    )
