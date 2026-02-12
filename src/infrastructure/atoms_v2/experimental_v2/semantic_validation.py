"""
S6 — Semantic Validation (State Machine Architecture)

Логическая валидация формы без изменения geometry.

Добавляет ТОЛЬКО flags и warnings:
- inconsistency flags (label без input, checkbox без label)
- language context warnings
- duplicate detection warnings

НЕ модифицирует geometry, slots или patterns!
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .slot_assignment import RowSlots, SlotAssignment, S4Result
from .pattern_analysis import UIPattern, S5Result
from .ocr_extractor import LanguageInfo

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ValidationFlag:
    """Validation issue or warning."""
    flag_type: str  # error, warning, info
    code: str       # unique code for flag type
    message: str
    
    # Context
    element_bbox: Optional[List[float]] = None
    row_index: Optional[int] = None
    related_elements: List[Any] = field(default_factory=list)


@dataclass
class S6Result:
    """Result of S6 — Semantic Validation."""
    flags: List[ValidationFlag]
    is_valid: bool  # True if no errors
    confidence_score: float  # overall form validity score
    diagnostics: Dict[str, Any]


# =============================================================================
# VALIDATION RULES
# =============================================================================

def validate_orphan_labels(
    row_slots: List[RowSlots],
) -> List[ValidationFlag]:
    """
    Check for labels without bound inputs.
    
    Warning: label has no bound input nearby.
    """
    flags = []
    
    for rs in row_slots:
        for a in rs.assignments:
            if a.slot == "LABEL" and a.bound_to is None:
                # Check if there's any input in same row
                has_input_in_row = any(
                    other.slot in ("INPUT", "TEXTAREA", "CHECKBOX", "RADIO")
                    for other in rs.assignments
                    if other is not a
                )
                
                if not has_input_in_row:
                    flags.append(ValidationFlag(
                        flag_type="warning",
                        code="ORPHAN_LABEL",
                        message="Label without bound input",
                        element_bbox=a.element.bbox,
                        row_index=rs.row.row_index,
                    ))
    
    return flags


def validate_orphan_inputs(
    row_slots: List[RowSlots],
) -> List[ValidationFlag]:
    """
    Check for inputs without labels.
    
    Warning: input has no label nearby.
    """
    flags = []
    
    for rs in row_slots:
        inputs = [a for a in rs.assignments if a.slot in ("INPUT", "TEXTAREA")]
        labels = [a for a in rs.assignments if a.slot == "LABEL"]
        
        # Check if any input is not bound to any label
        for inp in inputs:
            is_bound = any(lab.bound_to is inp for lab in labels)
            
            if not is_bound:
                flags.append(ValidationFlag(
                    flag_type="info",
                    code="ORPHAN_INPUT",
                    message="Input without label",
                    element_bbox=inp.element.bbox,
                    row_index=rs.row.row_index,
                ))
    
    return flags


def validate_checkbox_radio_labels(
    row_slots: List[RowSlots],
    patterns: List[UIPattern],
) -> List[ValidationFlag]:
    """
    Check checkbox/radio groups have labels.
    
    Warning: checkbox/radio group without descriptive label.
    """
    flags = []
    
    # Find checkbox/radio groups
    for pattern in patterns:
        if pattern.pattern_type not in ("checkbox_group", "radio_group"):
            continue
        
        # Check if group has any associated label
        row_indices = set(pattern.row_indices)
        
        has_label = False
        for rs in row_slots:
            if rs.row.row_index not in row_indices:
                continue
            
            for a in rs.assignments:
                if a.slot == "LABEL":
                    has_label = True
                    break
            if has_label:
                break
        
        if not has_label:
            flags.append(ValidationFlag(
                flag_type="info",
                code="GROUP_NO_LABEL",
                message=f"{pattern.pattern_type} without label",
                row_index=pattern.row_indices[0] if pattern.row_indices else None,
            ))
    
    return flags


def validate_button_texts(
    row_slots: List[RowSlots],
    language: LanguageInfo,
) -> List[ValidationFlag]:
    """
    Check button texts match expected patterns for language.
    
    Warning: button text doesn't match common action words.
    """
    flags = []
    
    # Common action words by language
    action_words_ru = {"отправить", "сохранить", "отмена", "далее", "назад", "добавить", 
                       "удалить", "создать", "войти", "выйти", "закрыть", "применить", "ок", "да", "нет"}
    action_words_en = {"submit", "save", "cancel", "next", "back", "add", 
                       "delete", "create", "login", "logout", "close", "apply", "ok", "yes", "no"}
    
    action_words = action_words_ru if language.primary == "ru" else action_words_en
    
    for rs in row_slots:
        for a in rs.assignments:
            if a.slot != "ACTION":
                continue
            
            # Get button text
            text = ""
            if a.element.ocr_block:
                text = a.element.ocr_block.text.lower().strip()
            
            if not text:
                continue
            
            # Check if matches common action word
            matches = any(word in text for word in action_words)
            
            if not matches and len(text) > 1:
                flags.append(ValidationFlag(
                    flag_type="info",
                    code="UNUSUAL_BUTTON_TEXT",
                    message=f"Button text '{text}' may not be a standard action",
                    element_bbox=a.element.bbox,
                    row_index=rs.row.row_index,
                ))
    
    return flags


def validate_language_consistency(
    row_slots: List[RowSlots],
    language: LanguageInfo,
) -> List[ValidationFlag]:
    """
    Check for language mixing in labels/buttons.
    
    Warning: form uses mixed languages.
    """
    flags = []
    
    if language.primary == "mixed":
        flags.append(ValidationFlag(
            flag_type="info",
            code="MIXED_LANGUAGE",
            message=f"Form uses mixed languages (ru={language.ru_ratio:.0%}, en={language.en_ratio:.0%})",
        ))
    
    return flags


def validate_duplicate_labels(
    row_slots: List[RowSlots],
) -> List[ValidationFlag]:
    """
    Check for duplicate label texts.
    
    Warning: same label appears multiple times.
    """
    flags = []
    
    label_texts: Dict[str, List[ValidationFlag]] = {}
    
    for rs in row_slots:
        for a in rs.assignments:
            if a.slot != "LABEL":
                continue
            
            if a.element.ocr_block:
                text = a.element.ocr_block.text.lower().strip()
                text = re.sub(r'[*:\s]+$', '', text)  # Remove trailing markers
                
                if len(text) < 2:
                    continue
                
                if text not in label_texts:
                    label_texts[text] = []
                
                label_texts[text].append(a)
    
    # Find duplicates
    for text, assignments in label_texts.items():
        if len(assignments) > 1:
            flags.append(ValidationFlag(
                flag_type="warning",
                code="DUPLICATE_LABEL",
                message=f"Label '{text}' appears {len(assignments)} times",
                related_elements=[a.element.bbox for a in assignments],
            ))
    
    return flags


def validate_form_structure(
    row_slots: List[RowSlots],
) -> List[ValidationFlag]:
    """
    Validate overall form structure.
    
    Error: form has no inputs or no buttons.
    """
    flags = []
    
    total_inputs = 0
    total_actions = 0
    
    for rs in row_slots:
        for a in rs.assignments:
            if a.slot in ("INPUT", "TEXTAREA", "CHECKBOX", "RADIO"):
                total_inputs += 1
            if a.slot == "ACTION":
                total_actions += 1
    
    if total_inputs == 0:
        flags.append(ValidationFlag(
            flag_type="error",
            code="NO_INPUTS",
            message="Form has no input elements",
        ))
    
    if total_actions == 0:
        flags.append(ValidationFlag(
            flag_type="warning",
            code="NO_ACTIONS",
            message="Form has no action buttons",
        ))
    
    return flags


def validate_required_fields(
    row_slots: List[RowSlots],
) -> List[ValidationFlag]:
    """
    Identify required fields (marked with *).
    
    Info: field is marked as required.
    """
    flags = []
    
    for rs in row_slots:
        for a in rs.assignments:
            if a.slot != "LABEL":
                continue
            
            if a.element.ocr_block:
                text = a.element.ocr_block.text
                if "*" in text or "обязательно" in text.lower() or "required" in text.lower():
                    bound_slot = a.bound_to.slot if a.bound_to else "unknown"
                    flags.append(ValidationFlag(
                        flag_type="info",
                        code="REQUIRED_FIELD",
                        message=f"Required field: {text.strip('* ')} ({bound_slot})",
                        element_bbox=a.element.bbox,
                        row_index=rs.row.row_index,
                    ))
    
    return flags


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def validate_form(
    s4_result: S4Result,
    s5_result: S5Result,
    language: LanguageInfo,
) -> S6Result:
    """
    S6 — Semantic Validation.
    
    Validates form semantics. Does NOT modify geometry or slots!
    Only adds flags and warnings.
    
    Args:
        s4_result: результат S4 с slot assignments
        s5_result: результат S5 с patterns
        language: информация о языке
    
    Returns:
        S6Result с flags и validity
    """
    diagnostics: Dict[str, Any] = {
        "total_flags": 0,
        "errors": 0,
        "warnings": 0,
        "info": 0,
    }
    
    all_flags: List[ValidationFlag] = []
    row_slots = s4_result.row_slots
    patterns = s5_result.patterns
    
    # Run all validations
    all_flags.extend(validate_orphan_labels(row_slots))
    all_flags.extend(validate_orphan_inputs(row_slots))
    all_flags.extend(validate_checkbox_radio_labels(row_slots, patterns))
    all_flags.extend(validate_button_texts(row_slots, language))
    all_flags.extend(validate_language_consistency(row_slots, language))
    all_flags.extend(validate_duplicate_labels(row_slots))
    all_flags.extend(validate_form_structure(row_slots))
    all_flags.extend(validate_required_fields(row_slots))
    
    # Count by type
    errors = [f for f in all_flags if f.flag_type == "error"]
    warnings = [f for f in all_flags if f.flag_type == "warning"]
    info = [f for f in all_flags if f.flag_type == "info"]
    
    diagnostics["total_flags"] = len(all_flags)
    diagnostics["errors"] = len(errors)
    diagnostics["warnings"] = len(warnings)
    diagnostics["info"] = len(info)
    
    # Compute validity
    is_valid = len(errors) == 0
    
    # Confidence score (penalize warnings and errors)
    base_confidence = 1.0
    confidence_penalty = len(errors) * 0.3 + len(warnings) * 0.1
    confidence_score = max(0.0, base_confidence - confidence_penalty)
    
    logger.info(f"S6 completed: {len(all_flags)} flags (errors={len(errors)}, warnings={len(warnings)}), valid={is_valid}")
    
    return S6Result(
        flags=all_flags,
        is_valid=is_valid,
        confidence_score=confidence_score,
        diagnostics=diagnostics,
    )


def format_flags_report(s6_result: S6Result) -> str:
    """Format validation flags as human-readable report."""
    lines = []
    lines.append("=== Semantic Validation Report ===")
    lines.append(f"Valid: {s6_result.is_valid}")
    lines.append(f"Confidence: {s6_result.confidence_score:.2f}")
    lines.append("")
    
    if s6_result.flags:
        # Group by type
        errors = [f for f in s6_result.flags if f.flag_type == "error"]
        warnings = [f for f in s6_result.flags if f.flag_type == "warning"]
        info = [f for f in s6_result.flags if f.flag_type == "info"]
        
        if errors:
            lines.append("ERRORS:")
            for f in errors:
                lines.append(f"  [{f.code}] {f.message}")
            lines.append("")
        
        if warnings:
            lines.append("WARNINGS:")
            for f in warnings:
                lines.append(f"  [{f.code}] {f.message}")
            lines.append("")
        
        if info:
            lines.append("INFO:")
            for f in info:
                lines.append(f"  [{f.code}] {f.message}")
    else:
        lines.append("No issues found.")
    
    return "\n".join(lines)
