"""
State Machine Pipeline для анализа форм.

Этапы:
- S1: Visual Geometry Extraction (visual_geometry_extractor)
- S2: OCR Extraction (ocr_extractor)
- S3: Structural Segmentation (structural_segmentation)
- S4: Slot Assignment (slot_assignment)
- S5: Pattern Analysis (pattern_analysis)
- S6: Semantic Validation (semantic_validation)
- S7: Pipeline Integration (run_state_machine_pipeline)
"""

from .visual_geometry_extractor import (
    VisualElement,
    GeometryContext,
    S1Result,
    extract_visual_geometry,
)

from .ocr_extractor import (
    OCRBlock,
    LanguageInfo,
    S2Result,
    extract_ocr,
    detect_language,
)

from .structural_segmentation import (
    RowElement,
    FormRow,
    S3Result,
    segment_into_rows,
)

from .slot_assignment import (
    SlotAssignment,
    RowSlots,
    S4Result,
    assign_slots,
    get_form_atoms,
)

from .pattern_analysis import (
    UIPattern,
    S5Result,
    analyze_patterns,
)

from .semantic_validation import (
    ValidationFlag,
    S6Result,
    validate_form,
    format_flags_report,
)

from .run_state_machine_pipeline import (
    PipelineConfig,
    PipelineResult,
    run_state_machine_pipeline,
    run_pipeline_batch,
)

# S0: Container Detection
from .form_container_detector import (
    detect_form_containers,
    get_best_container,
)

__all__ = [
    # S1
    "VisualElement",
    "GeometryContext", 
    "S1Result",
    "extract_visual_geometry",
    # S2
    "OCRBlock",
    "LanguageInfo",
    "S2Result",
    "extract_ocr",
    "detect_language",
    # S3
    "RowElement",
    "FormRow",
    "S3Result",
    "segment_into_rows",
    # S4
    "SlotAssignment",
    "RowSlots",
    "S4Result",
    "assign_slots",
    "get_form_atoms",
    # S5
    "UIPattern",
    "S5Result",
    "analyze_patterns",
    # S6
    "ValidationFlag",
    "S6Result",
    "validate_form",
    "format_flags_report",
    # S7
    "PipelineConfig",
    "PipelineResult",
    "run_state_machine_pipeline",
    "run_pipeline_batch",
    # S0
    "detect_form_containers",
    "get_best_container",
]
