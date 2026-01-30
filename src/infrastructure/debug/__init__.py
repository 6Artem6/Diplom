"""Debug services for layout / text detection / OCR. Used by /debug/* HTTP endpoints."""

from .services import (
    run_layout,
    run_ui_regions,
    run_text_detect,
    run_text_detect_roi,
    run_text_detect_per_regions,
    run_ocr_boxes,
    run_full_pipeline,
    save_debug_image_regions,
    save_debug_image_boxes,
    save_debug_image_ui_regions_hierarchy,
    save_debug_image_full_pipeline,
)

__all__ = [
    "run_layout",
    "run_ui_regions",
    "run_text_detect",
    "run_text_detect_roi",
    "run_text_detect_per_regions",
    "run_ocr_boxes",
    "run_full_pipeline",
    "save_debug_image_regions",
    "save_debug_image_boxes",
    "save_debug_image_ui_regions_hierarchy",
    "save_debug_image_full_pipeline",
]
