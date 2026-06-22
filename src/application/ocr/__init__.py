"""OCR normalization application layer."""

from .ocr_normalizer import (
    OCRNormalizationResult,
    build_embedding_input,
    normalize_class_for_matching,
    normalize_ocr,
)

__all__ = [
    "OCRNormalizationResult",
    "build_embedding_input",
    "normalize_class_for_matching",
    "normalize_ocr",
]
