"""
Apply OCR normalization to GUI blocks after detection.
"""

from __future__ import annotations

import logging
from typing import List

from src.application.detection.element_builder import resolve_element_class
from src.application.ocr.ocr_normalizer import normalize_ocr
from src.domain.models.gui_block import GUIBlock

logger = logging.getLogger(__name__)


def normalize_blocks_ocr(blocks: List[GUIBlock]) -> None:
    """
    Mutate blocks: preserve raw OCR, set cleaned text on ocr_text.

    Logs before/after samples for debug (first N blocks).
    """
    log_samples = int(__import__("os").environ.get("OCR_NORMALIZE_LOG_SAMPLES", "5"))
    for i, block in enumerate(blocks):
        raw = (block.ocr_text or "").strip()
        block.ocr_text_raw = raw
        result = normalize_ocr(raw)
        block.ocr_text = result.cleaned_text
        block.ocr_noisy = result.is_noisy

        if i < log_samples and raw:
            logger.info(
                "OCR normalize [%s] class=%s raw=%r → cleaned=%r noisy=%s changes=%s",
                block.id,
                resolve_element_class(block, {}),
                raw[:120],
                result.cleaned_text[:120],
                result.is_noisy,
                result.changes,
            )
