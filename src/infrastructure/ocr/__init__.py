"""
OCR infrastructure: per-block text enrichment and OCR-line blocks.

Layout is produced by VisualLayoutDetector; OCR runs inside those boxes.
get_ocr_line_blocks adds text-line blocks for plain text (e.g. "Products", "iPhone 14").
"""

from .ocr_enrichment import enrich_blocks_with_ocr, get_ocr_line_blocks

__all__ = ["enrich_blocks_with_ocr", "get_ocr_line_blocks"]
