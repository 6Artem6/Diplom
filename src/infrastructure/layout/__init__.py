"""
Internal layout primitives for flow-based backend (OCR → lines → blocks).

These models are NOT exposed to the domain layer; they are used only inside
infrastructure/layout and mapped to GUIBlock at the very end.

Output contract (ocr_result): blocks with lines, bounding_box (x0,y0,x1,y1), text;
lines with words and text; provenance preserved.
"""

from .atoms import Word, Line, TextBlock
from .ocr_result import BlockOutput, LineOutput, from_text_blocks as ocr_result_from_blocks

__all__ = [
    "Word",
    "Line",
    "TextBlock",
    "BlockOutput",
    "LineOutput",
    "ocr_result_from_blocks",
]

