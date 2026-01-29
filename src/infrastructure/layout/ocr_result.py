"""
Output contract for OCR → lines → blocks pipeline (Tesseract/PDF-style).

Each block: lines, bounding_box (x0,y0,x1,y1), text.
Each line: words, text.
Provenance: which words and lines belong to each block is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .atoms import Word, Line, TextBlock


@dataclass
class LineOutput:
    """One visual line: list of words and concatenated text (provenance preserved)."""

    words: List[Word]
    text: str

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """x0, y0, x1, y1 (union of word bboxes)."""
        if not self.words:
            return 0, 0, 0, 0
        x0 = min(w.x for w in self.words)
        y0 = min(w.y for w in self.words)
        x1 = max(w.x + w.w for w in self.words)
        y1 = max(w.y + w.h for w in self.words)
        return x0, y0, x1, y1


@dataclass
class BlockOutput:
    """One text block (paragraph): lines, bounding_box, text (provenance preserved)."""

    lines: List[LineOutput]
    bounding_box: Tuple[int, int, int, int]  # x0, y0, x1, y1
    text: str


def from_text_blocks(
    text_blocks: List[TextBlock],
) -> Tuple[List[BlockOutput], List[LineOutput]]:
    """
    Build spec output from pipeline result: list of blocks and flat list of lines.

    - blocks: each with lines, bounding_box (x0,y0,x1,y1), text
    - lines: each with words and text (provenance: which words/lines in each block)
    """
    all_lines_out: List[LineOutput] = []
    blocks_out: List[BlockOutput] = []

    for tb in text_blocks:
        line_outputs: List[LineOutput] = []
        for ln in tb.lines:
            text = ln.text
            lo = LineOutput(words=list(ln.words), text=text)
            line_outputs.append(lo)
            all_lines_out.append(lo)
        x0, y0 = tb.x, tb.y
        x1, y1 = tb.x + tb.w, tb.y + tb.h
        text = "\n".join(lo.text for lo in line_outputs)
        blocks_out.append(
            BlockOutput(
                lines=line_outputs,
                bounding_box=(x0, y0, x1, y1),
                text=text,
            )
        )

    return blocks_out, all_lines_out
