from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

BlockType = Literal["header", "paragraph", "standalone"]
TextColorClass = Literal["dark", "gray", "light"]
LineRole = Literal["body", "header", "button", "label"]
RegionType = Literal["text_region", "ui_region", "background"]


@dataclass
class Region:
    """CV-level region: top-level container. Words belong to one region; layout is inside region only."""

    x: int
    y: int
    w: int
    h: int
    region_type: RegionType
    area: int = 0  # w * h for diagnostics


@dataclass
class Word:
    """Canonical OCR atom: single word with bbox and optional confidence.

    Layout vs OCR: x, y, w, h are ALWAYS layout_bbox (original page coords).
    OCR may run on upscaled/dilated crops — that never changes layout bbox.
    """

    text: str
    x: int  # layout bbox (original page coords)
    y: int
    w: int
    h: int
    conf: Optional[float] = None
    # CV prepass: button/pill/CTA vs paragraph text
    has_background: bool = False
    bg_color_cluster: Optional[int] = None
    text_color_class: TextColorClass = "dark"
    font_weight: Optional[float] = None
    estimated_font_size_px: Optional[float] = None
    # Diagnostics: set when fallback OCR was run on this word
    ocr_fallback_dilation: bool = False
    ocr_fallback_inversion: bool = False
    ocr_fallback_upscale: float = 1.0
    # Bbox used for OCR (crop in page coords); layout uses x,y,w,h only
    ocr_bbox: Optional[tuple[int, int, int, int]] = None

    @property
    def layout_bbox(self) -> tuple[int, int, int, int]:
        """Layout bbox in page coords. All distances/merge use this."""
        return (self.x, self.y, self.w, self.h)


@dataclass
class Line:
    """Visual line: horizontally aligned group of words.
    is_header: from geometric header detection or from role.
    role: set by line_classifier after aggregation (never used to split words→lines).
    """

    words: List[Word]
    x: int
    y: int
    w: int
    h: int
    is_header: bool = False
    role: Optional[LineRole] = None
    # Median of words' estimated_font_size_px; used for merge/scale, not raw line.h.
    estimated_font_size_px: Optional[float] = None

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class HorizontalRule:
    """Horizontal divider/rule between content. Breaks block merge."""

    y_min: float
    y_max: float
    x_min: float = 0.0
    x_max: float = 1e9
    width_ratio: float = 1.0  # width / page_width, >= 0.6 to count as divider


@dataclass
class VerticalRule:
    """Vertical divider/rule between content. Breaks line merge."""

    x_min: float
    x_max: float
    y_min: float = 0.0
    y_max: float = 1e9


@dataclass
class TextBlock:
    """Paragraph / heading / message body: vertical stack of lines. block_type and stats for output."""

    lines: List[Line]
    x: int
    y: int
    w: int
    h: int
    block_type: BlockType = "paragraph"
    stats: Optional[Dict[str, Any]] = None

