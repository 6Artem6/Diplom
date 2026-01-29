from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

BlockType = Literal["header", "paragraph", "standalone"]
TextColorClass = Literal["dark", "gray", "light"]
LineRole = Literal["body", "header", "button", "label"]


@dataclass
class Word:
    """Canonical OCR atom: single word with bbox and optional confidence.
    Visual attributes (set by CV prepass): has_background, bg_color_cluster, text_color_class.
    """

    text: str
    x: int
    y: int
    w: int
    h: int
    conf: Optional[float] = None
    # CV prepass: button/pill/CTA vs paragraph text
    has_background: bool = False
    bg_color_cluster: Optional[int] = None
    text_color_class: TextColorClass = "dark"
    # Typography: 400=normal, 700=bold (from OCR if available). Used for header vs body.
    font_weight: Optional[float] = None


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

