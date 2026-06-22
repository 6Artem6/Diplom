from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


def flatten_gui_blocks(roots: List["GUIBlock"]) -> List["GUIBlock"]:
    """
    Flatten tree of GUIBlocks (self + children recursively) for embedding/linking.
    Overlay and downstream use this when backends return nested structures.
    """
    out: List[GUIBlock] = []
    for b in roots:
        out.append(b)
        if getattr(b, "children", None):
            out.extend(flatten_gui_blocks(b.children))
    return out


class GUIBlock(BaseModel):
    """
    Grouped GUI elements forming a semantic block.

    Domain rationale:
    - GUI blocks are intermediate representation between raw detection and entities
    - Enables grouping related elements (e.g., "product card" = image + title + price)
    - LLM agents can reason about blocks as atomic UI units
    - children: optional nested blocks (e.g. card → text + buttons)
    """

    id: str
    screenshot_id: str
    bounding_box: dict  # {'x','y','width','height'} and/or {'x1','y1','x2','y2'}
    element_types: List[str]  # e.g. ['button'], ['text'], ['card']
    ocr_text: str = ""  # cleaned text (post-normalization)
    ocr_text_raw: str = ""  # raw Tesseract output
    ocr_noisy: bool = False  # True if normalization marked noisy OCR
    visual_features: List[float] = []
    screenshot_path: Optional[str] = None
    children: Optional[List["GUIBlock"]] = None  # nested elements (card → text, buttons)
