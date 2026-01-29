"""
Visual Layout Detector Interface

Layout detection is bbox-first and deterministic: image → list of (bbox, coarse_type).
OCR and hierarchy are NOT part of this interface; they belong to separate steps.

Domain rationale:
- Corrected architecture (see docs/ARCHITECTURE_ANALYSIS.md):
  - Layout detector produces boxes + types only.
  - OCR enrichment fills text inside those boxes later.
  - Hierarchy is derived from bbox containment after layout exists.
- Enables swapping layout backends (YOLO primary, others optional) without
  mixing layout, OCR, and hierarchy in one abstraction.
"""

from abc import ABC, abstractmethod
from typing import List, TypedDict


class RawLayoutBox(TypedDict):
    """
    Single layout box from a visual detector: bbox + coarse type only.
    No ocr_text, no children — those come from enrichment and hierarchy steps.
    """
    bbox: dict  # {'x','y','width','height'} and/or {'x1','y1','x2','y2'}
    coarse_type: str  # e.g. 'button' | 'text' | 'card' | 'list' | 'header' | 'input' | 'image'


class VisualLayoutDetector(ABC):
    """
    Interface for deterministic, bbox-first layout detection.

    Implementations must return boxes from a detection/segmentation model,
    not from parsing VLM output or from OCR word clustering.
    """

    @abstractmethod
    async def detect_layout(self, image_path: str) -> List[RawLayoutBox]:
        """
        Detect UI elements: return list of (bbox, coarse_type) per element.

        Args:
            image_path: Path to screenshot image.

        Returns:
            List of raw layout boxes; each has 'bbox' and 'coarse_type'.
        """
        pass
