"""
GUI Detection Service Interface

Handles YOLO-based element detection and grouping into GUI blocks.
"""

from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

from ..models.bpg_models import GUIManifestation
from ..models.gui_block import GUIBlock


class GUIDetectionService(ABC):
    """
    Interface for GUI element detection and grouping.

    Domain rationale:
    - Abstracts YOLO implementation details
    - Enables swapping detection models
    - Allows testing with mock detections
    """

    @abstractmethod
    async def detect_gui_blocks(
        self,
        screenshot_path: str,
        ocr_text: str,
    ) -> List[GUIBlock]:
        """
        Detect GUI elements and group into semantic blocks.

        Args:
            screenshot_path: Path to screenshot image
            ocr_text: OCR text from screenshot

        Returns:
            List of detected GUI blocks
        """
        pass
