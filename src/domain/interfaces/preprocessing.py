"""
Preprocessing Service Interface

Handles screenshot loading, OCR, and normalization.
"""

from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from pydantic import BaseModel


class ScreenshotData(BaseModel):
    """Normalized screenshot data with OCR."""

    screenshot_id: str
    image_path: Path
    ocr_text: str
    metadata: dict


class PreprocessingService(ABC):
    """
    Interface for preprocessing screenshots.

    Domain rationale:
    - Separates data loading from processing logic
    - Enables testing with mock data
    - Allows swapping OCR implementations
    """

    @abstractmethod
    async def load_screenshots(
        self,
        screenshot_paths: List[Path],
    ) -> List[ScreenshotData]:
        """
        Load and normalize screenshots.

        Args:
            screenshot_paths: Paths to screenshot files

        Returns:
            List of normalized screenshot data with OCR
        """
        pass
