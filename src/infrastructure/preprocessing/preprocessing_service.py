"""
Preprocessing Service Implementation

Skeleton implementation with OCR placeholder.
"""

from typing import List
from pathlib import Path

from src.domain.interfaces.preprocessing import (
    PreprocessingService,
    ScreenshotData,
)


class PreprocessingServiceImpl(PreprocessingService):
    """
    Skeleton implementation of preprocessing service.

    Architecture rationale:
    - Separates data loading from OCR logic
    - Enables swapping OCR implementations (Tesseract, PaddleOCR)
    - Placeholder OCR returns empty string (no real OCR in skeleton)
    """

    def __init__(self, ocr_service=None):
        """
        Initialize preprocessing service.

        Args:
            ocr_service: Optional OCR service (placeholder: None)
        """
        self.ocr_service = ocr_service

    async def load_screenshots(
        self,
        screenshot_paths: List[Path],
    ) -> List[ScreenshotData]:
        """
        Load screenshots and extract OCR text.

        Skeleton implementation:
        - Validates file existence
        - Generates screenshot IDs
        - Placeholder OCR (returns empty string)
        """
        screenshots = []
        for path in screenshot_paths:
            if not path.exists():
                raise FileNotFoundError(f"Screenshot not found: {path}")

            # Use path stem so blocks (which use Path(screenshot_path).stem) match views
            screenshot_id = Path(path).stem

            # Placeholder OCR (real implementation would call OCR service)
            ocr_text = ""
            if self.ocr_service:
                ocr_text = await self.ocr_service.extract_text(str(path))
            else:
                # Skeleton: return empty OCR
                ocr_text = ""

            screenshot = ScreenshotData(
                screenshot_id=screenshot_id,
                image_path=path,
                ocr_text=ocr_text,
                metadata={"path": str(path)},
            )
            screenshots.append(screenshot)

        return screenshots
