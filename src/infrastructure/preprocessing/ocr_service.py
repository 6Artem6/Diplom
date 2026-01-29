"""
OCR Service Interface

Placeholder for OCR implementation (Tesseract, PaddleOCR).
"""

from abc import ABC, abstractmethod


class OCRService(ABC):
    """Interface for OCR services."""

    @abstractmethod
    async def extract_text(self, image_path: str) -> str:
        """Extract text from image."""
        pass
