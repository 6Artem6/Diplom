"""
YOLO Detector Interface

Placeholder for YOLO model integration.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class YOLODetector(ABC):
    """Interface for YOLO-based GUI element detection."""

    @abstractmethod
    async def detect(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect GUI elements in image.

        Returns:
            List of detections with bounding boxes and class labels
        """
        pass
