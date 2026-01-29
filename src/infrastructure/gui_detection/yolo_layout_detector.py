"""
YOLO-based Visual Layout Detector.

Wraps YOLODetectorImpl; implements VisualLayoutDetector.
Maps detections to RawLayoutBox (bbox + coarse_type). No OCR, no hierarchy.
"""

from pathlib import Path
from typing import List

from src.domain.interfaces.visual_layout_detector import (
    VisualLayoutDetector,
    RawLayoutBox,
)
from .yolo_detector import YOLODetector

import logging

logger = logging.getLogger(__name__)


class YOLOLayoutDetector(VisualLayoutDetector):
    """
    Layout detection via YOLO: bbox-first, deterministic.
    Implements VisualLayoutDetector; wraps existing YOLODetector.
    """

    def __init__(self, yolo_detector: YOLODetector) -> None:
        self.yolo_detector = yolo_detector

    async def detect_layout(self, image_path: str) -> List[RawLayoutBox]:
        """
        Run YOLO, map each detection to RawLayoutBox(bbox, coarse_type).
        """
        detections = await self.yolo_detector.detect(image_path)
        out: List[RawLayoutBox] = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x, y = float(x1), float(y1)
            w, h = float(x2 - x1), float(y2 - y1)
            bbox = {
                "x": int(x), "y": int(y), "width": int(w), "height": int(h),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            }
            out.append({
                "bbox": bbox,
                "coarse_type": det["class_label"],
            })
        logger.debug(
            "YOLOLayoutDetector: path=%s -> %d layout box(es)",
            image_path, len(out),
        )
        return out
