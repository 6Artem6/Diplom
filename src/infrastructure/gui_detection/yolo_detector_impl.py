"""
UI YOLO Detector Implementation

Uses UI-trained YOLO (e.g. MacPaw yolov11l-ui-elements-detection) for GUI element detection.
Loads ONLY from local path; no HuggingFace/network at runtime.
"""

from typing import List, Dict, Any
import logging
import os
from pathlib import Path

from ultralytics import YOLO

from .yolo_detector import YOLODetector

logger = logging.getLogger(__name__)

# Supported UI classes (target set)
SUPPORTED_CLASSES = {"button", "input", "text", "list-item", "card", "image", "checkbox", "dropdown"}

# Map model-specific class names to canonical names (MacPaw uses AX* naming)
CLASS_ALIAS: Dict[str, str] = {
    "axbutton": "button",
    "button": "button",
    "axtextfield": "input",
    "input": "input",
    "axstatictext": "text",
    "text": "text",
    "axcell": "list-item",
    "list_item": "list-item",
    "list-item": "list-item",
    "aximage": "image",
    "image": "image",
    "axcheckbox": "checkbox",
    "checkbox": "checkbox",
    "axcombobox": "dropdown",
    "dropdown": "dropdown",
    "axgroup": "card",
    "card": "card",
}


def _normalize_class(label: str) -> str:
    """Map model class label to canonical UI class."""
    key = label.lower().replace(" ", "_").replace("-", "_")
    return CLASS_ALIAS.get(key, label.lower())


class YOLODetectorImpl(YOLODetector):
    """
    UI YOLO detector: loads from local path, no network.

    Architecture rationale:
    - UI-trained model yields ≥10 detections per view on bootstrap-like pages
    - Confidence from env (default 0.15); all thresholds via env
    - Startup: if model file missing → clear error, no auto-download
    - Logs each detection (class, confidence, bbox) for diagnostics
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        """
        Initialize UI YOLO detector from local path.

        Args:
            model_path: Path to .pt file; default from UI_YOLO_MODEL_PATH or /app/models/yolo/ui-elements-detection.pt
            confidence_threshold: Min confidence; default from DETECTION_CONFIDENCE_THRESHOLD or 0.15
        """
        self.model_path = Path(
            model_path or os.getenv("UI_YOLO_MODEL_PATH", "/app/models/yolo/ui-elements-detection.pt")
        )
        self.confidence_threshold = float(
            confidence_threshold
            if confidence_threshold is not None
            else os.getenv("DETECTION_CONFIDENCE_THRESHOLD", "0.15")
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"UI YOLO model not found at {self.model_path.resolve()}. "
                "Place ui-elements-detection.pt there for offline operation. "
                "Download once with: huggingface-cli download MacPaw/yolov11l-ui-elements-detection "
                "ui-elements-detection.pt --local-dir /app/models/yolo"
            )
        logger.info(
            "YOLODetector: Loading UI model from %s (conf=%.2f)",
            self.model_path,
            self.confidence_threshold,
        )
        self.model = YOLO(str(self.model_path))
        logger.info("YOLODetector: UI model loaded successfully")

    async def detect(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect GUI elements in image.

        Args:
            image_path: Path to screenshot image

        Returns:
            List of detections: bbox [x1,y1,x2,y2], class_label (canonical), confidence
        """
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info("YOLODetector: Detecting in %s", image_path)
        results = self.model(image_path, conf=self.confidence_threshold, verbose=False)

        detections: List[Dict[str, Any]] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                confidence = float(box.conf.cpu().item())
                class_id = int(box.cls.cpu().item())
                raw_label = r.names.get(class_id, str(class_id))
                class_label = _normalize_class(raw_label)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_label": class_label,
                    "confidence": confidence,
                })
                logger.debug(
                    "YOLODetector: detection class=%s conf=%.3f bbox=[%.0f,%.0f,%.0f,%.0f]",
                    class_label,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                )

        # Keep up to 50 per view (bootstrap pages expect ≥10)
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)[:50]

        logger.info(
            "YOLODetector: Found %d detections (threshold=%.2f) in %s",
            len(detections),
            self.confidence_threshold,
            image_path,
        )
        if len(detections) == 0:
            logger.warning(
                "YOLODetector: 0 detections — model=%s, path=%s, conf=%.2f. "
                "Check that the image contains UI elements and threshold is not too high.",
                self.model_path,
                image_path,
                self.confidence_threshold,
            )
        return detections
