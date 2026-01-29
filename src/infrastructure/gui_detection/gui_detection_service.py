"""
GUI Detection Service Implementation

Production path (yolo_clip): VisualLayoutDetector (YOLO) → GUIBlocks → OCR enrichment.
Layout first, then per-block OCR; see docs/ARCHITECTURE_ANALYSIS.md.
"""

from typing import List
import logging
from pathlib import Path

from src.domain.interfaces.gui_detection import GUIDetectionService
from src.domain.models.gui_block import GUIBlock
from src.domain.interfaces.visual_layout_detector import RawLayoutBox
from .yolo_detector import YOLODetector
from .yolo_detector_impl import YOLODetectorImpl
from .yolo_layout_detector import YOLOLayoutDetector
from src.infrastructure.ocr import enrich_blocks_with_ocr, get_ocr_line_blocks

logger = logging.getLogger(__name__)


def _raw_boxes_to_gui_blocks(
    raw_boxes: List[RawLayoutBox],
    screenshot_id: str,
    screenshot_path: str,
    fallback_ocr: str,
) -> List[GUIBlock]:
    """Convert RawLayoutBox list to GUIBlock list; ocr_text left empty for enrichment."""
    blocks: List[GUIBlock] = []
    for i, raw in enumerate(raw_boxes):
        bbox = raw["bbox"]
        blocks.append(
            GUIBlock(
                id=f"{screenshot_id}_block_{i}",
                screenshot_id=screenshot_id,
                screenshot_path=screenshot_path,
                bounding_box=dict(bbox),
                element_types=[raw["coarse_type"]],
                ocr_text=fallback_ocr or "",
                visual_features=[],
            )
        )
    return blocks


class GUIDetectionServiceImpl(GUIDetectionService):
    """
    Production GUI detection: layout (YOLO) → blocks → OCR enrichment.

    Architecture (docs/ARCHITECTURE_ANALYSIS.md):
    - Layout from VisualLayoutDetector (YOLOLayoutDetector), bbox-first.
    - OCR runs inside each block's crop; preprocessing ocr_text used as fallback.
    """

    def __init__(self, yolo_detector: YOLODetector | None = None) -> None:
        if yolo_detector is None:
            self.yolo_detector: YOLODetector = YOLODetectorImpl()
        else:
            self.yolo_detector = yolo_detector
        self._layout_detector = YOLOLayoutDetector(self.yolo_detector)

    async def detect_gui_blocks(
        self,
        screenshot_path: str,
        ocr_text: str,
    ) -> List[GUIBlock]:
        """
        1) Layout: YOLOLayoutDetector.detect_layout → RawLayoutBox.
        2) Build GUIBlocks from raw boxes.
        3) OCR enrichment: per-block crop → block.ocr_text.
        4) Fallback: if block.ocr_text still empty, use preprocessing ocr_text.
        """
        screenshot_id = Path(screenshot_path).stem
        logger.info("GUIDetection: Detecting in %s (layout then OCR enrichment)", screenshot_path)

        raw_boxes = await self._layout_detector.detect_layout(screenshot_path)
        blocks = _raw_boxes_to_gui_blocks(
            raw_boxes, screenshot_id, screenshot_path, ocr_text
        )

        enrich_blocks_with_ocr(screenshot_path, blocks)

        for b in blocks:
            if not b.ocr_text and ocr_text:
                b.ocr_text = ocr_text

        # Add text-line blocks from OCR so plain text ("Products", "iPhone 14", "$799") is not ignored
        exclude_bboxes = [b.bounding_box for b in blocks]
        line_blocks = get_ocr_line_blocks(
            screenshot_path, screenshot_id, screenshot_path,
            exclude_bboxes=exclude_bboxes,
        )
        blocks.extend(line_blocks)

        logger.info(
            "GUIDetection: Created %d GUIBlock(s) from layout (yolo_clip + OCR enrichment + %d text-line)",
            len(blocks),
            len(line_blocks),
        )
        return blocks
