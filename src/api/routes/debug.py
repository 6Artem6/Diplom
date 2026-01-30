"""
Debug HTTP endpoints for layout / text detection / OCR.

No global flags; all parameters from request. Logs inputs and saves debug images to DEBUG_OUTPUT_DIR.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, List

import json
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from src.infrastructure.debug import (
    run_layout,
    run_text_detect,
    run_ocr_boxes,
    run_full_pipeline,
    save_debug_image_regions,
    save_debug_image_boxes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


def _save_upload_and_get_path(upload: UploadFile, suffix: str = ".png") -> str:
    """Save uploaded file to temp; return path. Caller must delete if needed."""
    ext = Path(upload.filename or "").suffix or suffix
    path = Path(tempfile.gettempdir()) / f"debug_{uuid.uuid4().hex}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = upload.file.read()
    path.write_bytes(content)
    return str(path)


# --- Response models ---


class RegionOut(BaseModel):
    x: int
    y: int
    w: int
    h: int
    type: str  # text_region | ui_region | background


class LayoutResponse(BaseModel):
    regions: List[RegionOut]
    debug_image_path: str | None = None
    message: str = "Layout only; no OCR."


class TextBoxOut(BaseModel):
    x: int
    y: int
    w: int
    h: int


class TextDetectResponse(BaseModel):
    boxes: List[TextBoxOut]
    debug_image_path: str | None = None
    message: str = "Text detection only; no layout, no OCR."


class OcrResultOut(BaseModel):
    text: str
    confidence: float


class OcrResponse(BaseModel):
    results: List[OcrResultOut]
    message: str = "OCR on given boxes."


class FullPipelineResponse(BaseModel):
    regions: List[RegionOut]
    text_boxes: List[TextBoxOut]
    gui_blocks: List[Any]  # dict with container_bbox, text_bbox, text, etc.
    dropped_count: int = 0
    drop_reasons: List[str] = Field(default_factory=list)
    debug_image_path: str | None = None
    message: str = "Full pipeline: layout → text detect → OCR → blocks."


# --- Endpoints ---


@router.post("/layout", response_model=LayoutResponse)
async def debug_layout(image: UploadFile = File(..., description="Screenshot image")):
    """
    Layout detection only. Returns region bbox + type (text / ui / background). No OCR.
    Saves debug image with regions to DEBUG_OUTPUT_DIR.
    """
    logger.info("debug/layout: filename=%s content_type=%s", image.filename, image.content_type)
    path = _save_upload_and_get_path(image)
    try:
        regions = run_layout(path)
        out_name = f"layout_{Path(path).stem}.png"
        debug_path = save_debug_image_regions(path, regions, out_name)
        return LayoutResponse(
            regions=[RegionOut(x=r["x"], y=r["y"], w=r["w"], h=r["h"], type=r["type"]) for r in regions],
            debug_image_path=debug_path,
        )
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/text-detect", response_model=TextDetectResponse)
async def debug_text_detect(image: UploadFile = File(..., description="Screenshot image")):
    """
    Text detection only. Returns bbox/polygons of text. No layout, no OCR.
    Uses PaddleOCR if available; else returns empty boxes and logs.
    Saves debug image with boxes to DEBUG_OUTPUT_DIR.
    """
    logger.info("debug/text-detect: filename=%s content_type=%s", image.filename, image.content_type)
    path = _save_upload_and_get_path(image)
    try:
        boxes = run_text_detect(path)
        out_name = f"text_detect_{Path(path).stem}.png"
        debug_path = save_debug_image_boxes(path, boxes, out_name)
        return TextDetectResponse(
            boxes=[TextBoxOut(x=b["x"], y=b["y"], w=b["w"], h=b["h"]) for b in boxes],
            debug_image_path=debug_path,
        )
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/ocr", response_model=OcrResponse)
async def debug_ocr(
    image: UploadFile = File(..., description="Screenshot image"),
    boxes_json: str = Form("[]", description='JSON array of bbox: [{"x", "y", "w", "h"}, ...]'),
):
    """
    OCR on given bbox. Form: image=file, boxes_json='[{"x":0,"y":0,"w":100,"h":20}]'.
    Returns list of {text, confidence} per box. No layout, no text detection.
    """
    try:
        boxes_list = json.loads(boxes_json) if boxes_json else []
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid boxes_json: {e}")
    boxes = [b if isinstance(b, dict) else {"x": 0, "y": 0, "w": 1, "h": 1} for b in boxes_list]
    logger.info("debug/ocr: filename=%s boxes_count=%d", image.filename, len(boxes))
    path = _save_upload_and_get_path(image)
    try:
        results = run_ocr_boxes(path, boxes)
        return OcrResponse(
            results=[OcrResultOut(text=r["text"], confidence=r["confidence"]) for r in results],
        )
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/full-pipeline", response_model=FullPipelineResponse)
async def debug_full_pipeline(image: UploadFile = File(..., description="Screenshot image")):
    """
    Full pipeline: layout → text detection (inside region) → OCR → GUI blocks.
    Returns regions, text_boxes, gui_blocks. Saves debug image to DEBUG_OUTPUT_DIR.
    """
    logger.info("debug/full-pipeline: filename=%s content_type=%s", image.filename, image.content_type)
    path = _save_upload_and_get_path(image)
    try:
        out = run_full_pipeline(path)
        regions = out["regions"]
        text_boxes = out["text_boxes"]
        gui_blocks = out["gui_blocks"]
        out_name = f"full_{Path(path).stem}.png"
        debug_path = save_debug_image_regions(path, regions, out_name)
        if text_boxes and debug_path:
            debug_path_boxes = save_debug_image_boxes(
                path, text_boxes, f"full_boxes_{Path(path).stem}.png", color=(80, 80, 255)
            )
            if debug_path_boxes:
                debug_path = debug_path_boxes
        return FullPipelineResponse(
            regions=[RegionOut(x=r["x"], y=r["y"], w=r["w"], h=r["h"], type=r["type"]) for r in regions],
            text_boxes=[TextBoxOut(x=b["x"], y=b["y"], w=b["w"], h=b["h"]) for b in text_boxes],
            gui_blocks=gui_blocks,
            dropped_count=out.get("dropped_count", 0),
            drop_reasons=out.get("drop_reasons", []),
            debug_image_path=debug_path,
        )
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
