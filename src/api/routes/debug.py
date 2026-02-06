"""
Debug HTTP endpoints for layout / text detection / OCR.

No global flags; all parameters from request. Logs inputs and saves debug images to DEBUG_OUTPUT_DIR.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

import json
from fastapi import APIRouter, File, Form, Query, UploadFile, HTTPException
from pydantic import BaseModel, Field

from src.infrastructure.debug import (
    run_layout,
    run_ui_regions,
    run_text_detect,
    run_ocr_boxes,
    run_full_pipeline,
    run_improved_full_pipeline,
    save_debug_image_regions,
    save_debug_image_boxes,
    save_debug_image_ui_regions_hierarchy,
    save_debug_image_full_pipeline,
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
    type: str
    parent_region_id: int | None = None  # None = parent (navbar, card, section)


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


class TextBoxDropReason(BaseModel):
    box_index: int
    reason: str


class FullPipelineResponse(BaseModel):
    regions: List[RegionOut]
    raw_paddle_text_boxes: List[TextBoxOut] = Field(default_factory=list, description="Raw Paddle boxes before assign/filter")
    text_boxes: List[TextBoxOut]
    text_box_drop_reasons: List[TextBoxDropReason] = Field(default_factory=list, description="Why each dropped box was excluded")
    lines: List[List[TextBoxOut]] = Field(default_factory=list)
    paragraphs: List[List[List[TextBoxOut]]] = Field(default_factory=list)
    gui_blocks: List[Any]
    button_labels: Dict[int, str] = Field(default_factory=dict)
    dropped_count: int = 0
    drop_reasons: List[str] = Field(default_factory=list)
    debug_image_path: str | None = None
    message: str = "Layout-first; raw/final boxes and drop reasons for transparency."
    ocr_called: bool = Field(description="Whether text detection (Paddle) was invoked")
    ocr_engine: str = Field(default="paddle", description="Engine used for text detection")
    bpg_called: bool = Field(default=False, description="Whether full-page fallback was run after 0 boxes")
    reason_why_text_boxes_empty: str | None = Field(default=None, description="If no text boxes: why (e.g. paddle_returned_empty_or_unavailable)")


class GUIBlockOut(BaseModel):
    x: int
    y: int
    w: int
    h: int
    type: str  # button | card | text | input
    text: str = ""
    confidence: float = 0.0
    parent: str | None = None  # layout_id e.g. L1 (container: card, header, table)


class ImprovedFullPipelineResponse(BaseModel):
    gui_blocks: List[GUIBlockOut]
    layout_log: List[str] = Field(default_factory=list, description="Step 1: Layout detection log")
    ocr_log: List[str] = Field(default_factory=list, description="Step 2: OCR per block log")
    merge_log: List[str] = Field(default_factory=list, description="Step 3: Merge log")
    ocr_skipped: bool = Field(description="True when PaddleOCR in-process disabled (e.g. Mac); text can still come from Tesseract)")
    debug_image_path: str | None = Field(default=None, description="Path to saved debug image (regions + gui_blocks)")
    message: str = "Layout → OCR (per block) → GUI blocks."


# --- Endpoints ---


# Blue for UI regions (layout-first)
_UI_REGIONS_COLOR = (0, 0, 255)
# Red for text_boxes
_TEXT_BOXES_COLOR = (255, 0, 0)


@router.post("/layout", response_model=LayoutResponse)
async def debug_layout(image: UploadFile = File(..., description="Screenshot image")):
    """
    Layout detection only (legacy region_prepass). Returns region bbox + type. No OCR.
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


@router.post("/ui-regions", response_model=LayoutResponse)
async def debug_ui_regions(image: UploadFile = File(..., description="Screenshot image")):
    """
    Hierarchical UI region detection: parents (card, navbar, section) then children (button, badge, text_region, etc.) inside ROI.
    Returns bbox + type + parent_region_id. Debug image: parent=blue, child=purple.
    """
    logger.info("debug/ui-regions: filename=%s content_type=%s", image.filename, image.content_type)
    path = _save_upload_and_get_path(image)
    try:
        regions = run_ui_regions(path)
        out_name = f"ui_regions_{Path(path).stem}.png"
        debug_path = save_debug_image_ui_regions_hierarchy(path, regions, out_name)
        return LayoutResponse(
            regions=[
                RegionOut(
                    x=r["x"], y=r["y"], w=r["w"], h=r["h"],
                    type=r["type"],
                    parent_region_id=r.get("parent_region_id"),
                )
                for r in regions
            ],
            debug_image_path=debug_path,
            message="UI regions (hierarchical); parent=blue, child=purple.",
        )
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/text-boxes", response_model=TextDetectResponse)
async def debug_text_boxes(image: UploadFile = File(..., description="Screenshot image")):
    """
    Text detection only (bbox of text). No layout, no OCR. Uses PaddleOCR if available.
    Saves debug image with text_boxes in red to DEBUG_OUTPUT_DIR.
    """
    logger.info("debug/text-boxes: filename=%s content_type=%s", image.filename, image.content_type)
    path = _save_upload_and_get_path(image)
    try:
        boxes = run_text_detect(path)
        out_name = f"text_boxes_{Path(path).stem}.png"
        debug_path = save_debug_image_boxes(path, boxes, out_name, color=_TEXT_BOXES_COLOR)
        return TextDetectResponse(
            boxes=[TextBoxOut(x=b["x"], y=b["y"], w=b["w"], h=b["h"]) for b in boxes],
            debug_image_path=debug_path,
            message="Text boxes only; no layout, no OCR.",
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


def _box_out(b: Any) -> TextBoxOut:
    return TextBoxOut(x=b["x"], y=b["y"], w=b["w"], h=b["h"])


@router.post("/full-pipeline", response_model=FullPipelineResponse)
async def debug_full_pipeline(
    image: UploadFile = File(..., description="Screenshot image"),
    use_dl_only: bool = Form(False, description="DL layout only, full-page Paddle text (no CV second pass)"),
):
    """
    Layout → text detection → OCR → gui_blocks.
    use_dl_only=True: DL layout only (no CV second pass), full-page Paddle text, assign by overlap. Types: card, section, text_region.
    use_dl_only=False: run_ui_regions (DL+CV or CV), Paddle per ROI.
    """
    logger.info("debug/full-pipeline: filename=%s use_dl_only=%s", image.filename, use_dl_only)
    path = _save_upload_and_get_path(image)
    try:
        out = run_full_pipeline(path, use_dl_only=use_dl_only)
        regions = out["regions"]
        raw_boxes = out.get("raw_paddle_text_boxes", [])
        text_boxes = out["text_boxes"]
        box_drop_reasons = out.get("text_box_drop_reasons", [])
        lines = out.get("lines", [])
        paragraphs = out.get("paragraphs", [])
        gui_blocks = out["gui_blocks"]
        button_labels = out.get("button_labels", {})
        out_name = f"full_{Path(path).stem}.png"
        debug_path = save_debug_image_full_pipeline(
            path, regions, text_boxes, lines, paragraphs, out_name,
            button_labels=button_labels,
            skip_region_ids=out.get("empty_parent_ids"),
        )
        return FullPipelineResponse(
            regions=[
                RegionOut(x=r["x"], y=r["y"], w=r["w"], h=r["h"], type=r["type"], parent_region_id=r.get("parent_region_id"))
                for r in regions
            ],
            raw_paddle_text_boxes=[_box_out(b) for b in raw_boxes],
            text_boxes=[_box_out(b) for b in text_boxes],
            text_box_drop_reasons=[TextBoxDropReason(box_index=d["box_index"], reason=d["reason"]) for d in box_drop_reasons],
            lines=[[_box_out(b) for b in ln] for ln in lines],
            paragraphs=[[[_box_out(b) for b in ln] for ln in para] for para in paragraphs],
            gui_blocks=gui_blocks,
            button_labels=button_labels,
            dropped_count=out.get("dropped_count", 0),
            drop_reasons=out.get("drop_reasons", []),
            debug_image_path=debug_path,
            message=out.get("message", "Layout-first; raw/final boxes and drop reasons for transparency."),
            ocr_called=out.get("ocr_called", True),
            ocr_engine=out.get("ocr_engine", "paddle"),
            bpg_called=out.get("bpg_called", False),
            reason_why_text_boxes_empty=out.get("reason_why_text_boxes_empty"),
        )
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/improved-full-pipeline", response_model=ImprovedFullPipelineResponse)
async def debug_improved_full_pipeline(
    image: UploadFile = File(..., description="Screenshot image"),
):
    """
    Three-step pipeline: Layout → OCR inside blocks (adaptive) → Merge to GUI blocks.
    Returns list of {x, y, w, h, type, text, confidence}. type in: button, card, text, input.
    When OCR disabled (Mac): text may be empty; blocks still returned.
    """
    logger.info("debug/improved-full-pipeline: filename=%s", image.filename)
    path = _save_upload_and_get_path(image)
    try:
        out = run_improved_full_pipeline(path)
        return ImprovedFullPipelineResponse(
            gui_blocks=[GUIBlockOut(x=b["x"], y=b["y"], w=b["w"], h=b["h"], type=b["type"], text=b.get("text", ""), confidence=b.get("confidence", 0.0), parent=b.get("parent")) for b in out["gui_blocks"]],
            layout_log=out.get("layout_log", []),
            ocr_log=out.get("ocr_log", []),
            merge_log=out.get("merge_log", []),
            ocr_skipped=out.get("ocr_skipped", False),
            debug_image_path=out.get("debug_image_path"),
            message=out.get("message", ""),
        )
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


# --- Atoms_v2 pipeline (отдельный режим, не заменяет improved-full-pipeline) ---


class AtomsV2PipelineResponse(BaseModel):
    """Результат пайплайна atoms_v2: CV (atoms) + параллельный OCR, merge & conflict resolution, legacy grouping."""
    unified_ui: List[Any] = Field(default_factory=list, description="Дерево UI: card, form, navbar, panel с children")
    atoms: List[Any] = Field(default_factory=list, description="Атомарные элементы от Detectron2")
    regions: List[Any] = Field(default_factory=list, description="CV visual regions")
    text_blocks: List[Any] = Field(default_factory=list, description="Текстовые блоки по регионам")
    independent_text_blocks: List[Any] = Field(default_factory=list, description="Независимые текстовые блоки (paragraph, label) из legacy grouping")
    lines: List[Any] = Field(default_factory=list, description="Строки текста (legacy grouping)")
    paragraphs: List[Any] = Field(default_factory=list, description="Абзацы (legacy grouping)")
    text_inside_ui: Dict[str, Any] = Field(default_factory=dict, description="atom_id -> [{text, bbox, confidence}] — текст внутри UI-элементов")
    log: List[str] = Field(default_factory=list, description="Шаги пайплайна")
    debug_image_path: str | None = Field(default=None, description="Путь к сохранённому отладочному изображению")
    message: str = "Atoms_v2: CV atoms + parallel OCR, merge, legacy grouping."


@router.post("/atoms-v2-pipeline", response_model=AtomsV2PipelineResponse)
async def debug_atoms_v2_pipeline(
    image: UploadFile = File(..., description="Screenshot image"),
    parallel_ocr: bool = Query(True, description="Полностраничный OCR независимо от Detectron2"),
    legacy_text_pipeline: bool = Query(True, description="Группировка в строки/абзацы, фильтр шума"),
):
    """
    Режим atoms_v2: Detectron2 — атомы; параллельный полностраничный OCR (текст не подавляется UI);
    merge & conflict resolution; legacy grouping (строки/абзацы). improved-full-pipeline не вызывается.

    parallel_ocr=True: OCR по всему изображению независимо от Detectron2 (текст внутри input/button тоже извлекается).
    legacy_text_pipeline=True: группировка в строки/абзацы, фильтр шума.
    """
    logger.info("debug/atoms-v2-pipeline: filename=%s parallel_ocr=%s legacy_text_pipeline=%s", image.filename, parallel_ocr, legacy_text_pipeline)
    path = _save_upload_and_get_path(image)
    try:
        from src.infrastructure.atoms_v2 import run_atoms_v2_pipeline
        out = run_atoms_v2_pipeline(path, parallel_ocr=parallel_ocr, legacy_text_pipeline=legacy_text_pipeline)
        return AtomsV2PipelineResponse(
            unified_ui=out.get("unified_ui", []),
            atoms=out.get("atoms", []),
            regions=out.get("regions", []),
            text_blocks=out.get("text_blocks", []),
            independent_text_blocks=out.get("independent_text_blocks", []),
            lines=out.get("lines", []),
            paragraphs=out.get("paragraphs", []),
            text_inside_ui=out.get("text_inside_ui", {}),
            log=out.get("log", []),
            debug_image_path=out.get("debug_image_path"),
        )
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
