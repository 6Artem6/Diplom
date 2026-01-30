"""
Debug pipeline: layout → text detection → OCR → UI blocks.

Used by HTTP /debug/* endpoints. No global flags; all params passed explicitly.
Logs counts and drop reasons. Saves debug images to configurable output dir.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEBUG_OUTPUT_DIR = os.environ.get("DEBUG_OUTPUT_DIR", "debug/output")
MAX_BLOCK_SCREEN_RATIO = 0.8  # block ≥ 80% screen → invalid


def _ensure_debug_dir() -> Path:
    p = Path(DEBUG_OUTPUT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_layout(image_path: str) -> List[Dict[str, Any]]:
    """
    Layout detection (regions only). No OCR.
    Returns list of {"x", "y", "w", "h", "type"}.
    Uses OpenCV region prepass (layoutparser optional later).
    """
    try:
        from src.infrastructure.layout.region_prepass import cv_detect_regions
    except ImportError:
        logger.warning("debug: layout dependency not available")
        return []
    regions = cv_detect_regions(image_path)
    n_total = len(regions)
    n_text = sum(1 for r in regions if r.region_type == "text_region")
    n_ui = sum(1 for r in regions if r.region_type == "ui_region")
    n_bg = sum(1 for r in regions if r.region_type == "background")
    logger.info(
        "debug/layout: image_path=%s regions=%d (text=%d ui=%d bg=%d)",
        image_path, n_total, n_text, n_ui, n_bg,
    )
    return [
        {"x": r.x, "y": r.y, "w": r.w, "h": r.h, "type": r.region_type}
        for r in regions
    ]


def run_text_detect(image_path: str) -> List[Dict[str, Any]]:
    """
    Text detection only (bbox/polygons). No layout, no OCR.
    Returns list of {"x", "y", "w", "h"} (or "polygon" if available).
    Uses PaddleOCR if available; else returns [] and logs.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        logger.warning("debug/text-detect: cv2/PIL not available")
        return []
    path = Path(image_path)
    if not path.exists():
        logger.warning("debug/text-detect: image not found %s", image_path)
        return []
    try:
        img = np.array(Image.open(path).convert("RGB"))
    except Exception as e:
        logger.warning("debug/text-detect: failed to load %s: %s", image_path, e)
        return []

    boxes: List[Dict[str, Any]] = []
    try:
        from paddleocr import PaddleOCR
        ocr_engine = PaddleOCR(use_angle_cls=False, show_log=False, use_gpu=False)
        result = ocr_engine.ocr(img, cls=False)
        if result and result[0]:
            for line in result[0]:
                if not line or len(line) < 2:
                    continue
                box_points, (text, conf) = line[0], line[1]
                x_coords = [p[0] for p in box_points]
                y_coords = [p[1] for p in box_points]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "polygon": [[int(p[0]), int(p[1])] for p in box_points]})
        logger.info("debug/text-detect: image_path=%s boxes=%d (PaddleOCR)", image_path, len(boxes))
    except ImportError:
        logger.info(
            "debug/text-detect: PaddleOCR not available; install paddleocr for text detection. Returning empty boxes."
        )
    except Exception as e:
        logger.warning("debug/text-detect: PaddleOCR failed for %s: %s", image_path, e)
    return boxes


def run_ocr_boxes(image_path: str, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    OCR on given bbox list. Returns list of {"text", "confidence"} per box.
    """
    try:
        import numpy as np
        import pytesseract
        from PIL import Image
    except ImportError as e:
        logger.warning("debug/ocr: dependency not available: %s", e)
        return [{"text": "", "confidence": 0.0} for _ in boxes]
    path = Path(image_path)
    if not path.exists():
        logger.warning("debug/ocr: image not found %s", image_path)
        return [{"text": "", "confidence": 0.0} for _ in boxes]
    try:
        img = np.array(Image.open(path).convert("RGB"))
    except Exception as e:
        logger.warning("debug/ocr: failed to load %s: %s", image_path, e)
        return [{"text": "", "confidence": 0.0} for _ in boxes]

    results: List[Dict[str, Any]] = []
    for i, box in enumerate(boxes):
        x = int(box.get("x", 0))
        y = int(box.get("y", 0))
        w = int(box.get("w", 0))
        h = int(box.get("h", 0))
        if w <= 0 or h <= 0:
            results.append({"text": "", "confidence": 0.0})
            logger.debug("debug/ocr: box[%d] skipped (invalid w/h)", i)
            continue
        h_img, w_img = img.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            results.append({"text": "", "confidence": 0.0})
            continue
        crop = img[y1:y2, x1:x2]
        pil_crop = Image.fromarray(crop)
        try:
            text = pytesseract.image_to_string(pil_crop, config="--psm 7").strip()
            data = pytesseract.image_to_data(pil_crop, config="--psm 7", output_type=pytesseract.Output.DICT)
            confs = [c for c in data.get("conf", []) if c != -1]
            conf = float(sum(confs) / len(confs)) / 100.0 if confs else 0.0
        except Exception as e:
            logger.debug("debug/ocr: box[%d] tesseract failed: %s", i, e)
            text, conf = "", 0.0
        results.append({"text": text, "confidence": conf})
    logger.info("debug/ocr: image_path=%s boxes=%d results=%d", image_path, len(boxes), len(results))
    return results


def run_full_pipeline(image_path: str) -> Dict[str, Any]:
    """
    Full pipeline: layout → text detection (inside region) → OCR → GUI blocks.
    Returns {"regions", "text_boxes", "gui_blocks", "dropped_count", "drop_reasons"}.
    """
    regions = run_layout(image_path)
    # Text detect: full image (PaddleOCR returns all boxes); then assign boxes to regions
    text_boxes = run_text_detect(image_path)
    if not text_boxes:
        # Fallback: use existing flow layout to get word bboxes as "text_boxes" for demo
        try:
            from src.infrastructure.gui_analysis.flow_layout_service import _ocr_words_bbox
            raw = _ocr_words_bbox(image_path)
            text_boxes = [{"x": x, "y": y, "w": w, "h": h} for (_, x, y, w, h, _) in raw]
            logger.info("debug/full-pipeline: using OCR bbox fallback for text_boxes, count=%d", len(text_boxes))
        except Exception as e:
            logger.warning("debug/full-pipeline: fallback text_boxes failed: %s", e)
    ocr_results = run_ocr_boxes(image_path, text_boxes) if text_boxes else []

    # Build gui_blocks: regions as containers; boxes inside with text
    gui_blocks: List[Dict[str, Any]] = []
    drop_reasons: List[str] = []
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    except Exception:
        img_w, img_h = 800, 600
    image_area = img_w * img_h

    def box_inside_region(box: Dict[str, Any], reg: Dict[str, Any]) -> bool:
        bx, by, bw, bh = box.get("x", 0), box.get("y", 0), box.get("w", 0), box.get("h", 0)
        rx, ry, rw, rh = reg["x"], reg["y"], reg["w"], reg["h"]
        cx = bx + bw / 2
        cy = by + bh / 2
        return rx <= cx <= rx + rw and ry <= cy <= ry + rh

    for ri, reg in enumerate(regions):
        reg_type = reg.get("type", "text_region")
        inside_boxes: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
        for bi, box in enumerate(text_boxes):
            if box_inside_region(box, reg):
                ocr = ocr_results[bi] if bi < len(ocr_results) else {"text": "", "confidence": 0.0}
                inside_boxes.append((bi, box, ocr))
        texts = [ocr["text"] for (_, _, ocr) in inside_boxes if ocr.get("text")]
        text_bbox = None
        if inside_boxes:
            xs = [b["x"] for (_, b, _) in inside_boxes]
            ys = [b["y"] for (_, b, _) in inside_boxes]
            x2s = [b["x"] + b["w"] for (_, b, _) in inside_boxes]
            y2s = [b["y"] + b["h"] for (_, b, _) in inside_boxes]
            text_bbox = {"x": min(xs), "y": min(ys), "w": max(x2s) - min(xs), "h": max(y2s) - min(ys)}
        else:
            text_bbox = {"x": reg["x"], "y": reg["y"], "w": reg["w"], "h": reg["h"]}
        block_area = (text_bbox["w"] * text_bbox["h"]) if text_bbox else 0
        if image_area > 0 and block_area >= MAX_BLOCK_SCREEN_RATIO * image_area:
            drop_reasons.append(f"block area {block_area} >= 80% screen")
            continue
        gui_blocks.append({
            "region_index": ri,
            "region_type": reg_type,
            "container_bbox": reg,
            "text_bbox": text_bbox,
            "text": " ".join(texts),
            "text_boxes_count": len(inside_boxes),
        })
    logger.info(
        "debug/full-pipeline: image_path=%s regions=%d text_boxes=%d gui_blocks=%d dropped=%d",
        image_path, len(regions), len(text_boxes), len(gui_blocks), len(drop_reasons),
    )
    return {
        "regions": regions,
        "text_boxes": text_boxes,
        "gui_blocks": gui_blocks,
        "dropped_count": len(drop_reasons),
        "drop_reasons": drop_reasons,
    }


def save_debug_image_regions(
    image_path: str,
    regions: List[Dict[str, Any]],
    output_filename: str,
) -> Optional[str]:
    """Draw region bbox on image and save to DEBUG_OUTPUT_DIR. Returns path or None."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    out_dir = _ensure_debug_dir()
    out_path = out_dir / output_filename
    try:
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        colors = {"text_region": (100, 200, 100), "ui_region": (200, 150, 50), "background": (120, 120, 120)}
        for r in regions:
            x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
            t = r.get("type", "text_region")
            color = colors.get(t, (128, 128, 128))
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        img.save(out_path, "PNG")
        logger.info("debug: saved layout image to %s", out_path)
        return str(out_path)
    except Exception as e:
        logger.warning("debug: failed to save layout image %s: %s", out_path, e)
        return None


def save_debug_image_boxes(
    image_path: str,
    boxes: List[Dict[str, Any]],
    output_filename: str,
    color: Tuple[int, int, int] = (255, 100, 100),
) -> Optional[str]:
    """Draw text boxes on image and save to DEBUG_OUTPUT_DIR. Returns path or None."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    out_dir = _ensure_debug_dir()
    out_path = out_dir / output_filename
    try:
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for b in boxes:
            x, y, w, h = b.get("x", 0), b.get("y", 0), b.get("w", 0), b.get("h", 0)
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        img.save(out_path, "PNG")
        logger.info("debug: saved text-detect image to %s", out_path)
        return str(out_path)
    except Exception as e:
        logger.warning("debug: failed to save text-detect image %s: %s", out_path, e)
        return None
