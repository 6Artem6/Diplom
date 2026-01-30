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
    Uses OpenCV region prepass (legacy). Prefer run_ui_regions for layout-first.
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


def run_ui_regions(image_path: str) -> List[Dict[str, Any]]:
    """
    Layout-first UI region detection. No OCR.
    Returns list of {"x", "y", "w", "h", "type"}.
    type in: button, badge, card, text_region, navbar, background.
    Outline buttons by geometry (border), not by fill.
    """
    try:
        from src.infrastructure.layout.ui_region_detection import cv_detect_ui_regions
    except ImportError:
        logger.warning("debug: ui_region_detection not available")
        return []
    regions = cv_detect_ui_regions(image_path)
    logger.info("debug/ui-regions: image_path=%s regions=%d", image_path, len(regions))
    return regions


ROI_TEXT_SCALE = 2.0  # scale ROI before text detection (button/badge thin text)


def _run_text_detect_on_image(img: Any) -> List[Dict[str, Any]]:
    """Run PaddleOCR text detection on numpy image (BGR or RGB). Returns list of {x, y, w, h}."""
    boxes: List[Dict[str, Any]] = []
    try:
        import numpy as np
        from paddleocr import PaddleOCR
        if img is None or img.size == 0:
            return []
        ocr_engine = PaddleOCR(use_angle_cls=False, show_log=False, use_gpu=False)
        result = ocr_engine.ocr(img, cls=False)
        if result and result[0]:
            for line in result[0]:
                if not line or len(line) < 2:
                    continue
                box_points, _ = line[0], line[1]
                x_coords = [p[0] for p in box_points]
                y_coords = [p[1] for p in box_points]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    except ImportError:
        pass
    except Exception:
        pass
    return boxes


def run_text_detect(image_path: str) -> List[Dict[str, Any]]:
    """
    Text detection on full image (bbox only). No layout, no OCR.
    For pipeline prefer run_text_detect_per_regions so text is detected INSIDE each region (ROI + scale).
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return []
    path = Path(image_path)
    if not path.exists():
        return []
    try:
        img = np.array(Image.open(path).convert("RGB"))
    except Exception:
        return []
    boxes = _run_text_detect_on_image(img)
    logger.info("debug/text-detect: image_path=%s boxes=%d", image_path, len(boxes))
    return boxes


def run_text_detect_roi(
    image_path: str,
    region: Dict[str, Any],
    scale: float = ROI_TEXT_SCALE,
) -> List[Dict[str, Any]]:
    """
    Text detection INSIDE region: crop ROI, scale ×scale, run detector, remap bbox to global.
    Returns list of {x, y, w, h} in global image coordinates. Required for button/badge thin text.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        return []
    path = Path(image_path)
    if not path.exists():
        return []
    try:
        full = np.array(Image.open(path).convert("RGB"))
    except Exception:
        return []
    rx = int(region.get("x", 0))
    ry = int(region.get("y", 0))
    rw = int(region.get("w", 0))
    rh = int(region.get("h", 0))
    if rw <= 0 or rh <= 0:
        return []
    h_full, w_full = full.shape[:2]
    x2 = min(w_full, rx + rw)
    y2 = min(h_full, ry + rh)
    roi = full[ry:y2, rx:x2]
    if roi.size == 0:
        return []
    roi_h, roi_w = roi.shape[:2]
    scaled = cv2.resize(roi, (int(roi_w * scale), int(roi_h * scale)), interpolation=cv2.INTER_LINEAR)
    boxes_local = _run_text_detect_on_image(scaled)
    out: List[Dict[str, Any]] = []
    for b in boxes_local:
        out.append({
            "x": rx + int(b["x"] / scale),
            "y": ry + int(b["y"] / scale),
            "w": max(1, int(b["w"] / scale)),
            "h": max(1, int(b["h"] / scale)),
        })
    return out


def run_text_detect_per_regions(
    image_path: str,
    regions: List[Dict[str, Any]],
    scale: float = ROI_TEXT_SCALE,
) -> List[Dict[str, Any]]:
    """
    Run text detection INSIDE each region (crop ROI, scale, detect, remap). Each box gets region_id.
    Returns flat list of {x, y, w, h, region_id}. No global run — text inside buttons/badges is found.
    """
    all_boxes: List[Dict[str, Any]] = []
    for ri, reg in enumerate(regions):
        boxes = run_text_detect_roi(image_path, reg, scale=scale)
        for b in boxes:
            all_boxes.append({**b, "region_id": ri})
    logger.info("debug/text-detect-per-regions: image_path=%s regions=%d total_boxes=%d", image_path, len(regions), len(all_boxes))
    return all_boxes


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
    Hierarchical layout-first: UI regions (parents + children in ROI) → text detect INSIDE each region → OCR → blocks.
    Paragraphs ONLY for text_region; button/badge/card/navbar = one block per region, no paragraph grouping.
    """
    from src.infrastructure.layout.text_grouping import (
        group_text_boxes_into_lines,
        group_lines_into_paragraphs,
    )

    regions = run_ui_regions(image_path)
    text_boxes = run_text_detect_per_regions(image_path, regions)
    boxes_with_index: List[Dict[str, Any]] = []
    for i, b in enumerate(text_boxes):
        boxes_with_index.append({**b, "box_index": i})
    ocr_results = run_ocr_boxes(image_path, text_boxes) if text_boxes else []

    try:
        from PIL import Image
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    except Exception:
        img_w, img_h = 800, 600
    image_area = img_w * img_h

    all_lines: List[List[Dict[str, Any]]] = []
    all_paragraphs: List[List[List[Dict[str, Any]]]] = []
    gui_blocks: List[Dict[str, Any]] = []
    drop_reasons: List[str] = []

    for rid, reg in enumerate(regions):
        reg_type = reg.get("type", "text_region")
        boxes_in_region = [b for b in boxes_with_index if b.get("region_id") == rid]

        if reg_type == "text_region":
            lines = group_text_boxes_into_lines(boxes_with_index, rid)
            all_lines.extend(lines)
            paragraphs = group_lines_into_paragraphs(lines)
            all_paragraphs.extend(paragraphs)
            for para in paragraphs:
                boxes_in_para = [b for ln in para for b in ln]
                xs = [b["x"] for b in boxes_in_para]
                ys = [b["y"] for b in boxes_in_para]
                x2s = [b["x"] + b["w"] for b in boxes_in_para]
                y2s = [b["y"] + b["h"] for b in boxes_in_para]
                text_bbox = {"x": min(xs), "y": min(ys), "w": max(x2s) - min(xs), "h": max(y2s) - min(ys)}
                block_area = text_bbox["w"] * text_bbox["h"]
                if image_area > 0 and block_area >= MAX_BLOCK_SCREEN_RATIO * image_area:
                    drop_reasons.append(f"block area {block_area} >= 80% screen")
                    continue
                texts = [ocr_results[b["box_index"]].get("text", "") for b in boxes_in_para if 0 <= b.get("box_index", -1) < len(ocr_results)]
                gui_blocks.append({
                    "region_index": rid,
                    "region_type": reg_type,
                    "container_bbox": reg,
                    "text_bbox": text_bbox,
                    "text": " ".join(t for t in texts if t),
                    "text_boxes_count": len(boxes_in_para),
                })
        else:
            container_bbox = {"x": reg["x"], "y": reg["y"], "w": reg["w"], "h": reg["h"]}
            if boxes_in_region:
                xs = [b["x"] for b in boxes_in_region]
                ys = [b["y"] for b in boxes_in_region]
                x2s = [b["x"] + b["w"] for b in boxes_in_region]
                y2s = [b["y"] + b["h"] for b in boxes_in_region]
                text_bbox = {"x": min(xs), "y": min(ys), "w": max(x2s) - min(xs), "h": max(y2s) - min(ys)}
            else:
                text_bbox = dict(container_bbox)
            block_area = text_bbox["w"] * text_bbox["h"]
            if image_area > 0 and block_area >= MAX_BLOCK_SCREEN_RATIO * image_area:
                drop_reasons.append(f"block area {block_area} >= 80% screen")
            else:
                texts = [ocr_results[b["box_index"]].get("text", "") for b in boxes_in_region if 0 <= b.get("box_index", -1) < len(ocr_results)]
                gui_blocks.append({
                    "region_index": rid,
                    "region_type": reg_type,
                    "container_bbox": container_bbox,
                    "text_bbox": text_bbox,
                    "text": " ".join(t for t in texts if t),
                    "text_boxes_count": len(boxes_in_region),
                })

    logger.info(
        "debug/full-pipeline: image_path=%s regions=%d text_boxes=%d lines=%d paragraphs=%d gui_blocks=%d dropped=%d",
        image_path, len(regions), len(text_boxes), len(all_lines), sum(len(p) for p in all_paragraphs), len(gui_blocks), len(drop_reasons),
    )
    return {
        "regions": regions,
        "text_boxes": text_boxes,
        "lines": all_lines,
        "paragraphs": all_paragraphs,
        "gui_blocks": gui_blocks,
        "dropped_count": len(drop_reasons),
        "drop_reasons": drop_reasons,
    }


def save_debug_image_regions(
    image_path: str,
    regions: List[Dict[str, Any]],
    output_filename: str,
    outline_color_override: Optional[Tuple[int, int, int]] = None,
) -> Optional[str]:
    """Draw region bbox on image and save to DEBUG_OUTPUT_DIR. Returns path or None.
    If outline_color_override is set (e.g. (0,0,255) for UI regions), all regions use that color."""
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
            if outline_color_override is not None:
                color = outline_color_override
            else:
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


# Debug: parent=blue, child=purple, text_boxes=red, button text_boxes=green
COLOR_PARENT_REGIONS = (0, 0, 255)
COLOR_CHILD_REGIONS = (128, 0, 128)
COLOR_TEXT_BOXES = (255, 0, 0)
COLOR_BUTTON_TEXT_BOXES = (0, 255, 0)
COLOR_LINES = (0, 200, 100)
COLOR_PARAGRAPHS = (255, 255, 0)


def _dashed_rect(draw: Any, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], width: int = 2) -> None:
    dash, gap = 6, 4
    step = dash + gap
    for x in range(x1, x2, step):
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
    for x in range(x1, x2, step):
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
    for y in range(y1, y2, step):
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
    for y in range(y1, y2, step):
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)


def save_debug_image_ui_regions_hierarchy(
    image_path: str,
    regions: List[Dict[str, Any]],
    output_filename: str,
) -> Optional[str]:
    """Draw parent regions blue, child regions purple. Saves to DEBUG_OUTPUT_DIR."""
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
        for r in regions:
            x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
            if r.get("parent_region_id") is None:
                _dashed_rect(draw, x, y, x + w, y + h, COLOR_PARENT_REGIONS)
            else:
                _dashed_rect(draw, x, y, x + w, y + h, COLOR_CHILD_REGIONS)
        img.save(out_path, "PNG")
        logger.info("debug: saved ui-regions hierarchy image to %s", out_path)
        return str(out_path)
    except Exception as e:
        logger.warning("debug: failed to save ui-regions image %s: %s", out_path, e)
        return None


def save_debug_image_full_pipeline(
    image_path: str,
    regions: List[Dict[str, Any]],
    text_boxes: List[Dict[str, Any]],
    lines: List[List[Dict[str, Any]]],
    paragraphs: List[List[List[Dict[str, Any]]]],
    output_filename: str,
) -> Optional[str]:
    """
    Draw: parent regions blue (dashed), child regions purple (dashed), text_boxes red, button text_boxes green,
    then lines (green rect), paragraphs (yellow rect). text_boxes must have region_id for button coloring.
    """
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
        for r in regions:
            x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
            if r.get("parent_region_id") is None:
                _dashed_rect(draw, x, y, x + w, y + h, COLOR_PARENT_REGIONS)
            else:
                _dashed_rect(draw, x, y, x + w, y + h, COLOR_CHILD_REGIONS)
        for b in text_boxes:
            x, y, w, h = b.get("x", 0), b.get("y", 0), b.get("w", 0), b.get("h", 0)
            rid = b.get("region_id", -1)
            reg_type = regions[rid].get("type", "") if 0 <= rid < len(regions) else ""
            color = COLOR_BUTTON_TEXT_BOXES if reg_type == "button" else COLOR_TEXT_BOXES
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        for line in lines:
            if not line:
                continue
            xs = [bx["x"] for bx in line]
            ys = [bx["y"] for bx in line]
            x2s = [bx["x"] + bx["w"] for bx in line]
            y2s = [bx["y"] + bx["h"] for bx in line]
            draw.rectangle([min(xs), min(ys), max(x2s), max(y2s)], outline=COLOR_LINES, width=1)
        for para in paragraphs:
            if not para:
                continue
            boxes_para = [b for ln in para for b in ln]
            xs = [b["x"] for b in boxes_para]
            ys = [b["y"] for b in boxes_para]
            x2s = [b["x"] + b["w"] for b in boxes_para]
            y2s = [b["y"] + b["h"] for b in boxes_para]
            draw.rectangle([min(xs), min(ys), max(x2s), max(y2s)], outline=COLOR_PARAGRAPHS, width=2)
        img.save(out_path, "PNG")
        logger.info("debug: saved full-pipeline image to %s", out_path)
        return str(out_path)
    except Exception as e:
        logger.warning("debug: failed to save full-pipeline image %s: %s", out_path, e)
        return None
