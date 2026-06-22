"""
OCR enrichment: fill block.ocr_text from crop(image, block.bbox);
optional OCR-line blocks for plain text lines not covered by layout.

Architecture (docs/ARCHITECTURE_ANALYSIS.md):
- Layout is produced by VisualLayoutDetector; OCR runs *inside* those boxes.
- get_ocr_line_blocks() adds extra GUIBlocks for text lines (e.g. "Products", "iPhone 14")
  when used in yolo_clip path so plain text is not ignored.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

from src.domain.models.gui_block import GUIBlock

logger = logging.getLogger(__name__)

# Языки OCR: английский + русский (можно переопределить через OCR_LANGS_TESSERACT)
OCR_LANGS_TESSERACT = os.environ.get("OCR_LANGS_TESSERACT", "eng+rus")

# Minimum crop size to run OCR (avoid tiny crops)
MIN_CROP_W = 4
MIN_CROP_H = 4
# OCR line grouping
LINE_DY_PX = 14  # max vertical distance for same line
MIN_LINE_OVERLAP_RATIO = 0.7  # skip line if overlap with exclude_bbox > this


def _bbox_to_xyxy(box: dict) -> tuple[int, int, int, int]:
    """Get (x1, y1, x2, y2) from block.bounding_box."""
    x1 = box.get("x1", box.get("x", 0))
    y1 = box.get("y1", box.get("y", 0))
    x2 = box.get("x2", x1 + box.get("width", 0))
    y2 = box.get("y2", y1 + box.get("height", 0))
    return int(x1), int(y1), int(x2), int(y2)


def enrich_blocks_with_ocr(
    image_path: str,
    blocks: List[GUIBlock],
    psm: int = 6,
) -> None:
    """
    For each block, run OCR on crop(image, block.bbox) and set block.ocr_text.
    Mutates blocks in place. Skips blocks with too-small crop or on OCR failure.
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("OCR enrichment: image not found %s", image_path)
        return
    try:
        from PIL import Image
        import pytesseract
    except ImportError as e:
        logger.debug("OCR enrichment: pytesseract/PIL not available: %s", e)
        return

    try:
        img = Image.open(path).convert("RGB")
        w_img, h_img = img.size
    except Exception as e:
        logger.warning("OCR enrichment: failed to load image %s: %s", image_path, e)
        return

    for block in blocks:
        x1, y1, x2, y2 = _bbox_to_xyxy(block.bounding_box)
        # Clamp to image
        x1 = max(0, min(x1, w_img))
        y1 = max(0, min(y1, h_img))
        x2 = max(x1, min(x2, w_img))
        y2 = max(y1, min(y2, h_img))
        cw, ch = x2 - x1, y2 - y1
        if cw < MIN_CROP_W or ch < MIN_CROP_H:
            continue
        try:
            crop = img.crop((x1, y1, x2, y2))
            text = pytesseract.image_to_string(crop, lang=OCR_LANGS_TESSERACT, config=f"--psm {psm}").strip()
            if text:
                block.ocr_text_raw = text
                block.ocr_text = text
        except Exception as e:
            logger.debug("OCR enrichment: block %s crop failed: %s", block.id, e)


def _ocr_words(image_path: str, psm: int = 6) -> List[Tuple[str, int, int, int, int]]:
    """Return (text, x, y, w, h) per word using pytesseract."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return []
    path = Path(image_path)
    if not path.exists():
        return []
    try:
        img = Image.open(path).convert("RGB")
        data = pytesseract.image_to_data(img, lang=OCR_LANGS_TESSERACT, output_type=pytesseract.Output.DICT, config=f"--psm {psm}")
        out: List[Tuple[str, int, int, int, int]] = []
        for i, t in enumerate(data.get("text", []) or []):
            t = (t or "").strip()
            if not t:
                continue
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            if w >= MIN_CROP_W and h >= MIN_CROP_H:
                out.append((t, x, y, w, h))
        return out
    except Exception as e:
        logger.debug("OCR line blocks: pytesseract error %s", e)
        return []


def _group_into_lines(
    words: List[Tuple[str, int, int, int, int]],
    line_dy: int = LINE_DY_PX,
) -> List[List[Tuple[str, int, int, int, int]]]:
    """Group words by similar y into lines."""
    if not words:
        return []
    sorted_w = sorted(words, key=lambda w: (w[2] + w[4] // 2, w[1]))
    lines: List[List[Tuple[str, int, int, int, int]]] = []
    cur = [sorted_w[0]]
    y_prev = sorted_w[0][2] + sorted_w[0][4] // 2
    for w in sorted_w[1:]:
        y_c = w[2] + w[4] // 2
        if abs(y_c - y_prev) <= line_dy:
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
            y_prev = y_c
    if cur:
        lines.append(cur)
    return lines


def _line_bbox(line: List[Tuple[str, int, int, int, int]]) -> Tuple[int, int, int, int]:
    """(x, y, w, h) union of word bboxes in line."""
    if not line:
        return 0, 0, 0, 0
    x1 = min(w[1] for w in line)
    y1 = min(w[2] for w in line)
    x2 = max(w[1] + w[3] for w in line)
    y2 = max(w[2] + w[4] for w in line)
    return x1, y1, x2 - x1, y2 - y1


def _overlap_ratio(
    line_xywh: Tuple[int, int, int, int],
    box: dict,
) -> float:
    """Area of intersection / line area; 0 if no overlap."""
    lx, ly, lw, lh = line_xywh
    if lw <= 0 or lh <= 0:
        return 0.0
    x1 = box.get("x1", box.get("x", 0))
    y1 = box.get("y1", box.get("y", 0))
    x2 = box.get("x2", x1 + box.get("width", 0))
    y2 = box.get("y2", y1 + box.get("height", 0))
    ix1 = max(lx, x1)
    iy1 = max(ly, y1)
    ix2 = min(lx + lw, x2)
    iy2 = min(ly + lh, y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    area_inter = (ix2 - ix1) * (iy2 - iy1)
    area_line = lw * lh
    return area_inter / area_line if area_line else 0.0


def get_ocr_line_blocks(
    image_path: str,
    screenshot_id: str,
    screenshot_path: str,
    exclude_bboxes: List[dict] | None = None,
    psm: int = 6,
    line_dy: int = LINE_DY_PX,
) -> List[GUIBlock]:
    """
    Build one GUIBlock per OCR text line (plain text not already inside layout blocks).

    Used in yolo_clip path so lines like "Products", "iPhone 14", "$799" become
    separate blocks. Lines that overlap heavily with exclude_bboxes are skipped
    to avoid duplicating button/card labels.
    """
    words = _ocr_words(image_path, psm=psm)
    if not words:
        return []
    lines = _group_into_lines(words, line_dy=line_dy)
    exclude = exclude_bboxes or []
    blocks: List[GUIBlock] = []
    for idx, line in enumerate(lines):
        x, y, w, h = _line_bbox(line)
        if w < MIN_CROP_W or h < MIN_CROP_H:
            continue
        # Skip if most of this line is inside an existing layout block (e.g. button label)
        if exclude:
            if any(_overlap_ratio((x, y, w, h), b) >= MIN_LINE_OVERLAP_RATIO for b in exclude):
                continue
        text = " ".join(w[0] for w in line)
        bid = f"{screenshot_id}_ocr_line_{idx}"
        blocks.append(
            GUIBlock(
                id=bid,
                screenshot_id=screenshot_id,
                screenshot_path=screenshot_path,
                bounding_box={
                    "x": x, "y": y, "width": w, "height": h,
                    "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                },
                element_types=["text"],
                ocr_text=text,
                ocr_text_raw=text,
                visual_features=[],
            )
        )
    if blocks:
        logger.info("OCR line blocks: added %d text-line block(s) for %s", len(blocks), image_path)
    return blocks
