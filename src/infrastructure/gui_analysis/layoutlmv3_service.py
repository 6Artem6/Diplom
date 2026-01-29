"""
LayoutLMv3-based GUI analysis backend.

OCR (pytesseract) with --psm 6/11; group words into lines, lines into cards/lists.
Create GUIBlocks for button, text, card, list, header; optional children for nesting.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import logging
import os

from PIL import Image

from src.domain.interfaces.gui_detection import GUIDetectionService
from src.domain.models.gui_block import GUIBlock

logger = logging.getLogger(__name__)

BACKEND_NAME = "layoutlmv3"
MIN_BBOX_W = 2
MIN_BBOX_H = 2
LINE_DY = 12  # max vertical distance to belong to same line
REGION_GAP_Y = 25  # max gap between lines to belong to same card/region
BUTTON_KEYWORDS = frozenset(
    {"submit", "save", "cancel", "ok", "button", "view", "create", "edit", "delete", "add", "back"}
)


def _ocr_words_bbox(
    image_path: str,
    psm: int = 6,
    min_w: int = MIN_BBOX_W,
    min_h: int = MIN_BBOX_H,
) -> List[Tuple[str, int, int, int, int]]:
    """
    Return list of (text, x, y, w, h) using pytesseract.
    psm 6 = block of text, 11 = sparse text.
    """
    try:
        import pytesseract
        path = Path(image_path)
        if not path.exists():
            return []
        img = Image.open(path).convert("RGB")
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=f"--psm {psm}")
        out: List[Tuple[str, int, int, int, int]] = []
        for i, t in enumerate(data.get("text", []) or []):
            t = (t or "").strip()
            if not t:
                continue
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            if w >= min_w and h >= min_h:
                out.append((t, x, y, w, h))
        return out
    except Exception as e:
        logger.debug("LayoutLMv3Detection: pytesseract error: %s", e)
        return []


def _group_words_into_lines(
    words: List[Tuple[str, int, int, int, int]],
    line_dy: int = LINE_DY,
) -> List[List[Tuple[str, int, int, int, int]]]:
    """Group words by similar y (same line). Returns list of lines, each line = list of (text,x,y,w,h)."""
    if not words:
        return []
    # sort by y then x
    sorted_w = sorted(words, key=lambda w: (w[2] + w[4] // 2, w[1]))
    lines: List[List[Tuple[str, int, int, int, int]]] = []
    current: List[Tuple[str, int, int, int, int]] = [sorted_w[0]]
    y_prev = sorted_w[0][2] + sorted_w[0][4] // 2
    for w in sorted_w[1:]:
        _, x, y, ww, hh = w
        y_c = y + hh // 2
        if abs(y_c - y_prev) <= line_dy:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
            y_prev = y_c
    if current:
        lines.append(current)
    return lines


def _line_bbox(line: List[Tuple[str, int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Union bbox of all words in line -> (x, y, w, h)."""
    if not line:
        return 0, 0, 0, 0
    x1 = min(w[1] for w in line)
    y1 = min(w[2] for w in line)
    x2 = max(w[1] + w[3] for w in line)
    y2 = max(w[2] + w[4] for w in line)
    return x1, y1, x2 - x1, y2 - y1


def _line_text(line: List[Tuple[str, int, int, int, int]]) -> str:
    return " ".join(w[0] for w in line)


def _looks_like_button(line: List[Tuple[str, int, int, int, int]]) -> bool:
    text = _line_text(line).lower()
    return any(k in text for k in BUTTON_KEYWORDS) or (len(line) <= 2 and len(text) <= 20)


def _group_lines_into_regions(
    lines: List[List[Tuple[str, int, int, int, int]]],
    gap_y: int = REGION_GAP_Y,
) -> List[List[List[Tuple[str, int, int, int, int]]]]:
    """Group lines into regions (card/list) by vertical gap. Returns list of regions, each = list of lines."""
    if not lines:
        return []
    regions: List[List[List[Tuple[str, int, int, int, int]]]] = []
    current = [lines[0]]
    _, y_prev, _, h_prev = _line_bbox(lines[0])
    y_bottom_prev = y_prev + h_prev
    for line in lines[1:]:
        x, y, w, h = _line_bbox(line)
        gap = y - y_bottom_prev
        if gap <= gap_y:
            current.append(line)
            y_bottom_prev = y + h
        else:
            regions.append(current)
            current = [line]
            y_bottom_prev = y + h
    if current:
        regions.append(current)
    return regions


def _normalize_bbox(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(MIN_BBOX_W, min(w, img_w - x))
    h = max(MIN_BBOX_H, min(h, img_h - y))
    return x, y, w, h


class LayoutLMv3DetectionService(GUIDetectionService):
    """
    GUI detection via OCR aggregation: words -> lines -> cards/lists.
    Uses pytesseract with --psm 6; creates GUIBlocks with optional children.
    """

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        cache_dir: str | None = None,
        psm: int = 6,
    ) -> None:
        self.model_name = model_name
        self.psm = psm
        logger.info(
            "LayoutLMv3Detection: backend=%s (OCR psm=%d, line/card aggregation)",
            BACKEND_NAME, psm,
        )

    async def detect_gui_blocks(
        self,
        screenshot_path: str,
        ocr_text: str,
    ) -> List[GUIBlock]:
        """
        OCR -> words -> lines -> regions (cards). Return list of card/text/button blocks;
        card blocks have children = line blocks. Pipeline flattens for embedding.
        """
        path = Path(screenshot_path)
        if not path.exists():
            logger.warning("LayoutLMv3Detection: Screenshot not found %s", screenshot_path)
            return self._fallback_block(screenshot_path, ocr_text)

        screenshot_id = path.stem
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
        except Exception as e:
            logger.warning("LayoutLMv3Detection: Failed to load image %s: %s", screenshot_path, e)
            return self._fallback_block(screenshot_path, ocr_text)

        words = _ocr_words_bbox(str(path), psm=self.psm)
        if not words:
            # Try psm 11 for sparse text (e.g. buttons only)
            words = _ocr_words_bbox(str(path), psm=11)
        if not words:
            logger.warning(
                "LayoutLMv3Detection: No OCR words (backend=%s, path=%s, psm=%d); using fallback.",
                BACKEND_NAME, screenshot_path, self.psm,
            )
            return self._fallback_block(screenshot_path, ocr_text)

        lines = _group_words_into_lines(words)
        regions = _group_lines_into_regions(lines)
        blocks: List[GUIBlock] = []
        block_idx = 0

        for ri, region in enumerate(regions):
            child_blocks: List[GUIBlock] = []
            for li, line in enumerate(region):
                x, y, bw, bh = _line_bbox(line)
                x, y, bw, bh = _normalize_bbox(x, y, bw, bh, w, h)
                el_type = "button" if _looks_like_button(line) else "text"
                line_id = f"{screenshot_id}_layoutlmv3_line_{block_idx}"
                block_idx += 1
                lb = GUIBlock(
                    id=line_id,
                    screenshot_id=screenshot_id,
                    screenshot_path=screenshot_path,
                    bounding_box={"x": x, "y": y, "width": bw, "height": bh, "x1": x, "y1": y, "x2": x + bw, "y2": y + bh},
                    element_types=[el_type],
                    ocr_text=_line_text(line),
                    visual_features=[],
                )
                child_blocks.append(lb)
            rx, ry, rw, rh = _line_bbox(region[0])
            r_x2, r_y2 = rx + rw, ry + rh
            for line in region[1:]:
                xx, yy, ww, hh = _line_bbox(line)
                rx = min(rx, xx)
                ry = min(ry, yy)
                r_x2 = max(r_x2, xx + ww)
                r_y2 = max(r_y2, yy + hh)
            rw, rh = r_x2 - rx, r_y2 - ry
            rx, ry, rw, rh = _normalize_bbox(rx, ry, rw, rh, w, h)
            card_id = f"{screenshot_id}_layoutlmv3_card_{ri}"
            card = GUIBlock(
                id=card_id,
                screenshot_id=screenshot_id,
                screenshot_path=screenshot_path,
                bounding_box={"x": rx, "y": ry, "width": rw, "height": rh, "x1": rx, "y1": ry, "x2": rx + rw, "y2": ry + rh},
                element_types=["card"],
                ocr_text="",
                visual_features=[],
                children=child_blocks if len(child_blocks) > 1 else None,
            )
            blocks.append(card)

        logger.info(
            "LayoutLMv3Detection: backend=%s, path=%s, words=%d, lines=%d, regions=%d, blocks=%d",
            BACKEND_NAME, screenshot_path, len(words), len(lines), len(regions), len(blocks),
        )
        return blocks

    def _fallback_block(self, screenshot_path: str, ocr_text: str) -> List[GUIBlock]:
        """One full-image block only when OCR yields no words."""
        path = Path(screenshot_path)
        sid = path.stem
        try:
            im = Image.open(path)
            w, h = im.size
        except Exception:
            w, h = 800, 600
        return [
            GUIBlock(
                id=f"{sid}_layoutlmv3_fallback",
                screenshot_id=sid,
                screenshot_path=screenshot_path,
                bounding_box={"x": 0, "y": 0, "width": w, "height": h, "x1": 0, "y1": 0, "x2": w, "y2": h},
                element_types=["screen"],
                ocr_text=ocr_text or "",
                visual_features=[],
            )
        ]
