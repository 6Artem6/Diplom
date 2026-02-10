"""
Flow-based / implicit layout backend.

Goal: detect large implicit containers (messages/sections) on flow-based UIs (e.g. chat),
using OCR text lines + YOLO elements, without relying on VLMs.
Implements GUIDetectionService and returns GUIBlock-compatible output.

Architecture:
- Layout: OCR lines → vertical flows → implicit containers.
- Explicit elements: YOLO (buttons/inputs/images).
- Hierarchy: containers as parents, YOLO elements as children (bbox containment).
- OCR is used only as structural signal + text for containers, not as direct GUIBlock per line.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Языки OCR: английский + русский (можно переопределить через OCR_LANGS_TESSERACT)
OCR_LANGS_TESSERACT = os.environ.get("OCR_LANGS_TESSERACT", "eng+rus")

from PIL import Image

from src.domain.interfaces.gui_detection import GUIDetectionService
from src.domain.models.gui_block import GUIBlock
from src.infrastructure.layout import Word, Line
from src.infrastructure.layout.atoms import Region, TextBlock
from src.infrastructure.layout.cv_prepass import run_cv_prepass
from src.infrastructure.layout.region_prepass import cv_detect_regions, assign_words_to_region
from src.infrastructure.layout.layout_debug import render_layout_debug
from src.infrastructure.layout.line_builder import words_to_lines
from src.infrastructure.layout.line_classifier import classify_line, classify_line_with_reason
from src.infrastructure.layout.block_builder import lines_to_blocks_with_headers
from src.infrastructure.ocr.ocr_preprocess import preprocess_full_page, preprocess_crop, crop_bbox

logger = logging.getLogger(__name__)

# Heuristic thresholds for flow grouping
LINE_DY_PX_DEFAULT = 18  # fallback when no words to estimate
# OCR→Word: do NOT filter by bbox size, color, or text length. If Tesseract returned bbox+text → Word always.
# Only skip invalid bbox (w<=0 or h<=0). Thin/light text must not be dropped.
MIN_BBOX_WIDTH = 1
MIN_BBOX_HEIGHT = 1
HORIZ_OVERLAP_RATIO = 0.4  # min horizontal overlap between consecutive lines
MAX_WIDTH_CHANGE_RATIO = 2.0  # width_j / width_i in [1/MAX, MAX]
# Gap between lines: ~1.2–1.5 × line height → new block (smaller = less over-merge, e.g. form fields)
BLOCK_GAP_LINE_HEIGHT_FACTOR = 1.4

# Region-based layout: block ≥ this fraction of screen → INVALID (do not emit)
BLOCK_MAX_SCREEN_AREA_RATIO = 0.8
# Badge vs button by semantics: badge = h_region/h_text ≤ 1.4, short text. NOT "text ≈ container".
UI_BADGE_H_REGION_TEXT_RATIO = 1.4
UI_BADGE_MAX_CHARS = 15
# Button: region must not be huge vs text (else container)
UI_MAX_BUTTON_AREA_MULT = 10

# Sanitizer thresholds
MIN_AREA_RATIO = 0.0005  # blocks smaller этой доли от экрана считаются шумом (если ещё и пустые)
MIN_BLOCK_WIDTH = 20
MIN_BLOCK_HEIGHT = 12
IOU_DEDUP_THRESHOLD = 0.7
OVERLAP_SAME_TYPE_THRESHOLD = 0.8

INTERACTIVE_TYPES = {
    "button",
    "input",
    "checkbox",
    "dropdown",
    "list-item",
    "card",
    "image",
}


def _ocr_words_bbox(image_path: str, psm: int = 11) -> List[Tuple[str, int, int, int, int, Optional[float]]]:
    """First OCR pass on preprocessed full page (grayscale, CLAHE, Otsu). No scaling."""
    logger.info("FlowLayout: OCR start for %s (psm=%d) with full-page preprocess", image_path, psm)
    try:
        import pytesseract
        import numpy as np
    except ImportError as e:  # pragma: no cover
        logger.warning("FlowLayout: pytesseract/numpy not available: %s", e)
        return []
    path = Path(image_path)
    if not path.exists():
        logger.warning("FlowLayout: image not found for OCR %s", image_path)
        return []
    try:
        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)
    except Exception as e:  # pragma: no cover
        logger.warning("FlowLayout: failed to load image %s: %s", image_path, e)
        return []
    img_prep = preprocess_full_page(img_np)
    img_prep_pil = Image.fromarray(img_prep)
    try:
        data = pytesseract.image_to_data(
            img_prep_pil, lang=OCR_LANGS_TESSERACT, output_type=pytesseract.Output.DICT, config=f"--psm {psm}"
        )
    except Exception as e:  # pragma: no cover
        logger.warning("FlowLayout: pytesseract.image_to_data failed for %s: %s", image_path, e)
        return []

    texts = data.get("text", []) or []
    n_non_empty = sum(1 for t in texts if (t or "").strip())
    logger.info("FlowLayout: OCR (preprocessed) for %s n_non_empty=%d (psm=%d)", image_path, n_non_empty, psm)

    out: List[Tuple[str, int, int, int, int, Optional[float]]] = []
    conf_list = data.get("conf", [])
    for i, t in enumerate(texts):
        t = (t or "").strip()
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        if w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT:
            if t:
                logger.debug("FlowLayout: OCR skipped invalid bbox %r w=%d h=%d", t, w, h)
            continue
        conf_val = conf_list[i] if conf_list and i < len(conf_list) else None
        conf: Optional[float] = None
        if conf_val is not None and conf_val != -1 and conf_val != "-1":
            try:
                conf = float(conf_val) / 100.0
            except (ValueError, TypeError):
                pass
        out.append((t, x, y, w, h, conf))
    n_empty = sum(1 for (t, *_) in out if not t)
    logger.info("FlowLayout: OCR for %s words=%d (empty_bbox=%d)", image_path, len(out), n_empty)
    return out


# Short button-like words: never drop if len≤4, light/gray, has_background, aspect≥1.2
SHORT_BUTTON_MAX_LEN = 4
SHORT_BUTTON_ASPECT_MIN = 1.2


def _is_short_button_like(w: Word) -> bool:
    """Keep such words even with low conf or empty after fallback."""
    text = (w.text or "").strip()
    if len(text) > SHORT_BUTTON_MAX_LEN:
        return False
    if getattr(w, "text_color_class", "dark") not in ("light", "gray"):
        return False
    if not getattr(w, "has_background", False):
        return False
    if w.h <= 0:
        return False
    aspect = w.w / float(w.h)
    return aspect >= SHORT_BUTTON_ASPECT_MIN


def _ocr_fallback_light_words(image_path: str, words: List[Word]) -> List[Word]:
    """Retry OCR with aggressive preprocess for empty/short light/gray words on primary/secondary.
    Layout bbox (x,y,w,h) never changes; ocr_bbox set when fallback used.
    """
    try:
        import pytesseract
        import numpy as np
    except ImportError:
        return words
    path = Path(image_path)
    if not path.exists():
        return words
    try:
        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)
    except Exception:
        return words
    heights = [w.h for w in words if w.h and w.h > 0]
    median_body_h = float(sorted(heights)[len(heights) // 2]) if heights else 24.0
    out: List[Word] = []
    for w in words:
        text = (w.text or "").strip()
        need_fallback = (
            (not text or len(text) <= SHORT_BUTTON_MAX_LEN)
            and getattr(w, "text_color_class", "dark") in ("light", "gray")
            and (getattr(w, "has_background", False) or (w.h > 0 and (w.w / float(w.h)) >= SHORT_BUTTON_ASPECT_MIN))
        )
        if not need_fallback:
            out.append(w)
            continue
        crop, _, _ = crop_bbox(img_np, w.x, w.y, w.w, w.h, padding=2)
        if crop.size == 0:
            out.append(w)
            continue
        res = preprocess_crop(crop, dilation_px=2, upscale_factor=2.0, invert_if_dark=True)
        try:
            crop_pil = Image.fromarray(res.image)
            fallback_text = (pytesseract.image_to_string(crop_pil, lang=OCR_LANGS_TESSERACT, config="--psm 7") or "").strip()
        except Exception:
            fallback_text = ""
        new_text = fallback_text.replace("\n", " ").strip() if fallback_text else ""
        layout_bbox = (w.x, w.y, w.w, w.h)
        if new_text:
            out.append(
                Word(
                    text=new_text,
                    x=w.x,
                    y=w.y,
                    w=w.w,
                    h=w.h,
                    conf=w.conf,
                    has_background=w.has_background,
                    bg_color_cluster=w.bg_color_cluster,
                    text_color_class=w.text_color_class,
                    font_weight=getattr(w, "font_weight", None),
                    estimated_font_size_px=getattr(w, "estimated_font_size_px", None) or float(w.h),
                    ocr_fallback_dilation=res.applied_dilation,
                    ocr_fallback_inversion=res.applied_inversion,
                    ocr_fallback_upscale=res.upscale_factor,
                    ocr_bbox=layout_bbox,
                )
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "FlowLayout: fallback OCR at (%s,%s) dilation=%s inv=%s upscale=%s len=%s",
                    w.x, w.y, res.applied_dilation, res.applied_inversion, res.upscale_factor, len(new_text),
                )
            continue
        if _is_short_button_like(w):
            out.append(
                Word(
                    text=w.text,
                    x=w.x,
                    y=w.y,
                    w=w.w,
                    h=w.h,
                    conf=w.conf,
                    has_background=w.has_background,
                    bg_color_cluster=w.bg_color_cluster,
                    text_color_class=w.text_color_class,
                    font_weight=getattr(w, "font_weight", None),
                    estimated_font_size_px=getattr(w, "estimated_font_size_px", None) or float(w.h),
                    ocr_bbox=layout_bbox,
                )
            )
            continue
        out.append(w)
    return out


def _group_words_into_lines(
    words: List[Tuple[str, int, int, int, int]], line_dy: int = LINE_DY_PX_DEFAULT
) -> List[List[Tuple[str, int, int, int, int]]]:
    """Group words by similar y (same visual line)."""
    if not words:
        return []
    # sort by center-y then x
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
    """Union bbox of all words in line → (x, y, w, h)."""
    x1 = min(w[1] for w in line)
    y1 = min(w[2] for w in line)
    x2 = max(w[1] + w[3] for w in line)
    y2 = max(w[2] + w[4] for w in line)
    return x1, y1, x2 - x1, y2 - y1


def _estimate_line_height(words: List[Word]) -> float:
    """Estimate typical line height = median(font_size_px); font_size_px from word, not raw bbox h."""
    if not words:
        return float(LINE_DY_PX_DEFAULT)
    heights = [getattr(w, "estimated_font_size_px", None) or float(w.h) for w in words]
    heights.sort()
    n = len(heights)
    return float(heights[n // 2]) if n else float(LINE_DY_PX_DEFAULT)


def _estimate_char_width(words: List[Word]) -> float:
    """Estimate average character width (px per char) for horizontal continuity. Used for width merge."""
    if not words:
        return 8.0
    per_char: List[float] = []
    for w in words:
        n = max(1, len(w.text))
        per_char.append(w.w / n)
    per_char.sort()
    n = len(per_char)
    return per_char[n // 2] if n else 8.0


def _text_blocks_to_gui_blocks(
    text_blocks: List[TextBlock],
    screenshot_id: str,
    screenshot_path: str,
    image_area: int,
    block_start_index: int,
    container_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> List[GUIBlock]:
    """
    Map TextBlocks to GUIBlock. bounding_box = text_bbox (tight from words).
    If container_bbox (x,y,w,h) is given, add it to bounding_box dict as container_bbox.
    Skip blocks with area >= BLOCK_MAX_SCREEN_AREA_RATIO * image_area.
    """
    gui_blocks: List[GUIBlock] = []
    for i, tb in enumerate(text_blocks):
        area = tb.w * tb.h
        if image_area > 0 and area >= BLOCK_MAX_SCREEN_AREA_RATIO * image_area:
            logger.warning(
                "FlowLayout: invalid block (area %.0f >= %.0f%% screen) skipped bbox=(%d,%d,%d,%d)",
                area, BLOCK_MAX_SCREEN_AREA_RATIO * 100, tb.x, tb.y, tb.w, tb.h,
            )
            continue
        text = "\n".join(ln.text for ln in tb.lines)
        etypes: List[str] = ["text_block"]
        if tb.block_type == "header":
            etypes = ["header", "text_block"]
        elif any(getattr(ln, "role", None) == "button" for ln in tb.lines):
            etypes = ["button", "text_block"]
        bbox: Dict[str, Any] = {
            "x": tb.x,
            "y": tb.y,
            "width": tb.w,
            "height": tb.h,
            "x1": tb.x,
            "y1": tb.y,
            "x2": tb.x + tb.w,
            "y2": tb.y + tb.h,
        }
        if container_bbox is not None:
            cx, cy, cw, ch = container_bbox
            bbox["container_bbox"] = {
                "x": cx, "y": cy, "width": cw, "height": ch,
                "x1": cx, "y1": cy, "x2": cx + cw, "y2": cy + ch,
            }
        gui_blocks.append(
            GUIBlock(
                id=f"{screenshot_id}_tb_{block_start_index + i}",
                screenshot_id=screenshot_id,
                screenshot_path=screenshot_path,
                bounding_box=bbox,
                element_types=etypes,
                ocr_text=text,
                visual_features=[],
            )
        )
    return gui_blocks


def _layout_inside_region(
    region: Region,
    region_words: List[Word],
    line_dy_px: int,
    char_width_px: float,
    horizontal_rules: List[Any],
    vertical_rules: List[Any],
    screenshot_id: str,
    screenshot_path: str,
    image_area: int,
    block_index_offset: int,
) -> List[GUIBlock]:
    """
    Run words→lines→blocks inside one text_region. Region = context only.
    lines_to_blocks_with_headers: header/body/button do NOT merge. No merge_blocks_inside_region.
    TextBlock bbox = text (tight); container_bbox = region. Block bbox clipped to region.
    """
    if not region_words:
        return []
    lines = words_to_lines(
        region_words,
        line_dy_px=line_dy_px,
        char_width_px=char_width_px,
        horizontal_rules=horizontal_rules,
        vertical_rules=vertical_rules,
    )
    lines_with_roles: List[Line] = []
    for ln in lines:
        role = classify_line(ln, lines)
        lines_with_roles.append(
            Line(
                words=ln.words,
                x=ln.x,
                y=ln.y,
                w=ln.w,
                h=ln.h,
                is_header=(role == "header"),
                role=role,
                estimated_font_size_px=getattr(ln, "estimated_font_size_px", None),
            )
        )
    text_blocks = lines_to_blocks_with_headers(
        lines_with_roles,
        char_width_px=char_width_px,
        dividers=horizontal_rules,
    )
    # Clip each block to region (no block extends beyond region)
    rx, ry, rw, rh = region.x, region.y, region.w, region.h
    clipped: List[TextBlock] = []
    for tb in text_blocks:
        x1 = max(tb.x, rx)
        y1 = max(tb.y, ry)
        x2 = min(tb.x + tb.w, rx + rw)
        y2 = min(tb.y + tb.h, ry + rh)
        if x2 <= x1 or y2 <= y1:
            continue
        clipped.append(
            TextBlock(
                lines=tb.lines,
                x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                block_type=tb.block_type,
                stats=tb.stats,
            )
        )
    return _text_blocks_to_gui_blocks(
        clipped,
        screenshot_id,
        screenshot_path,
        image_area,
        block_index_offset,
        container_bbox=(rx, ry, rw, rh),
    )


class FlowLayoutDetectionService(GUIDetectionService):
    """
    Flow-based backend (step 0–1):
    WORD → LINE → TEXT_BLOCK → GUIBlock(text_block)

    No YOLO, no regions, no roles, no patterns.
    """

    def __init__(self, *_, **__) -> None:
        # yolo_detector may be passed from DI, but is intentionally unused at this stage.
        pass

    async def detect_gui_blocks(
        self, screenshot_path: str, ocr_text: str
    ) -> List[GUIBlock]:
        path = Path(screenshot_path)
        screenshot_id = path.stem

        # 1) OCR → Words (psm 11 = sparse text, better vertical separation for lists/cards)
        raw_words = _ocr_words_bbox(screenshot_path, psm=11)
        words: List[Word] = [
            Word(
                text=t,
                x=x,
                y=y,
                w=w,
                h=h,
                conf=conf,
                estimated_font_size_px=float(h),
            )
            for (t, x, y, w, h, conf) in raw_words
        ]

        # 2) CV prepass: visual context (background, text color, separators)
        words, horizontal_rules, vertical_rules = run_cv_prepass(screenshot_path, words)

        # 2b) Fallback OCR for empty/short light/gray text (dilation + upscale per bbox)
        words = _ocr_fallback_light_words(screenshot_path, words)

        # Image size for region fallback and "whole screen" invalid block rule
        try:
            with Image.open(path) as im:
                img_w, img_h = im.size
        except Exception:
            img_w = max((w.x + w.w for w in words), default=800)
            img_h = max((w.y + w.h for w in words), default=600)
        image_area = max(1, img_w * img_h)

        # 3) Region prepass: CV regions before OCR grouping; layout only inside each region
        regions = cv_detect_regions(screenshot_path)
        if not regions:
            # No regions: words → lines → blocks (header/body/button separated). No region merge.
            line_height_px = _estimate_line_height(words)
            line_dy_px = max(10, int(line_height_px * 0.8))
            char_width_px = _estimate_char_width(words)
            logger.debug(
                "FlowLayout: no regions, pipeline line_height=%.0f line_dy_px=%d char_width=%.1f for %s",
                line_height_px, line_dy_px, char_width_px, screenshot_path,
            )
            lines = words_to_lines(
                words,
                line_dy_px=line_dy_px,
                char_width_px=char_width_px,
                horizontal_rules=horizontal_rules,
                vertical_rules=vertical_rules,
            )
            lines_with_roles = []
            for idx, ln in enumerate(lines):
                role = classify_line(ln, lines)
                lines_with_roles.append(
                    Line(
                        words=ln.words,
                        x=ln.x, y=ln.y, w=ln.w, h=ln.h,
                        is_header=(role == "header"),
                        role=role,
                        estimated_font_size_px=getattr(ln, "estimated_font_size_px", None),
                    )
                )
                if logger.isEnabledFor(logging.DEBUG):
                    _, reason = classify_line_with_reason(ln, lines, page_width=None)
                    short_info = [
                        (w.text or "?", getattr(w, "conf", None), getattr(w, "ocr_fallback_dilation", False), getattr(w, "text_color_class", "?"))
                        for w in ln.words if len((w.text or "").strip()) <= SHORT_BUTTON_MAX_LEN
                    ]
                    logger.debug(
                        "Line(id=%s): words=[%s] role=%s reason=%s short_words=%s",
                        idx, ",".join((w.text or "").strip() or "?" for w in ln.words), role, reason, short_info,
                    )
            text_blocks = lines_to_blocks_with_headers(
                lines_with_roles, char_width_px=char_width_px, dividers=horizontal_rules,
            )
            return _text_blocks_to_gui_blocks(
                text_blocks, screenshot_id, screenshot_path, image_area, 0, container_bbox=None,
            )

        # Regions: assign words to region; layout per region; ui_region → one button/badge block
        region_assignments = assign_words_to_region(words, regions)
        for r, rwords in region_assignments:
            logger.info(
                "FlowLayout: region bbox=(%d,%d,%d,%d) type=%s area=%d words=%d",
                r.x, r.y, r.w, r.h, r.region_type, r.area, len(rwords),
            )
        line_height_px = _estimate_line_height(words)
        line_dy_px = max(10, int(line_height_px * 0.8))
        char_width_px = _estimate_char_width(words)
        gui_blocks = []
        block_index = 0
        for region, region_words in region_assignments:
            if region.region_type == "background":
                continue
            if region.region_type == "ui_region":
                # Button/badge: text_bbox = tight from words (or region if no words); container_bbox = region.
                # Badge vs button by semantics: h_region/h_text ≤ 1.4 and short text → badge.
                # Region too large vs text → section (not button).
                text = " ".join((w.text or "").strip() for w in region_words if (w.text or "").strip()).strip()
                if region_words:
                    tx1 = min(w.x for w in region_words)
                    ty1 = min(w.y for w in region_words)
                    tx2 = max(w.x + w.w for w in region_words)
                    ty2 = max(w.y + w.h for w in region_words)
                    pad = 2
                    tx1 = max(region.x, tx1 - pad)
                    ty1 = max(region.y, ty1 - pad)
                    tx2 = min(region.x + region.w, tx2 + pad)
                    ty2 = min(region.y + region.h, ty2 + pad)
                    text_h = ty2 - ty1
                    text_w = tx2 - tx1
                    text_area = max(1, text_w * text_h)
                    # Huge region → section, not button
                    if region.area > text_area * UI_MAX_BUTTON_AREA_MULT:
                        etypes = ["section", "text_block"]
                    else:
                        is_badge = (
                            text_h > 0
                            and region.h / text_h <= UI_BADGE_H_REGION_TEXT_RATIO
                            and len(text) <= UI_BADGE_MAX_CHARS
                        )
                        etypes = ["badge", "text_block"] if is_badge else ["button", "text_block"]
                else:
                    tx1, ty1 = region.x, region.y
                    tx2, ty2 = region.x + region.w, region.y + region.h
                    etypes = ["button", "text_block"]
                bbox: Dict[str, Any] = {
                    "x": tx1, "y": ty1,
                    "width": tx2 - tx1, "height": ty2 - ty1,
                    "x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2,
                }
                bbox["container_bbox"] = {
                    "x": region.x, "y": region.y,
                    "width": region.w, "height": region.h,
                    "x1": region.x, "y1": region.y,
                    "x2": region.x + region.w, "y2": region.y + region.h,
                }
                gui_blocks.append(
                    GUIBlock(
                        id=f"{screenshot_id}_ui_{block_index}",
                        screenshot_id=screenshot_id,
                        screenshot_path=screenshot_path,
                        bounding_box=bbox,
                        element_types=etypes,
                        ocr_text=text or "",
                        visual_features=[],
                    )
                )
                block_index += 1
                logger.debug(
                    "FlowLayout: ui_region → %s text_bbox=(%d,%d,%d,%d) container=(%d,%d,%d,%d) text=%r",
                    etypes[0],
                    tx1, ty1, tx2 - tx1, ty2 - ty1,
                    region.x, region.y, region.w, region.h,
                    text or "(none)",
                )
                continue
            # text_region (or fallback page): words → lines → blocks inside region
            region_blocks = _layout_inside_region(
                region,
                region_words,
                line_dy_px,
                char_width_px,
                horizontal_rules,
                vertical_rules,
                screenshot_id,
                screenshot_path,
                image_area,
                block_index,
            )
            gui_blocks.extend(region_blocks)
            block_index += len(region_blocks)
        # Debug: draw regions + words + final gui_blocks
        debug_out = render_layout_debug(
            screenshot_path,
            regions,
            region_assignments=region_assignments,
            gui_blocks=gui_blocks,
            output_path=str(path.parent / f"debug_regions_{path.stem}.png"),
        )
        if debug_out and logger.isEnabledFor(logging.DEBUG):
            logger.debug("FlowLayout: region debug image -> %s", debug_out)
        logger.info(
            "FlowLayout: regions=%d words=%d gui_blocks=%d for %s",
            len(regions), len(words), len(gui_blocks), screenshot_path,
        )
        return gui_blocks


def _horizontal_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Horizontal overlap length / min(width_a, width_b)."""
    ax1, ax2 = a["x"], a["x"] + a["w"]
    bx1, bx2 = b["x"], b["x"] + b["w"]
    ix1 = max(ax1, bx1)
    ix2 = min(ax2, bx2)
    if ix2 <= ix1:
        return 0.0
    overlap = ix2 - ix1
    return overlap / max(1, min(a["w"], b["w"]))


def _width_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """max(width_a, width_b) / min(width_a, width_b)."""
    wa, wb = a["w"], b["w"]
    mn = min(wa, wb)
    mx = max(wa, wb)
    if mn <= 0:
        return MAX_WIDTH_CHANGE_RATIO + 1.0
    return mx / mn


def _build_hierarchy(
    containers: List[GUIBlock], explicit: List[GUIBlock]
) -> List[GUIBlock]:
    """Assign explicit blocks as children of nearest containing container.

    - Containers: implicit flow blocks (section/text_block).
    - Explicit: YOLO elements (button, input, image, etc.).
    - Children are attached via bbox containment; roots are containers.
    """
    roots = containers.copy()

    for blk in explicit:
        parent = _find_smallest_container(blk, containers)
        if parent is None:
            # No container; treat as separate root
            roots.append(blk)
        else:
            children = list(parent.children or [])
            children.append(blk)
            parent.children = children

    return roots


def _find_smallest_container(
    child: GUIBlock, containers: List[GUIBlock]
) -> GUIBlock | None:
    """Return smallest container whose bbox fully contains child bbox."""
    cb = child.bounding_box
    cx1, cy1, cx2, cy2 = cb["x1"], cb["y1"], cb["x2"], cb["y2"]
    candidates: List[Tuple[float, GUIBlock]] = []
    for cont in containers:
        b = cont.bounding_box
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        if x1 <= cx1 and y1 <= cy1 and x2 >= cx2 and y2 >= cy2:
            area = (x2 - x1) * (y2 - y1)
            candidates.append((area, cont))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def _sanitize_layout(roots: List[GUIBlock], img_w: int, img_h: int) -> List[GUIBlock]:
    """Remove obviously noisy blocks and merge near-duplicates on the same level.

    Rules (approximate implementation of LayoutSanitizer spec):
    - Удаляем блок, если выполняются ≥2 условий:
      * площадь сильно мала относительно экрана,
      * пустой текст,
      * нет детей,
      * экстремальные пропорции (очень узкий/низкий),
      * сильно перекрывается с более крупным блоком того же типа.
    - Не возвращаем блоки без текста, детей или интерактивных element_types.
    - На одном уровне выполняем merge для пар с большим IoU и совместимыми типами.
    """

    image_area = max(1, img_w * img_h)

    def is_interactive(block: GUIBlock) -> bool:
        return any(t in INTERACTIVE_TYPES for t in block.element_types)

    def block_area(b: GUIBlock) -> int:
        bb = b.bounding_box
        return max(0, (bb["x2"] - bb["x1"]) * (bb["y2"] - bb["y1"]))

    def iou(a: GUIBlock, b: GUIBlock) -> float:
        ba, bb = a.bounding_box, b.bounding_box
        ax1, ay1, ax2, ay2 = ba["x1"], ba["y1"], ba["x2"], ba["y2"]
        bx1, by1, bx2, by2 = bb["x1"], bb["y1"], bb["x2"], bb["y2"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        ua = block_area(a)
        ub = block_area(b)
        union = ua + ub - inter
        return inter / union if union > 0 else 0.0

    def same_type(a: GUIBlock, b: GUIBlock) -> bool:
        return bool(a.element_types and b.element_types and a.element_types[0] == b.element_types[0])

    def compatible_types(a: GUIBlock, b: GUIBlock) -> bool:
        t1 = a.element_types[0] if a.element_types else ""
        t2 = b.element_types[0] if b.element_types else ""
        pair = {t1, t2}
        # button + text, input + text, card + text_block, etc.
        compatible_sets = [
            {"button", "text"},
            {"input", "text"},
            {"card", "text_block"},
            {"card", "text"},
        ]
        if pair in compatible_sets:
            return True
        # identical type: allow merge if bbox почти совпадает
        return t1 == t2

    def merge_blocks(a: GUIBlock, b: GUIBlock) -> GUIBlock:
        """Union bbox, choose dominant type, merge text и children."""
        ba, bb = a.bounding_box, b.bounding_box
        x1 = min(ba["x1"], bb["x1"])
        y1 = min(ba["y1"], bb["y1"])
        x2 = max(ba["x2"], bb["x2"])
        y2 = max(ba["y2"], bb["y2"])
        w = x2 - x1
        h = y2 - y1

        def dominant_type(types: List[str]) -> str:
            order = ["button", "input", "card", "list-item", "image", "text_block", "section", "text"]
            for t in order:
                if t in types:
                    return t
            return types[0] if types else "text_block"

        types = list({*(a.element_types or []), *(b.element_types or [])})
        etype = dominant_type(types)

        texts = []
        for t in [a.ocr_text, b.ocr_text]:
            t_norm = (t or "").strip()
            if t_norm and t_norm not in texts:
                texts.append(t_norm)
        merged_text = "\n".join(texts)

        children = []
        if a.children:
            children.extend(a.children)
        if b.children:
            children.extend(b.children)

        return GUIBlock(
            id=a.id,  # keep one id; upstream uses ids only as local identifiers
            screenshot_id=a.screenshot_id,
            screenshot_path=a.screenshot_path,
            bounding_box={
                "x": x1,
                "y": y1,
                "width": w,
                "height": h,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            element_types=[etype],
            ocr_text=merged_text,
            visual_features=[],
            children=children or None,
        )

    def prune_and_dedup(level_blocks: List[GUIBlock]) -> List[GUIBlock]:
        # 1) рекурсивно обработать детей
        for b in level_blocks:
            if b.children:
                b.children = prune_and_dedup(list(b.children)) or None

        # 2) первичное удаление шумовых блоков
        kept: List[GUIBlock] = []
        for b in level_blocks:
            bb = b.bounding_box
            w = bb["x2"] - bb["x1"]
            h = bb["y2"] - bb["y1"]
            area = block_area(b)
            area_ratio = area / image_area
            empty_text = not (b.ocr_text or "").strip()
            no_children = not b.children
            tiny = area_ratio < MIN_AREA_RATIO
            bad_shape = w < MIN_BLOCK_WIDTH or h < MIN_BLOCK_HEIGHT
            # не интерактивный и без детей/текста
            non_interactive = not is_interactive(b)

            # считаем кол-во "плохих" признаков
            bad_flags = [
                tiny,
                empty_text,
                no_children,
                bad_shape,
            ]
            bad_count = sum(1 for f in bad_flags if f)

            # Жёсткое правило: не возвращаем блоки без текста, детей и интерактива
            if empty_text and no_children and not is_interactive(b):
                continue

            # Удаляем блок, если ≥2 плохих признаков и он не интерактивный
            if bad_count >= 2 and non_interactive:
                continue

            kept.append(b)

        # 3) удаление сильно перекрывающихся блоков одинакового типа
        pruned: List[GUIBlock] = []
        for b in kept:
            drop = False
            for other in kept:
                if other is b:
                    continue
                if same_type(b, other) and block_area(other) > block_area(b):
                    if iou(b, other) >= OVERLAP_SAME_TYPE_THRESHOLD:
                        drop = True
                        break
            if not drop:
                pruned.append(b)

        # 4) merge-проход: объединяем пары с большим IoU и совместимыми типами
        merged: List[GUIBlock] = []
        for b in pruned:
            merged_into_existing = False
            for i, m in enumerate(merged):
                if compatible_types(b, m) and iou(b, m) >= IOU_DEDUP_THRESHOLD:
                    merged[i] = merge_blocks(m, b)
                    merged_into_existing = True
                    break
            if not merged_into_existing:
                merged.append(b)

        return merged

    sanitized = prune_and_dedup(roots)
    # Простая диагностика пересечений на верхнем уровне
    intersections = 0
    for i, a in enumerate(sanitized):
        for b in sanitized[i + 1 :]:
            if iou(a, b) > 0.0:
                intersections += 1
    if intersections:
        logger.info("FlowLayout: sanitizer detected %d intersections at root level", intersections)
    return sanitized


def _log_group_signatures(roots: List[GUIBlock], img_w: int, img_h: int) -> None:
    """Extract coarse signatures for large containers to enable pattern mining later.

    Signature features (per container):
    - normalized_bbox_ratio (area / image_area)
    - children_types_sequence (e.g., ['text', 'text', 'button'])
    - text_density (len(text) / area)
    - has_interaction (any child interactive)
    """

    image_area = max(1, img_w * img_h)

    def is_large_container(b: GUIBlock) -> bool:
        et = (b.element_types or [""])[0]
        return et in {"section", "text_block", "card", "list-item", "message"}

    def walk(blocks: List[GUIBlock]) -> List[GUIBlock]:
        acc: List[GUIBlock] = []
        for b in blocks:
            if is_large_container(b):
                acc.append(b)
            if b.children:
                acc.extend(walk(list(b.children)))
        return acc

    containers = walk(roots)
    if not containers:
        return

    signatures: Dict[Tuple, int] = {}
    for b in containers:
        bb = b.bounding_box
        area = max(1, (bb["x2"] - bb["x1"]) * (bb["y2"] - bb["y1"]))
        norm_ratio = area / image_area
        text = (b.ocr_text or "").strip()
        text_density = len(text) / area
        child_types = []
        has_interaction = False
        for ch in (b.children or []):
            et = (ch.element_types or [""])[0]
            child_types.append(et)
            if et in INTERACTIVE_TYPES:
                has_interaction = True
        key = (
            round(norm_ratio, 3),
            tuple(child_types),
            round(text_density, 3),
            has_interaction,
        )
        signatures[key] = signatures.get(key, 0) + 1

    # Логируем только повторяющиеся паттерны (count > 1)
    repeated = {k: c for k, c in signatures.items() if c > 1}
    if repeated:
        logger.info("FlowLayout: repeated container patterns (pattern_id -> count):")
        for idx, (sig, count) in enumerate(repeated.items(), start=1):
            norm_ratio, child_seq, density, has_inter = sig
            logger.info(
                "  pattern_%d: norm_ratio=%.3f, children=%s, text_density=%.3f, has_interaction=%s, count=%d",
                idx,
                norm_ratio,
                list(child_seq),
                density,
                has_inter,
                count,
            )

