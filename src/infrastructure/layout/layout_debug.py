"""
Debug visualization for layout: regions (dashed), text_bbox (solid), words.

Region = context (dashed). Text bbox = tight from words (solid). Words = thin outline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Any

from .atoms import Region, Word

logger = logging.getLogger(__name__)

# RGB for PIL: тёмные тона, чтобы были видны на светлом фоне; подписи — с обводкой (белый по чёрному)
COLOR_TEXT_REGION = (0, 140, 0)       # dark green
COLOR_UI_REGION = (0, 80, 180)        # dark orange
COLOR_BACKGROUND = (80, 80, 80)       # dark gray
COLOR_WORD = (0, 0, 200)              # dark red
COLOR_TEXT_BBOX = (180, 80, 0)        # dark blue
COLOR_CONTAINER = (0, 120, 120)       # dark yellow/teal

DASH_LEN = 6
GAP_LEN = 4


def _dashed_rectangle(draw: Any, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, ...], width: int = 2) -> None:
    """Draw dashed rectangle (region/container)."""
    w = x2 - x1
    h = y2 - y1
    step = DASH_LEN + GAP_LEN
    # top
    x = x1
    while x < x2:
        draw.line([(x, y1), (min(x + DASH_LEN, x2), y1)], fill=color, width=width)
        x += step
    # bottom
    x = x1
    while x < x2:
        draw.line([(x, y2), (min(x + DASH_LEN, x2), y2)], fill=color, width=width)
        x += step
    # left
    y = y1
    while y < y2:
        draw.line([(x1, y), (x1, min(y + DASH_LEN, y2))], fill=color, width=width)
        y += step
    # right
    y = y1
    while y < y2:
        draw.line([(x2, y), (x2, min(y + DASH_LEN, y2))], fill=color, width=width)
        y += step


def render_layout_debug(
    image_path: str,
    regions: List[Region],
    region_assignments: Optional[List[Tuple[Region, List[Word]]]] = None,
    text_blocks: Optional[List[Any]] = None,
    gui_blocks: Optional[List[Any]] = None,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Draw regions (by type), optional words per region, and optional text_blocks on image.
    Saves to output_path or image_path parent / debug_regions_<stem>.png.
    Returns path to saved PNG or None.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("layout_debug: PIL not available")
        return None
    path = Path(image_path)
    if not path.exists():
        logger.warning("layout_debug: image not found %s", image_path)
        return None
    if output_path is None:
        output_path = str(path.parent / f"debug_regions_{path.stem}.png")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "C:\\Windows\\Fonts\\arial.ttf" if Path("C:\\Windows\\Fonts\\arial.ttf").exists()
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10
            )
        except Exception:
            font = ImageFont.load_default()

        from src.infrastructure.debug_draw import pil_text_visible

        # 1) Regions (dashed = context, not text)
        for i, r in enumerate(regions):
            if r.region_type == "text_region":
                color = COLOR_TEXT_REGION
            elif r.region_type == "ui_region":
                color = COLOR_UI_REGION
            else:
                color = COLOR_BACKGROUND
            _dashed_rectangle(draw, r.x, r.y, r.x + r.w, r.y + r.h, tuple(min(255, c) for c in color), width=2)
            label = f"{r.region_type[:4]}"
            pil_text_visible(draw, (r.x, max(0, r.y - 12)), label, font)

        # 2) Words (if region_assignments given)
        if region_assignments:
            for _region, words in region_assignments:
                for w in words:
                    draw.rectangle(
                        [w.x, w.y, w.x + w.w, w.y + w.h],
                        outline=tuple(min(255, c) for c in COLOR_WORD),
                        width=1,
                    )

        from src.infrastructure.debug_draw import pil_rectangle_visible

        # 3) TextBlocks (if given; solid = text bbox)
        if text_blocks:
            for tb in text_blocks:
                if hasattr(tb, "x") and hasattr(tb, "y") and hasattr(tb, "w") and hasattr(tb, "h"):
                    pil_rectangle_visible(draw, (tb.x, tb.y, tb.x + tb.w, tb.y + tb.h), COLOR_TEXT_BBOX, width=2)
        # 4) GUIBlocks: text_bbox = solid; container_bbox = dashed
        if gui_blocks:
            for blk in gui_blocks:
                bb = getattr(blk, "bounding_box", None) or {}
                x1 = int(bb.get("x1", bb.get("x", 0)))
                y1 = int(bb.get("y1", bb.get("y", 0)))
                x2 = int(bb.get("x2", x1 + bb.get("width", 0)))
                y2 = int(bb.get("y2", y1 + bb.get("height", 0)))
                pil_rectangle_visible(draw, (x1, y1, x2, y2), COLOR_TEXT_BBOX, width=2)
                etypes = getattr(blk, "element_types", None) or []
                label = etypes[0] if etypes else "block"
                pil_text_visible(draw, (x1, max(0, y1 - 10)), label, font)
                container = bb.get("container_bbox")
                if container:
                    cx1 = int(container.get("x1", container.get("x", 0)))
                    cy1 = int(container.get("y1", container.get("y", 0)))
                    cx2 = int(container.get("x2", cx1 + container.get("width", 0)))
                    cy2 = int(container.get("y2", cy1 + container.get("height", 0)))
                    _dashed_rectangle(draw, cx1, cy1, cx2, cy2, COLOR_CONTAINER, width=1)

        img.save(out, "PNG")
        logger.info(
            "layout_debug: saved regions=%d to %s",
            len(regions),
            out,
        )
        return str(out)
    except Exception as e:
        logger.warning("layout_debug: failed to save %s: %s", output_path, e)
        return None
