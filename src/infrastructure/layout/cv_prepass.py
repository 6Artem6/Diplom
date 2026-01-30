"""
Minimal CV prepass before words→lines: visual context (background, text color, separators).

Runs once per screenshot; enriches Word with has_background, bg_color_cluster, text_color_class
and returns HorizontalRule / VerticalRule for layout to enforce hard boundaries.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .atoms import HorizontalRule, VerticalRule, Word
from .atoms import TextColorClass

logger = logging.getLogger(__name__)

# Background: border variance below this → uniform background (button/pill)
BORDER_VARIANCE_THRESHOLD = 400.0  # grayscale 0–255² scale
PADDING_PX = 2
# Luminance buckets for text/bg: dark < 85, gray 85–170, light > 170
LUM_DARK = 85
LUM_LIGHT = 170
# Separator: min width/height ratio of image to count as rule; min length
MIN_LINE_LENGTH_RATIO = 0.15
MIN_LINE_PX = 20
# Row/col considered "line" if variance below this and mean in mid range
SEP_ROW_VAR_THRESHOLD = 300.0
SEP_LUM_MIN, SEP_LUM_MAX = 40, 220


def _luminance(rgb: np.ndarray) -> float:
    """Rec. 601 grayscale."""
    if rgb.ndim == 2:
        return float(np.mean(rgb))
    return float(0.299 * rgb[..., 0].mean() + 0.587 * rgb[..., 1].mean() + 0.114 * rgb[..., 2].mean())


def _luminance_class(lum: float) -> TextColorClass:
    if lum < LUM_DARK:
        return "dark"
    if lum > LUM_LIGHT:
        return "light"
    return "gray"


def _enrich_word(img: np.ndarray, w: Word, img_w: int, img_h: int) -> Word:
    """Set has_background, bg_color_cluster, text_color_class from image crop."""
    x1 = max(0, w.x - PADDING_PX)
    y1 = max(0, w.y - PADDING_PX)
    x2 = min(img_w, w.x + w.w + PADDING_PX)
    y2 = min(img_h, w.y + w.h + PADDING_PX)
    if x2 <= x1 or y2 <= y1:
        return w
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return w
    # Border: outer 1-pixel ring (top/bottom/left/right edges)
    h_c, w_c = crop.shape[:2]
    if h_c < 3 or w_c < 3:
        return w
    top = crop[0, :].reshape(-1, crop.shape[2] if crop.ndim == 3 else 1)
    bottom = crop[-1, :].reshape(-1, crop.shape[2] if crop.ndim == 3 else 1)
    left = crop[1:-1, 0].reshape(-1, crop.shape[2] if crop.ndim == 3 else 1)
    right = crop[1:-1, -1].reshape(-1, crop.shape[2] if crop.ndim == 3 else 1)
    border = np.vstack([top, bottom, left, right])
    if crop.ndim == 3:
        border_lum = 0.299 * border[:, 0] + 0.587 * border[:, 1] + 0.114 * border[:, 2]
    else:
        border_lum = border.flatten()
    border_var = float(np.var(border_lum))
    border_mean = float(np.mean(border_lum))
    has_bg = border_var < BORDER_VARIANCE_THRESHOLD
    # Cluster: 0 dark, 1 mid, 2 light
    if border_mean < LUM_DARK:
        bg_cluster = 0
    elif border_mean > LUM_LIGHT:
        bg_cluster = 2
    else:
        bg_cluster = 1
    # Text color: inner region (shrink 1px) vs border
    inner = crop[1 : h_c - 1, 1 : w_c - 1]
    if inner.size == 0:
        text_class: TextColorClass = "dark"
    else:
        if inner.ndim == 3:
            inner_lum = 0.299 * inner[..., 0] + 0.587 * inner[..., 1] + 0.114 * inner[..., 2]
        else:
            inner_lum = inner
        inner_mean = float(np.mean(inner_lum))
        text_class = _luminance_class(inner_mean)
    return Word(
        text=w.text,
        x=w.x,
        y=w.y,
        w=w.w,
        h=w.h,
        conf=w.conf,
        has_background=has_bg,
        bg_color_cluster=bg_cluster if has_bg else None,
        text_color_class=text_class,
        font_weight=getattr(w, "font_weight", None),
        estimated_font_size_px=getattr(w, "estimated_font_size_px", None),
        ocr_fallback_dilation=getattr(w, "ocr_fallback_dilation", False),
        ocr_fallback_inversion=getattr(w, "ocr_fallback_inversion", False),
        ocr_fallback_upscale=getattr(w, "ocr_fallback_upscale", 1.0),
        ocr_bbox=getattr(w, "ocr_bbox", None),
    )


def _detect_horizontal_rules(img: np.ndarray, img_w: int, img_h: int) -> List[HorizontalRule]:
    """Thin horizontal lines: low row variance, spanning a fraction of width."""
    if img.ndim == 3:
        gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float64)
    else:
        gray = img.astype(np.float64)
    min_len = max(MIN_LINE_PX, int(img_w * MIN_LINE_LENGTH_RATIO))
    rules: List[HorizontalRule] = []
    y = 0
    while y < img_h:
        row = gray[y, :]
        var = float(np.var(row))
        mean = float(np.mean(row))
        if var < SEP_ROW_VAR_THRESHOLD and SEP_LUM_MIN < mean < SEP_LUM_MAX:
            # Extend vertically for thin band
            y_start = y
            while y < img_h and np.var(gray[y, :]) < SEP_ROW_VAR_THRESHOLD and SEP_LUM_MIN < np.mean(gray[y, :]) < SEP_LUM_MAX:
                y += 1
            y_end = y
            if y_end - y_start <= max(3, img_h // 100):  # thin
                # Find x span where line is present
                band = gray[y_start:y_end, :]
                col_means = np.mean(band, axis=0)
                in_line = (np.abs(col_means - np.mean(band)) < 30) & (col_means > SEP_LUM_MIN) & (col_means < SEP_LUM_MAX)
                xs = np.where(in_line)[0]
                if len(xs) >= min_len:
                    x_min = float(xs[0])
                    x_max = float(xs[-1] + 1)
                    rules.append(
                        HorizontalRule(
                            y_min=float(y_start),
                            y_max=float(y_end),
                            x_min=x_min,
                            x_max=x_max,
                            width_ratio=(x_max - x_min) / max(1, img_w),
                        )
                    )
            continue
        y += 1
    return rules


def _detect_vertical_rules(img: np.ndarray, img_w: int, img_h: int) -> List[VerticalRule]:
    """Thin vertical lines: low column variance."""
    if img.ndim == 3:
        gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float64)
    else:
        gray = img.astype(np.float64)
    min_len = max(MIN_LINE_PX, int(img_h * MIN_LINE_LENGTH_RATIO))
    rules: List[VerticalRule] = []
    x = 0
    while x < img_w:
        col = gray[:, x]
        var = float(np.var(col))
        mean = float(np.mean(col))
        if var < SEP_ROW_VAR_THRESHOLD and SEP_LUM_MIN < mean < SEP_LUM_MAX:
            x_start = x
            while x < img_w and np.var(gray[:, x]) < SEP_ROW_VAR_THRESHOLD and SEP_LUM_MIN < np.mean(gray[:, x]) < SEP_LUM_MAX:
                x += 1
            x_end = x
            if x_end - x_start <= max(3, img_w // 100):
                band = gray[:, x_start:x_end]
                row_means = np.mean(band, axis=1)
                in_line = (np.abs(row_means - np.mean(band)) < 30) & (row_means > SEP_LUM_MIN) & (row_means < SEP_LUM_MAX)
                ys = np.where(in_line)[0]
                if len(ys) >= min_len:
                    y_min = float(ys[0])
                    y_max = float(ys[-1] + 1)
                    rules.append(
                        VerticalRule(
                            x_min=float(x_start),
                            x_max=float(x_end),
                            y_min=y_min,
                            y_max=y_max,
                        )
                    )
            continue
        x += 1
    return rules


def run_cv_prepass(
    image_path: str,
    words: List[Word],
) -> Tuple[List[Word], List[HorizontalRule], List[VerticalRule]]:
    """
    Load image, enrich each word with visual attributes, detect separators.
    Returns (enriched_words, horizontal_rules, vertical_rules).
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("CV prepass: image not found %s", image_path)
        return words, [], []
    try:
        pil_img = Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning("CV prepass: failed to load %s: %s", image_path, e)
        return words, [], []
    img = np.array(pil_img)
    img_h, img_w = img.shape[:2]
    enriched = [_enrich_word(img, w, img_w, img_h) for w in words]
    h_rules = _detect_horizontal_rules(img, img_w, img_h)
    v_rules = _detect_vertical_rules(img, img_w, img_h)
    n_bg = sum(1 for w in enriched if w.has_background)
    logger.info(
        "CV prepass: %s words enriched, has_background=%d, h_rules=%d, v_rules=%d",
        len(enriched),
        n_bg,
        len(h_rules),
        len(v_rules),
    )
    return enriched, h_rules, v_rules
