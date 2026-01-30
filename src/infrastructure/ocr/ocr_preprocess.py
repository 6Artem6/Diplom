"""
Pre-OCR preprocessing for thin/light text on saturated backgrounds (Bootstrap primary/secondary, badges).

- Full page: LAB L-channel + CLAHE + Otsu (or optional adaptive threshold). Better on colored primary.
- Local crop: grayscale, CLAHE, Otsu, dilation, invert if dark, upscale.
Does NOT scale the whole page. Color is never used to drop words — only for preprocessing and role.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = 8
# Use LAB L-channel for full-page (better on primary/secondary colored background)
USE_LAB_L_FOR_FULL_PAGE = True
# Adaptive threshold block size (odd); 0 = use Otsu only
ADAPTIVE_BLOCK_SIZE = 0  # set to e.g. 31 to enable adaptive on full page
# Dilation for thin glyphs (1–2 px)
DILATION_KERNEL_SIZE = 2
# Dark background: mean below this → invert
DARK_BG_MEAN_THRESHOLD = 128
# Min crop size for preprocessing
MIN_CROP_SIDE = 4


@dataclass
class PreprocessResult:
    """Result of preprocess_crop with diagnostics."""

    image: np.ndarray
    applied_dilation: bool = False
    applied_inversion: bool = False
    upscale_factor: float = 1.0


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    return np.round(
        0.299 * img[:, :, 0].astype(np.float64)
        + 0.587 * img[:, :, 1].astype(np.float64)
        + 0.114 * img[:, :, 2].astype(np.float64)
    ).astype(np.uint8)


def _to_lab_l(img: np.ndarray) -> np.ndarray:
    """Extract L channel (0..255) from LAB. Use for CLAHE on colored backgrounds."""
    try:
        import cv2
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]
    except ImportError:
        return _to_grayscale(img)


def preprocess_full_page(img: np.ndarray) -> np.ndarray:
    """
    Preprocess full page for first OCR pass.
    Uses LAB L-channel + CLAHE + Otsu (better on primary/secondary colored background);
    optional adaptive threshold if ADAPTIVE_BLOCK_SIZE > 0.
    No scaling, no dilation (to avoid merging glyphs globally).
    """
    try:
        import cv2
    except ImportError:
        logger.warning("ocr_preprocess: cv2 not available, returning grayscale only")
        return _to_grayscale(img)

    if USE_LAB_L_FOR_FULL_PAGE and img.ndim >= 3:
        # CLAHE on L (LAB): improves contrast on colored backgrounds (primary blue, etc.)
        gray = _to_lab_l(img)
    else:
        gray = _to_grayscale(img)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_GRID_SIZE, CLAHE_GRID_SIZE))
    enhanced = clahe.apply(gray)

    if ADAPTIVE_BLOCK_SIZE > 0 and ADAPTIVE_BLOCK_SIZE % 2 == 1:
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, 8
        )
    else:
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_crop(
    crop: np.ndarray,
    dilation_px: int = 1,
    upscale_factor: float = 2.0,
    invert_if_dark: bool = True,
) -> PreprocessResult:
    """
    Preprocess a single bbox crop for OCR: grayscale, CLAHE, Otsu, optional dilation, invert if dark, upscale.

    Used for fallback OCR on light/gray text regions.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("ocr_preprocess: cv2 not available")
        gray = _to_grayscale(crop)
        return PreprocessResult(image=gray, applied_dilation=False, applied_inversion=False, upscale_factor=1.0)

    if crop.size == 0:
        return PreprocessResult(image=crop, applied_dilation=False, applied_inversion=False, upscale_factor=1.0)

    gray = _to_grayscale(crop)
    h, w = gray.shape[:2]
    if w < MIN_CROP_SIDE or h < MIN_CROP_SIDE:
        return PreprocessResult(image=gray, applied_dilation=False, applied_inversion=False, upscale_factor=1.0)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_GRID_SIZE, CLAHE_GRID_SIZE))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    applied_inversion = False
    if invert_if_dark and float(np.mean(binary)) < DARK_BG_MEAN_THRESHOLD:
        binary = 255 - binary
        applied_inversion = True

    applied_dilation = False
    if dilation_px >= 1:
        k = max(1, min(dilation_px, 3))
        kernel = np.ones((k, k), np.uint8)
        binary = cv2.dilate(binary, kernel)
        applied_dilation = True

    out = binary
    upscale_factor_used = 1.0
    if upscale_factor > 1.0 and w * h < 400 * 400:
        new_w = max(MIN_CROP_SIDE, int(w * upscale_factor))
        new_h = max(MIN_CROP_SIDE, int(h * upscale_factor))
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        upscale_factor_used = upscale_factor

    return PreprocessResult(
        image=out,
        applied_dilation=applied_dilation,
        applied_inversion=applied_inversion,
        upscale_factor=upscale_factor_used,
    )


def crop_bbox(
    img: np.ndarray, x: int, y: int, w: int, h: int, padding: int = 2
) -> Tuple[np.ndarray, int, int]:
    """
    Crop image to bbox with padding. Returns (crop, crop_x_offset, crop_y_offset) for mapping back.
    """
    img_h, img_w = img.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    if x2 <= x1 or y2 <= y1:
        return np.array([], dtype=np.uint8).reshape(0, 0), x, y
    crop = img[y1:y2, x1:x2].copy()
    return crop, x1, y1
