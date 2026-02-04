"""
Region prepass: CV-level regions (text / UI) before OCR grouping.

Regions are top-level; words are assigned to one region; layout runs only inside each region.
Classic CV: color (HSV/LAB), k-means, morphology, contours, filter by aspect/area/rectangularity.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .atoms import Region, RegionType, Word
from .region_merge import merge_regions_atoms, OVERLAP_RATIO

logger = logging.getLogger(__name__)

# K-means for color layers
KMEANS_K = 4
# Morphology
CLOSE_KERNEL = (7, 3)  # rect kernel: glue text into lines
OPEN_KERNEL = (3, 3)
# Contour filter
MIN_AREA_RATIO = 0.001   # min contour area / image_area
MAX_AREA_RATIO = 0.95    # ignore near-full-screen
MIN_RECTANGULARITY = 0.7  # contour area / bounding_rect_area
UI_ASPECT_MIN = 1.2       # ui_region: aspect w/h >= 1.2
MIN_REGION_SIDE = 12     # px
# UI: uniform background, padding, aspect; allow small CTAs (View details)
UI_MIN_AREA_RATIO = 0.0001


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2LAB)
    except ImportError:
        return rgb


def _kmeans_labels(img_flat: np.ndarray, k: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        return km.fit_predict(img_flat)
    except ImportError:
        return np.zeros(img_flat.shape[0], dtype=np.int32)


def _get_regions_from_mask(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    region_type: RegionType,
) -> List[Region]:
    try:
        import cv2
    except ImportError:
        return []
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_KERNEL)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    contours, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = img_w * img_h
    regions: List[Region] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_REGION_SIDE * MIN_REGION_SIDE:
            continue
        if area < MIN_AREA_RATIO * image_area:
            continue
        if area > MAX_AREA_RATIO * image_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < MIN_REGION_SIDE or h < MIN_REGION_SIDE:
            continue
        rect_area = w * h
        if rect_area <= 0:
            continue
        rectangularity = area / rect_area
        if rectangularity < MIN_RECTANGULARITY:
            continue
        aspect = w / max(1, h)
        if region_type == "ui_region" and aspect < UI_ASPECT_MIN:
            continue
        regions.append(
            Region(x=x, y=y, w=w, h=h, region_type=region_type, area=rect_area)
        )
    return regions


def cv_detect_regions(image_path: str) -> List[Region]:
    """
    Detect regions from image (before OCR). Returns list of Region (bbox + type).
    Pipeline: load → LAB → k-means → per-cluster mask → morphology → contours → filter.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("region_prepass: cv2 not available")
        return []
    path = Path(image_path)
    if not path.exists():
        logger.warning("region_prepass: image not found %s", image_path)
        return []
    try:
        img = cv2.imread(str(path))
        if img is None:
            img = cv2.cvtColor(
                np.array(__import__("PIL.Image").Image.open(path).convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )
    except Exception as e:
        logger.warning("region_prepass: failed to load %s: %s", image_path, e)
        return []
    if img is None or img.size == 0:
        return []
    img_h, img_w = img.shape[:2]
    lab = _to_lab(img)
    # Use L channel + ab for clustering (reshape to Nx3)
    if lab.ndim == 3:
        flat = lab.reshape(-1, 3).astype(np.float32)
    else:
        flat = lab.reshape(-1, 1).astype(np.float32)
    k = min(KMEANS_K, len(np.unique(flat, axis=0)))
    if k < 2:
        k = 2
    labels = _kmeans_labels(flat, k)
    labels_2d = labels.reshape(img_h, img_w)

    image_area = img_w * img_h
    all_regions: List[Region] = []
    for ki in range(k):
        mask = (labels_2d == ki).astype(np.uint8) * 255
        mean_l = float(np.mean(lab[labels_2d == ki, 0])) if np.any(labels_2d == ki) else 128
        # Heuristic: very dark or very light cluster → likely background
        if mean_l < 30 or mean_l > 220:
            regs = _get_regions_from_mask(mask, img_w, img_h, "background")
            for r in regs:
                if r.area < image_area * 0.5:
                    all_regions.append(r)
            continue
        # Mid luminance: text or UI
        cluster_area_ratio = np.sum(mask > 0) / max(1, image_area)
        if cluster_area_ratio > 0.7:
            continue
        regs = _get_regions_from_mask(mask, img_w, img_h, "text_region")
        for r in regs:
            all_regions.append(r)
        # UI: compact, aspect >= 1.2; include small CTAs (no upper area cap for single buttons)
        ui_regs = _get_regions_from_mask(mask, img_w, img_h, "ui_region")
        for r in ui_regs:
            if r.area >= image_area * UI_MIN_AREA_RATIO and r.area <= image_area * 0.5:
                all_regions.append(r)

    # Mandatory merge: intersection / min(area_a, area_b) >= 0.9 → merge (before OCR/classification)
    all_regions = merge_regions_atoms(all_regions, overlap_threshold=OVERLAP_RATIO)
    # Sort by y then x
    all_regions.sort(key=lambda r: (r.y, r.x))
    n_text = sum(1 for r in all_regions if r.region_type == "text_region")
    n_ui = sum(1 for r in all_regions if r.region_type == "ui_region")
    n_bg = sum(1 for r in all_regions if r.region_type == "background")
    logger.info(
        "region_prepass: %d regions (text=%d ui=%d bg=%d) for %s",
        len(all_regions), n_text, n_ui, n_bg, image_path,
    )
    return all_regions


def _iou_rect(a: Region, b: Region) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    u = a.area + b.area - inter
    return inter / u if u > 0 else 0.0


# Word assigned to region only if overlap >= this fraction of word area (else fallback)
WORD_REGION_OVERLAP_MIN = 0.3


def assign_words_to_region(
    words: List[Word],
    regions: List[Region],
) -> List[Tuple[Region, List[Word]]]:
    """
    Assign each word to at most one region (max overlap). Returns (region, words) per region.
    Words with overlap < WORD_REGION_OVERLAP_MIN or outside any region go to fallback "page" region.
    Region does NOT filter out words — fallback gets full layout.
    """
    if not regions:
        return []
    img_w = max((w.x + w.w for w in words), default=1)
    img_h = max((w.y + w.h for w in words), default=1)
    fallback_region = Region(0, 0, img_w, img_h, "text_region", img_w * img_h)
    region_words: List[List[Word]] = [[] for _ in range(len(regions) + 1)]
    for w in words:
        best_idx = -1
        best_overlap = 0.0
        w_area = max(1, w.w * w.h)
        for i, r in enumerate(regions):
            ox1 = max(w.x, r.x)
            oy1 = max(w.y, r.y)
            ox2 = min(w.x + w.w, r.x + r.w)
            oy2 = min(w.y + w.h, r.y + r.h)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            overlap = (ox2 - ox1) * (oy2 - oy1) / w_area
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        # Only assign to region if overlap is significant; else fallback (word never dropped)
        if best_idx < 0 or best_overlap < WORD_REGION_OVERLAP_MIN:
            region_words[-1].append(w)
        else:
            region_words[best_idx].append(w)
    result: List[Tuple[Region, List[Word]]] = []
    for i, r in enumerate(regions):
        result.append((r, region_words[i]))
    if region_words[-1]:
        result.append((fallback_region, region_words[-1]))
    return result
