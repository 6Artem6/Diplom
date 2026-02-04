"""
Hierarchical UI region detection: parents (card, navbar, section) then children (button, badge, text_region, etc.) inside ROI.

Two passes:
1. First pass: large regions only (area ≥ X% screen) → card, navbar, section.
2. Second pass: for each parent, crop ROI, scale ×1.5–2, run contour detection with different thresholds → button, badge, text_region, input-like, pill. Remap coords to global, set parent_region_id.

No single-level heuristics; children are found only inside parents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .region_merge import merge_regions_dict, OVERLAP_RATIO

logger = logging.getLogger(__name__)

UI_REGION_TYPES = ("button", "badge", "card", "text_region", "navbar", "section", "input_like", "pill")
PARENT_TYPES = ("navbar", "card", "section")
CHILD_TYPES = ("button", "badge", "text_region", "input_like", "pill")
# For 90% merge: merged type = higher priority
TYPE_PRIORITY_CHILDREN = ("button", "input_like", "pill", "badge", "text_region")

# --- First pass: large regions (parents) ---
PARENT_MIN_AREA_RATIO = 0.015   # ≥ 1.5% of screen
PARENT_MAX_AREA_RATIO = 0.95
PARENT_MIN_RECTANGULARITY = 0.6
PARENT_MIN_SIDE_PX = 30
NAVBAR_HEIGHT_RATIO = 0.12
NAVBAR_WIDTH_RATIO = 0.85
CARD_MIN_AREA_RATIO = 0.02
CARD_ASPECT_RANGE = (0.4, 5.0)

# --- Second pass: inside ROI (children) ---
ROI_SCALE = 2.0
CHILD_MIN_AREA_PX2 = 120   # skip tiny blobs (e.g. 9×11) from any UI control
CHILD_MIN_SIDE_PX = 8
CHILD_RECTANGULARITY = 0.55
BUTTON_ASPECT_MIN = 1.2
BUTTON_ASPECT_MAX = 5.0
BUTTON_MAX_AREA_RATIO_IN_ROI = 0.4
# Button: text-relative mins preferred; absolutes only as fallback (small "OK"/"Yes"/"Add")
BUTTON_MIN_WIDTH_PX = 16
BUTTON_MIN_HEIGHT_PX = 8
BUTTON_MIN_AREA_PX2 = 128
BUTTON_MAX_WIDTH_RATIO = 0.6
# Badge: tag-like only by geometry (narrow, small). NOT "text ≈ container" → badge.
BADGE_MAX_HEIGHT_PX = 45
BADGE_MAX_AREA_RATIO_IN_ROI = 0.15
BADGE_MIN_WIDTH_PX = 14
BADGE_MIN_HEIGHT_PX = 8
BADGE_MIN_AREA_PX2 = 100
BADGE_MAX_WIDTH_PX = 72  # tag-like: short label only; wider → button
PILL_MIN_WIDTH_PX = 20
PILL_MIN_HEIGHT_PX = 10
PILL_MIN_AREA_PX2 = 200
INPUT_LIKE_MIN_WIDTH_PX = 30
INPUT_LIKE_MIN_HEIGHT_PX = 12
INPUT_LIKE_MIN_AREA_PX2 = 360
OUTLINE_FILL_RATIO_MAX = 0.65


def _load_image(image_path: str) -> Tuple[Any, int, int]:
    path = Path(image_path)
    if not path.exists():
        logger.warning("ui_region: image not found %s", image_path)
        return None, 0, 0
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            img = cv2.cvtColor(
                np.array(__import__("PIL.Image").Image.open(path).convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )
    except Exception as e:
        logger.warning("ui_region: failed to load %s: %s", image_path, e)
        return None, 0, 0
    if img is None or img.size == 0:
        return None, 0, 0
    h, w = img.shape[:2]
    return img, w, h


def _contour_fill_ratio(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    try:
        import cv2
    except ImportError:
        return 1.0
    if w <= 0 or h <= 0:
        return 0.0
    x1, y1 = max(0, x), max(0, y)
    x2 = min(gray.shape[1], x + w)
    y2 = min(gray.shape[0], y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = gray[y1:y2, x1:x2]
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.sum(binary > 0)) / max(1, binary.size)


def _first_pass_parents(img: Any, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Large regions only: navbar (only if top strip has content), card, section."""
    import cv2
    image_area = img_w * img_h
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regions: List[Dict[str, Any]] = []

    edges = cv2.Canny(gray, 40, 120)
    navbar_h = max(PARENT_MIN_SIDE_PX, int(img_h * NAVBAR_HEIGHT_RATIO))
    NAVBAR_MIN_EDGE_PX = 80
    if navbar_h < img_h and int(img_w * NAVBAR_WIDTH_RATIO) > 0:
        top_edges = edges[0:navbar_h, 0:img_w]
        if np.sum(top_edges > 0) >= NAVBAR_MIN_EDGE_PX:
            regions.append({
                "x": 0, "y": 0, "w": img_w, "h": navbar_h,
                "type": "navbar",
                "parent_region_id": None,
            })
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < image_area * PARENT_MIN_AREA_RATIO:
            continue
        if area > image_area * PARENT_MAX_AREA_RATIO:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < PARENT_MIN_SIDE_PX or h < PARENT_MIN_SIDE_PX:
            continue
        rect_area = w * h
        if rect_area <= 0:
            continue
        if area / rect_area < PARENT_MIN_RECTANGULARITY:
            continue
        if y < navbar_h and h <= navbar_h * 1.2:
            continue
        aspect = w / max(1, h)
        ar = rect_area / image_area
        if ar >= CARD_MIN_AREA_RATIO and CARD_ASPECT_RANGE[0] <= aspect <= CARD_ASPECT_RANGE[1]:
            regions.append({
                "x": x, "y": y, "w": w, "h": h,
                "type": "card",
                "parent_region_id": None,
            })
        elif ar >= PARENT_MIN_AREA_RATIO:
            regions.append({
                "x": x, "y": y, "w": w, "h": h,
                "type": "section",
                "parent_region_id": None,
            })

    # Mandatory merge: intersection/min(area) >= 0.9 (before classification/OCR)
    regions = merge_regions_dict(
        regions,
        overlap_threshold=OVERLAP_RATIO,
        type_priority=["card", "section", "navbar"],
    )
    regions.sort(key=lambda r: (r["y"], r["x"]))
    return regions


def _second_pass_children_in_roi(
    img: Any,
    parent: Dict[str, Any],
    parent_index: int,
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """Detect children (button, badge, text_region, etc.) inside parent ROI. Coords returned in global image space."""
    import cv2
    rx, ry, rw, rh = parent["x"], parent["y"], parent["w"], parent["h"]
    roi = img[ry : ry + rh, rx : rx + rw]
    if roi.size == 0:
        return []
    roi_h, roi_w = roi.shape[:2]
    scale = ROI_SCALE
    scaled = cv2.resize(roi, (int(roi_w * scale), int(roi_h * scale)), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    sw, sh = int(roi_w * scale), int(roi_h * scale)
    roi_area = sw * sh

    edges = cv2.Canny(gray, 60, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    children: List[Dict[str, Any]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < CHILD_MIN_AREA_PX2:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < CHILD_MIN_SIDE_PX or h < CHILD_MIN_SIDE_PX:
            continue
        rect_area = w * h
        if rect_area <= 0 or area / rect_area < CHILD_RECTANGULARITY:
            continue
        ar = rect_area / max(1, roi_area)
        aspect = w / max(1, h)
        fill = _contour_fill_ratio(gray, x, y, w, h)

        # Remap to global (scale down: coords were in scaled ROI)
        gx = rx + int(x / scale)
        gy = ry + int(y / scale)
        gw = max(1, int(w / scale))
        gh = max(1, int(h / scale))
        area_global = gw * gh
        max_w = int(BUTTON_MAX_WIDTH_RATIO * rw)
        max_h = int(BUTTON_MAX_WIDTH_RATIO * rh)

        if gw > max_w or gh > max_h:
            pass
        elif BUTTON_ASPECT_MIN <= aspect <= BUTTON_ASPECT_MAX and ar <= BUTTON_MAX_AREA_RATIO_IN_ROI and gw >= BUTTON_MIN_WIDTH_PX and gh >= BUTTON_MIN_HEIGHT_PX and area_global >= BUTTON_MIN_AREA_PX2:
            # Button: rectangular geometry, centering; allow no fill (outline/ghost). Badge decided later by text (h_region/h_text ≤ 1.4).
            children.append({"x": gx, "y": gy, "w": gw, "h": gh, "type": "button", "parent_region_id": parent_index})
            continue
        # Badge: tag-like only — narrow width, small; NOT "text ≈ container"
        if ar <= BADGE_MAX_AREA_RATIO_IN_ROI and gh <= BADGE_MAX_HEIGHT_PX and gw <= BADGE_MAX_WIDTH_PX and gw >= BADGE_MIN_WIDTH_PX and gh >= BADGE_MIN_HEIGHT_PX and area_global >= BADGE_MIN_AREA_PX2:
            children.append({"x": gx, "y": gy, "w": gw, "h": gh, "type": "badge", "parent_region_id": parent_index})
            continue
        if 2.0 <= aspect <= 8.0 and ar <= 0.2 and gw <= max_w and gh <= max_h and gw >= INPUT_LIKE_MIN_WIDTH_PX and gh >= INPUT_LIKE_MIN_HEIGHT_PX and area_global >= INPUT_LIKE_MIN_AREA_PX2:
            children.append({"x": gx, "y": gy, "w": gw, "h": gh, "type": "input_like", "parent_region_id": parent_index})
            continue
        if aspect >= 0.8 and aspect <= 2.5 and ar <= 0.25 and gw <= max_w and gh <= max_h and gw >= PILL_MIN_WIDTH_PX and gh >= PILL_MIN_HEIGHT_PX and area_global >= PILL_MIN_AREA_PX2:
            children.append({"x": gx, "y": gy, "w": gw, "h": gh, "type": "pill", "parent_region_id": parent_index})
            continue
        if ar <= 0.7:
            children.append({"x": gx, "y": gy, "w": gw, "h": gh, "type": "text_region", "parent_region_id": parent_index})

    # Mandatory merge: intersection/min(area) >= 0.9; type priority button > input_like > pill > badge > text_region
    merged = merge_regions_dict(
        children,
        overlap_threshold=OVERLAP_RATIO,
        type_priority=list(TYPE_PRIORITY_CHILDREN),
    )
    return merged


def cv_detect_ui_regions(image_path: str) -> List[Dict[str, Any]]:
    """
    Hierarchical UI region detection.
    Returns flat list: parents first (navbar, card, section), then children with parent_region_id set.
    Each item: {x, y, w, h, type, parent_region_id}.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("ui_region: cv2 not available")
        return []
    img, img_w, img_h = _load_image(image_path)
    if img is None or img_w <= 0 or img_h <= 0:
        return []

    parents = _first_pass_parents(img, img_w, img_h)
    all_regions: List[Dict[str, Any]] = []
    for i, p in enumerate(parents):
        all_regions.append({**p})
    for i, p in enumerate(parents):
        children = _second_pass_children_in_roi(img, p, i, img_w, img_h)
        for c in children:
            all_regions.append(c)

    # Global merge: any two regions with overlap >= 90% merged (type by priority)
    all_regions = merge_regions_dict(
        all_regions,
        overlap_threshold=OVERLAP_RATIO,
        type_priority=["button", "input_like", "pill", "badge", "text_region", "card", "section", "navbar"],
    )

    n_parents = sum(1 for r in all_regions if r.get("parent_region_id") is None)
    n_children = len(all_regions) - n_parents
    by_type: Dict[str, int] = {}
    for r in all_regions:
        t = r.get("type", "text_region")
        by_type[t] = by_type.get(t, 0) + 1
    logger.info(
        "ui_region: image_path=%s parents=%d children=%d by_type=%s",
        image_path, n_parents, n_children, by_type,
    )
    return all_regions


def is_parent_region(region: Dict[str, Any]) -> bool:
    return region.get("parent_region_id") is None


def is_child_region(region: Dict[str, Any]) -> bool:
    return region.get("parent_region_id") is not None
