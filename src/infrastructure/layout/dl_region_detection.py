"""
DL-based UI region detection: LayoutParser (Detectron2 + PubLayNet) for layout,
then CV second-pass inside each parent for button/badge/pill/input_like/text_region.

Output format matches cv_detect_ui_regions: flat list of
{x, y, w, h, type, parent_region_id}.
Types: card, section, text_region, button, badge, pill, input_like.
No navbar in DL path (avoids empty top strip).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.infrastructure.layout.ui_region_detection import (
    _load_image,
    _second_pass_children_in_roi,
)

logger = logging.getLogger(__name__)

# PubLayNet -> our types
PUBLAYNET_TO_TYPE: Dict[str, str] = {
    "Text": "text_region",
    "Title": "text_region",
    "List": "text_region",
    "Table": "section",
    "Figure": "card",
}

PARENT_TYPES = ("card", "section")
DL_MIN_AREA_PX2 = 400  # skip tiny DL blocks
DL_MIN_SIDE_PX = 20


def _image_to_rgb(image_path: str) -> Tuple[Optional[np.ndarray], int, int]:
    """Load image as RGB numpy (H, W, 3). Returns (img, w, h)."""
    img_bgr, w, h = _load_image(image_path)
    if img_bgr is None or w <= 0 or h <= 0:
        return None, 0, 0
    try:
        import cv2
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        img_rgb = np.array(img_bgr)
    return img_rgb, w, h


def _layout_to_regions(
    layout: Any,
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """
    Convert LayoutParser Layout to list of {x, y, w, h, type}.
    Filter small blocks; map PubLayNet types to card/section/text_region.
    """
    regions: List[Dict[str, Any]] = []
    image_area = img_w * img_h

    for block in layout:
        try:
            rect = block.block
            x_1, y_1 = float(rect.x_1), float(rect.y_1)
            x_2, y_2 = float(rect.x_2), float(rect.y_2)
        except Exception:
            continue
        w = max(0, int(round(x_2 - x_1)))
        h = max(0, int(round(y_2 - y_1)))
        x = int(round(x_1))
        y = int(round(y_1))
        if w < DL_MIN_SIDE_PX or h < DL_MIN_SIDE_PX:
            continue
        if w * h < DL_MIN_AREA_PX2:
            continue
        if w * h > image_area * 0.95:
            continue
        raw_type = getattr(block, "type", None) or getattr(block, "predicted_type", None)
        if isinstance(raw_type, str):
            our_type = PUBLAYNET_TO_TYPE.get(raw_type, "text_region")
        else:
            our_type = "text_region"
        regions.append({"x": x, "y": y, "w": w, "h": h, "type": our_type})
    return regions


def _contained_in(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True if a is fully inside b (with small tolerance)."""
    ax2 = a["x"] + a["w"]
    ay2 = a["y"] + a["h"]
    bx2 = b["x"] + b["w"]
    by2 = b["y"] + b["h"]
    tol = 2
    return (
        a["x"] >= b["x"] - tol
        and a["y"] >= b["y"] - tol
        and ax2 <= bx2 + tol
        and ay2 <= by2 + tol
    )


def _build_hierarchy(
    dl_regions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split into parents (card, section) and standalone text_regions.
    Text_regions that fall inside a parent are not returned here (CV pass will fill children).
    """
    parents: List[Dict[str, Any]] = []
    text_regions: List[Dict[str, Any]] = []

    for r in dl_regions:
        t = r.get("type", "text_region")
        if t in PARENT_TYPES:
            parents.append({**r, "parent_region_id": None})
        elif t == "text_region":
            text_regions.append(r)

    parents.sort(key=lambda r: (r["y"], r["x"]))
    standalone_text: List[Dict[str, Any]] = []
    for r in text_regions:
        inside_any = any(_contained_in(r, p) for p in parents)
        if not inside_any:
            standalone_text.append({**r, "parent_region_id": None})
    standalone_text.sort(key=lambda r: (r["y"], r["x"]))
    return parents, standalone_text


def dl_detect_ui_regions(image_path: str) -> List[Dict[str, Any]]:
    """
    DL-based UI region detection: LayoutParser (PubLayNet) + CV children inside parents.
    Returns flat list: parents (card, section), then CV children with parent_region_id,
    then standalone DL text_regions. Same contract as cv_detect_ui_regions.
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("dl_region: image not found %s", image_path)
        return []

    img_rgb, img_w, img_h = _image_to_rgb(image_path)
    if img_rgb is None or img_w <= 0 or img_h <= 0:
        return []

    layout: Any
    try:
        import layoutparser as lp
    except ImportError:
        logger.debug("dl_region: layoutparser not available")
        return []

    try:
        model = lp.Detectron2LayoutModel(
            "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        )
        layout = model.detect(img_rgb)
    except Exception as e:
        logger.warning("dl_region: Detectron2 layout detection failed: %s", e)
        return []

    if not layout:
        return []

    dl_regions = _layout_to_regions(layout, img_w, img_h)
    parents, standalone_text = _build_hierarchy(dl_regions)

    # Load BGR image for CV second pass
    img_bgr, _, _ = _load_image(image_path)
    if img_bgr is None:
        img_bgr = np.array(img_rgb)
        try:
            import cv2
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    all_regions: List[Dict[str, Any]] = []
    for i, p in enumerate(parents):
        all_regions.append({**p})
    for i, p in enumerate(parents):
        children = _second_pass_children_in_roi(
            img_bgr, p, i, img_w, img_h
        )
        for c in children:
            all_regions.append(c)
    for r in standalone_text:
        all_regions.append({**r})

    by_type: Dict[str, int] = {}
    for r in all_regions:
        t = r.get("type", "text_region")
        by_type[t] = by_type.get(t, 0) + 1
    logger.info(
        "dl_region: image_path=%s parents=%d children+standalone=%d by_type=%s",
        image_path, len(parents), len(all_regions) - len(parents), by_type,
    )
    return all_regions
