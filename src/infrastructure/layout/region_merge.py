"""
Deterministic region merge: overlapping regions must be merged before classification/OCR.

Rule A: intersection_area / min(area_a, area_b) >= 0.9 → merge.
Rule B: intersection_area / area_smaller >= 0.75 AND center(smaller) ∈ larger → merge
       (covers almost-contained regions, contour noise).
Merged type by priority: button > input_like > pill > badge > text_region (and parents unchanged).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

OVERLAP_RATIO = 0.9   # Rule A: intersection / min(area_a, area_b) >= this → merge
OVERLAP_RATIO_B = 0.75  # Rule B: intersection / area_smaller >= this
# Rule B: center of smaller must lie inside larger (with 1px tolerance)
CENTERS_TOLERANCE_PX = 1

# For dict-based regions with "type" key. Parents (card, section, navbar) are not merged with children.
TYPE_PRIORITY_UI: List[str] = [
    "button",
    "input_like",
    "pill",
    "badge",
    "text_region",
]

# For Region (atoms): only text_region, ui_region, background
RegionTypeAtom = Literal["text_region", "ui_region", "background"]
TYPE_PRIORITY_ATOM: List[RegionTypeAtom] = ["ui_region", "text_region", "background"]


def _area_rect(x: int, y: int, w: int, h: int) -> int:
    return max(0, w) * max(0, h)


def _intersection_rect(
    ax: int, ay: int, aw: int, ah: int,
    bx: int, by: int, bw: int, bh: int,
) -> int:
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def overlap_ratio_min_area(
    ax: int, ay: int, aw: int, ah: int,
    bx: int, by: int, bw: int, bh: int,
) -> float:
    """intersection_area / min(area_a, area_b). 0 if no overlap."""
    inter = _intersection_rect(ax, ay, aw, ah, bx, by, bw, bh)
    if inter <= 0:
        return 0.0
    area_a = _area_rect(ax, ay, aw, ah)
    area_b = _area_rect(bx, by, bw, bh)
    min_a = min(area_a, area_b)
    if min_a <= 0:
        return 0.0
    return inter / min_a


def _center_inside(
    ax: int, ay: int, aw: int, ah: int,
    bx: int, by: int, bw: int, bh: int,
    tol: int = CENTERS_TOLERANCE_PX,
) -> bool:
    """True if center of (ax,ay,aw,ah) lies inside (bx,by,bw,bh). Smaller first for Rule B."""
    cx = ax + aw // 2
    cy = ay + ah // 2
    return (
        bx - tol <= cx <= bx + bw + tol
        and by - tol <= cy <= by + bh + tol
    )


def _should_merge_rule_b(
    ax: int, ay: int, aw: int, ah: int,
    bx: int, by: int, bw: int, bh: int,
    overlap_threshold_b: float = OVERLAP_RATIO_B,
) -> bool:
    """
    Rule B: merge if intersection/area_smaller >= 0.75 AND center(smaller) ∈ larger.
    (ax,ay,aw,ah) = current region r, (bx,by,bw,bh) = existing region o.
    """
    inter = _intersection_rect(ax, ay, aw, ah, bx, by, bw, bh)
    area_a = _area_rect(ax, ay, aw, ah)
    area_b = _area_rect(bx, by, bw, bh)
    if area_a <= 0 or area_b <= 0 or inter <= 0:
        return False
    smaller_is_a = area_a <= area_b
    area_smaller = area_a if smaller_is_a else area_b
    if inter / area_smaller < overlap_threshold_b:
        return False
    if smaller_is_a:
        return _center_inside(ax, ay, aw, ah, bx, by, bw, bh)
    return _center_inside(bx, by, bw, bh, ax, ay, aw, ah)


def _merge_bbox(
    ax: int, ay: int, aw: int, ah: int,
    bx: int, by: int, bw: int, bh: int,
) -> Tuple[int, int, int, int]:
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def _choose_type_priority(t1: str, t2: str, priority: List[str]) -> str:
    i1 = next((i for i, p in enumerate(priority) if p == t1), len(priority))
    i2 = next((i for i, p in enumerate(priority) if p == t2), len(priority))
    return t1 if i1 <= i2 else t2


def merge_regions_dict(
    regions: List[Dict[str, Any]],
    overlap_threshold: float = OVERLAP_RATIO,
    overlap_threshold_b: float = OVERLAP_RATIO_B,
    type_priority: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Merge dict-based regions: each has x, y, w, h, type.
    Rule A: overlap_ratio_min_area >= overlap_threshold → merge.
    Rule B: intersection/area_smaller >= overlap_threshold_b AND center(smaller) ∈ larger → merge.
    type_priority: default TYPE_PRIORITY_UI. Types not in list keep first seen.
    """
    if not regions:
        return []
    priority = type_priority or TYPE_PRIORITY_UI
    out: List[Dict[str, Any]] = []
    for r in regions:
        rx = int(r.get("x", 0))
        ry = int(r.get("y", 0))
        rw = int(r.get("w", 0))
        rh = int(r.get("h", 0))
        rtype = r.get("type", "text_region")
        merged = False
        for i, o in enumerate(out):
            ox = int(o.get("x", 0))
            oy = int(o.get("y", 0))
            ow = int(o.get("w", 0))
            oh = int(o.get("h", 0))
            ratio = overlap_ratio_min_area(rx, ry, rw, rh, ox, oy, ow, oh)
            rule_a = ratio >= overlap_threshold
            rule_b = _should_merge_rule_b(rx, ry, rw, rh, ox, oy, ow, oh, overlap_threshold_b)
            if rule_a or rule_b:
                nx, ny, nw, nh = _merge_bbox(rx, ry, rw, rh, ox, oy, ow, oh)
                otype = o.get("type", "text_region")
                new_type = _choose_type_priority(rtype, otype, priority)
                out[i] = {
                    **o,
                    "x": nx, "y": ny, "w": nw, "h": nh,
                    "type": new_type,
                    "parent_region_id": None,  # merged region: hierarchy ambiguous
                }
                merged = True
                break
        if not merged:
            out.append({**r, "x": rx, "y": ry, "w": rw, "h": rh, "type": rtype})
    return out


class RegionLike(Protocol):
    """Protocol for Region (atoms) with x, y, w, h, region_type, area."""

    x: int
    y: int
    w: int
    h: int
    region_type: str
    area: int


def merge_regions_atoms(
    regions: List[RegionLike],
    overlap_threshold: float = OVERLAP_RATIO,
    overlap_threshold_b: float = OVERLAP_RATIO_B,
    type_priority: Optional[List[str]] = None,
) -> List[RegionLike]:
    """
    Merge Region-like objects (atoms). Rule A: 90%; Rule B: 75% + center in larger.
    Returns new list; Region type must support copy with updated x,y,w,h,region_type,area.
    """
    if not regions:
        return []
    priority = type_priority or list(TYPE_PRIORITY_ATOM)
    out: List[RegionLike] = []
    for r in regions:
        merged = False
        for i, o in enumerate(out):
            ratio = overlap_ratio_min_area(r.x, r.y, r.w, r.h, o.x, o.y, o.w, o.h)
            rule_a = ratio >= overlap_threshold
            rule_b = _should_merge_rule_b(r.x, r.y, r.w, r.h, o.x, o.y, o.w, o.h, overlap_threshold_b)
            if rule_a or rule_b:
                nx, ny, nw, nh = _merge_bbox(r.x, r.y, r.w, r.h, o.x, o.y, o.w, o.h)
                new_type = _choose_type_priority(getattr(r, "region_type", "text_region"), getattr(o, "region_type", "text_region"), priority)
                new_area = nw * nh
                # Build new instance same type as r (Region dataclass)
                out[i] = type(r)(x=nx, y=ny, w=nw, h=nh, region_type=new_type, area=new_area)  # type: ignore[call-arg]
                merged = True
                break
        if not merged:
            out.append(r)
    return out
