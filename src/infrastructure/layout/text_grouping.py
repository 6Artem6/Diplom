"""
Text grouping: layout-first. font_size = text_box.h. Same region_id only.

Horizontal → line: |baseline_y1 - baseline_y2| <= 0.5*font_size, font_size_ratio <= 1.5, gap <= 2-3*char_width.
Vertical → paragraph: font_size_ratio <= 1.3, vertical_gap <= 1.5*font_size.
No merge across region_id. No text_block wider than region.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASELINE_Y_TOLERANCE = 0.5   # |baseline_y1 - baseline_y2| <= 0.5 * font_size
FONT_SIZE_RATIO_LINE = 1.5
HORIZ_GAP_CHAR_FACTOR = 2.5  # horizontal_gap <= 2.5 * avg_char_width (2-3)
FONT_SIZE_RATIO_PARA = 1.3
VERT_GAP_FONT_FACTOR = 1.5   # vertical_gap <= 1.5 * font_size


def _baseline_y(box: Dict[str, Any]) -> float:
    """Approximate baseline Y (cap height ~0.75 from top). font_size = box.h."""
    h = box.get("h", 0)
    y = box.get("y", 0)
    return float(y) + float(h) * 0.75


def _font_size(box: Dict[str, Any]) -> float:
    """font_size = text_box.h (not word height for block logic)."""
    return float(box.get("h", 0))


def _avg_char_width(box: Dict[str, Any], text: str = "") -> float:
    """Estimate char width from box.w and text length or default."""
    w = box.get("w", 0)
    n = max(1, len((text or "").strip()) or 1)
    return float(w) / n


def assign_text_boxes_to_regions(
    text_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    min_overlap_ratio: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Assign each text_box to one region_id (index). Add region_id to each box.
    Boxes with no overlap go to region_id = -1 (page fallback; do not merge with others).
    """
    result = []
    for i, box in enumerate(text_boxes):
        bx, by, bw, bh = box.get("x", 0), box.get("y", 0), box.get("w", 0), box.get("h", 0)
        box_area = max(1, bw * bh)
        best_rid = -1
        best_overlap = 0.0
        for ri, reg in enumerate(regions):
            rx, ry, rw, rh = reg["x"], reg["y"], reg["w"], reg["h"]
            ox1 = max(bx, rx)
            oy1 = max(by, ry)
            ox2 = min(bx + bw, rx + rw)
            oy2 = min(by + bh, ry + rh)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            overlap = (ox2 - ox1) * (oy2 - oy1) / box_area
            if overlap > best_overlap:
                best_overlap = overlap
                best_rid = ri
        if best_overlap < min_overlap_ratio:
            best_rid = -1
        out = dict(box)
        out["region_id"] = best_rid
        out["box_index"] = i  # index into original text_boxes / ocr_results
        result.append(out)
    return result


def group_text_boxes_into_lines(
    boxes_with_region: List[Dict[str, Any]],
    region_id: int,
    texts: Optional[List[str]] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Group text_boxes with same region_id into lines.
    Rule: |baseline_y1 - baseline_y2| <= 0.5*font_size, font_size_ratio <= 1.5, horizontal_gap <= 2.5*avg_char_width.
    font_size = box.h.
    """
    subset_with_idx = [(b, i) for i, b in enumerate(boxes_with_region) if b.get("region_id") == region_id]
    if not subset_with_idx:
        return []
    sorted_boxes = sorted(subset_with_idx, key=lambda bi: (_baseline_y(bi[0]), bi[0].get("x", 0)))
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [sorted_boxes[0][0]]
    for b, _ in sorted_boxes[1:]:
        ref = current[0]
        ref_fs = _font_size(ref)
        b_fs = _font_size(b)
        fs_ratio = max(ref_fs, b_fs) / max(1, min(ref_fs, b_fs))
        if fs_ratio > FONT_SIZE_RATIO_LINE:
            lines.append(current)
            current = [b]
            continue
        by_ref = _baseline_y(ref)
        by_b = _baseline_y(b)
        if abs(by_b - by_ref) > BASELINE_Y_TOLERANCE * max(ref_fs, b_fs):
            lines.append(current)
            current = [b]
            continue
        last = current[-1]
        gap = b.get("x", 0) - (last.get("x", 0) + last.get("w", 0))
        avg_cw_ref = _avg_char_width(last)
        avg_cw_b = _avg_char_width(b)
        if gap > HORIZ_GAP_CHAR_FACTOR * (avg_cw_ref + avg_cw_b) / 2.0:
            lines.append(current)
            current = [b]
        else:
            current.append(b)
    if current:
        lines.append(current)
    return lines


def group_lines_into_paragraphs(
    lines: List[List[Dict[str, Any]]],
    font_size_fn: Any = _font_size,
) -> List[List[List[Dict[str, Any]]]]:
    """
    Group lines into paragraphs. Same region assumed (call per region).
    Rule: font_size_ratio <= 1.3, vertical_gap <= 1.5*font_size.
    """
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda ln: min(b.get("y", 0) for b in ln))
    paragraphs: List[List[List[Dict[str, Any]]]] = []
    current: List[List[Dict[str, Any]]] = [sorted_lines[0]]
    for ln in sorted_lines[1:]:
        prev_ln = current[-1]
        prev_bottom = max(b.get("y", 0) + b.get("h", 0) for b in prev_ln)
        curr_top = min(b.get("y", 0) for b in ln)
        gap = curr_top - prev_bottom
        prev_fs = font_size_fn(prev_ln[0]) if prev_ln else 18.0
        curr_fs = font_size_fn(ln[0]) if ln else 18.0
        fs_ratio = max(prev_fs, curr_fs) / max(1, min(prev_fs, curr_fs))
        if gap > VERT_GAP_FONT_FACTOR * min(prev_fs, curr_fs) or fs_ratio > FONT_SIZE_RATIO_PARA:
            paragraphs.append(current)
            current = [ln]
        else:
            current.append(ln)
    if current:
        paragraphs.append(current)
    return paragraphs
