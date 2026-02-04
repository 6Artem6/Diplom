"""
Text grouping: layout-first. font_size ≈ box_height. Same region_id only.

Lines/paragraphs merge only if relative font size difference <= 0.25:
  abs(h1 - h2) / max(h1, h2) <= 0.25  =>  max(h1,h2)/min(h1,h2) <= 1/(1-0.25) ≈ 1.333.
Horizontal → line: baseline align, font_size_ratio <= 1.333, gap <= 2.5*char_width.
Vertical → paragraph: font_size_ratio <= 1.333, vertical_gap <= 1.5*font_size.
No merge across region_id.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Font size: merge only if abs(h1-h2)/max(h1,h2) <= 0.25 (same line/paragraph)
FONT_SIZE_RELATIVE_DIFF_MAX = 0.25
# max(h1,h2)/min(h1,h2) <= 1/(1 - 0.25)
FONT_SIZE_RATIO_LINE = 1.0 / (1.0 - FONT_SIZE_RELATIVE_DIFF_MAX)  # ~1.333
FONT_SIZE_RATIO_PARA = FONT_SIZE_RATIO_LINE

BASELINE_Y_TOLERANCE = 0.5   # |baseline_y1 - baseline_y2| <= 0.5 * font_size
HORIZ_GAP_CHAR_FACTOR = 2.5  # horizontal_gap <= 2.5 * avg_char_width
VERT_GAP_FONT_FACTOR = 1.5   # vertical_gap <= 1.5 * font_size

# Strict: no gluing across cards/containers. Line ≠ paragraph ≠ card.
# Ужесточено: меньше объединения по вертикали и горизонтали (карточки по границам, не по словам).
HORIZ_GAP_CHAR_FACTOR_STRICT = 1.5   # same line only if gap <= 1.5 * avg_char_width
VERT_GAP_FONT_FACTOR_STRICT = 0.8    # same paragraph only if vertical_gap <= 0.8 * font_size
MAX_VERTICAL_GAP_PX_STRICT = 25      # same paragraph only if vertical_gap <= 25px
MAX_VERTICAL_GAP_PX_HARD = 120       # never merge if vertical gap > 120px
MAX_HORIZONTAL_GAP_PX_HARD = 100     # never merge into same line if gap_x > 100px (vertical closeness does NOT compensate)


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
    prefer_smallest_region: bool = True,
) -> List[Dict[str, Any]]:
    """
    Assign each text_box to one region_id (index). Add region_id to each box.
    Boxes with no overlap go to region_id = -1 (page fallback).
    When prefer_smallest_region=True and several regions overlap the box,
    choose the region with smallest area (most specific container; avoids losing nested elements).
    """
    result = []
    for i, box in enumerate(text_boxes):
        bx, by, bw, bh = box.get("x", 0), box.get("y", 0), box.get("w", 0), box.get("h", 0)
        box_area = max(1, bw * bh)
        best_rid = -1
        best_overlap = 0.0
        best_region_area: Optional[float] = None
        for ri, reg in enumerate(regions):
            rx, ry, rw, rh = reg["x"], reg["y"], reg["w"], reg["h"]
            ox1 = max(bx, rx)
            oy1 = max(by, ry)
            ox2 = min(bx + bw, rx + rw)
            oy2 = min(by + bh, ry + rh)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            overlap = (ox2 - ox1) * (oy2 - oy1) / box_area
            if overlap < min_overlap_ratio:
                continue
            reg_area = float(rw * rh)
            if prefer_smallest_region:
                # Prefer smallest containing region (most specific)
                if best_rid < 0 or reg_area < (best_region_area or 1e18) or (reg_area == best_region_area and overlap > best_overlap):
                    best_overlap = overlap
                    best_rid = ri
                    best_region_area = reg_area
            else:
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_rid = ri
        if best_overlap < min_overlap_ratio:
            best_rid = -1
        out = dict(box)
        out["region_id"] = best_rid
        out["box_index"] = i
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


def group_text_boxes_into_lines_strict(
    boxes_with_region: List[Dict[str, Any]],
    region_id: int,
) -> List[List[Dict[str, Any]]]:
    """
    Same as group_text_boxes_into_lines but with stricter horizontal gap.
    Forbidden to merge if gap > HORIZ_GAP_CHAR_FACTOR_STRICT * avg_char_width (no gluing across elements).
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
        gap_px = b.get("x", 0) - (last.get("x", 0) + last.get("w", 0))
        # Hard limit: vertical closeness does NOT compensate horizontal gap
        if gap_px > MAX_HORIZONTAL_GAP_PX_HARD:
            id1, id2 = last.get("box_index", "?"), b.get("box_index", "?")
            logger.info(
                "MERGE_REJECT reason=horizontal_gap gap_px=%d boxes=(%s,%s)",
                gap_px, id1, id2,
            )
            lines.append(current)
            current = [b]
            continue
        x_center_b = b.get("x", 0) + 0.5 * b.get("w", 0)
        x_center_ref = current[0].get("x", 0) + 0.5 * current[0].get("w", 0)
        if abs(x_center_b - x_center_ref) > MAX_HORIZONTAL_GAP_PX_HARD:
            id1, id2 = current[0].get("box_index", "?"), b.get("box_index", "?")
            logger.info(
                "MERGE_REJECT reason=horizontal_center_gap gap_px=%d boxes=(%s,%s)",
                int(abs(x_center_b - x_center_ref)), id1, id2,
            )
            lines.append(current)
            current = [b]
            continue
        avg_cw_ref = _avg_char_width(last)
        avg_cw_b = _avg_char_width(b)
        if gap_px > HORIZ_GAP_CHAR_FACTOR_STRICT * (avg_cw_ref + avg_cw_b) / 2.0:
            lines.append(current)
            current = [b]
        else:
            current.append(b)
    if current:
        lines.append(current)
    return lines


def group_lines_into_paragraphs_strict(
    lines: List[List[Dict[str, Any]]],
    font_size_fn: Any = _font_size,
) -> List[List[List[Dict[str, Any]]]]:
    """
    Same as group_lines_into_paragraphs but with stricter vertical gap.
    Forbidden to merge if vertical_gap > VERT_GAP_FONT_FACTOR_STRICT * font_size or > MAX_VERTICAL_GAP_PX_STRICT.
    Prevents gluing text across cards/containers.
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
        max_gap_by_font = VERT_GAP_FONT_FACTOR_STRICT * min(prev_fs, curr_fs)
        if gap > max_gap_by_font or gap > MAX_VERTICAL_GAP_PX_STRICT or gap > MAX_VERTICAL_GAP_PX_HARD or fs_ratio > FONT_SIZE_RATIO_PARA:
            logger.debug(
                "MERGE_REJECT reason=vertical_gap gap_px=%d boxes=(line_prev, line_curr)",
                gap,
            )
            paragraphs.append(current)
            current = [ln]
        else:
            current.append(ln)
    if current:
        paragraphs.append(current)
    return paragraphs


def _line_bbox(boxes: List[Dict[str, Any]]) -> tuple[int, int, int, int]:
    """Union bbox of boxes. Returns (x, y, w, h)."""
    if not boxes:
        return 0, 0, 0, 0
    xs = [b.get("x", 0) for b in boxes]
    ys = [b.get("y", 0) for b in boxes]
    x2s = [b.get("x", 0) + b.get("w", 0) for b in boxes]
    y2s = [b.get("y", 0) + b.get("h", 0) for b in boxes]
    x1, y1 = min(xs), min(ys)
    return x1, y1, max(x2s) - x1, max(y2s) - y1


def _overlap_ratio_line_region(
    line_xywh: tuple[int, int, int, int],
    reg: Dict[str, Any],
) -> float:
    """Intersection area / line area. 0 if no overlap."""
    lx, ly, lw, lh = line_xywh
    if lw <= 0 or lh <= 0:
        return 0.0
    rx, ry = reg.get("x", 0), reg.get("y", 0)
    rw, rh = reg.get("w", 0), reg.get("h", 0)
    ix1 = max(lx, rx)
    iy1 = max(ly, ry)
    ix2 = min(lx + lw, rx + rw)
    iy2 = min(ly + lh, ry + rh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / (lw * lh)


def group_text_boxes_into_lines_any_region(
    boxes_with_region: List[Dict[str, Any]],
    region_id_filter: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Group text_boxes into lines (same font size rule). If region_id_filter is None, use all boxes.
    Used to build global lines for one-line-one-region assignment.
    """
    if region_id_filter is not None:
        subset = [(b, i) for i, b in enumerate(boxes_with_region) if b.get("region_id") == region_id_filter]
    else:
        subset = [(b, i) for i, b in enumerate(boxes_with_region)]
    if not subset:
        return []
    sorted_boxes = sorted(subset, key=lambda bi: (_baseline_y(bi[0]), bi[0].get("x", 0)))
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


def ensure_one_line_one_region(
    boxes_with_region: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Ensure each visual line belongs to exactly one region.
    Build lines from all boxes; for each line assign to region with max overlap; update box region_id.
    Returns new list of boxes with updated region_id (no duplicate line across regions).
    """
    if not regions or not boxes_with_region:
        return list(boxes_with_region)
    lines_global = group_text_boxes_into_lines_any_region(boxes_with_region, region_id_filter=None)
    # Map each box (id) to its line index
    box_to_line_idx: Dict[int, int] = {}
    for li, line in enumerate(lines_global):
        for b in line:
            box_to_line_idx[id(b)] = li
    result = []
    for b in boxes_with_region:
        line_idx = box_to_line_idx.get(id(b), -1)
        if line_idx < 0:
            result.append({**b})
            continue
        line = lines_global[line_idx]
        line_xywh = _line_bbox(line)
        best_rid = -1
        best_overlap = 0.0
        for ri, reg in enumerate(regions):
            ov = _overlap_ratio_line_region(line_xywh, reg)
            if ov > best_overlap:
                best_overlap = ov
                best_rid = ri
        result.append({**b, "region_id": best_rid if best_rid >= 0 else b.get("region_id", -1)})
    return result


def group_lines_into_paragraphs(
    lines: List[List[Dict[str, Any]]],
    font_size_fn: Any = _font_size,
) -> List[List[List[Dict[str, Any]]]]:
    """
    Group lines into paragraphs. Same region assumed (call per region).
    Rule: font_size_ratio <= 1.333 (abs(h1-h2)/max <= 0.25), vertical_gap <= 1.5*font_size.
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
