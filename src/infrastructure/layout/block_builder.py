from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .atoms import BlockType, Line, TextBlock, HorizontalRule

# Spec §3: |x0_prev - x0_curr| <= 0.3 * char_width
ALIGN_CHAR_WIDTH_RATIO = 0.3
WIDTH_RATIO_MAX = 3.0
# §3A: gap <= 1.4 * min(prev_line_height, curr_line_height); §3 strict: gap > 2.0 * min → FORBIDDEN
GAP_FACTOR_MAX = 1.4
GAP_FORBID_FACTOR = 2.0
# §3B: height_ratio <= 1.25 (different font size → no merge; header and body stay separate)
HEIGHT_RATIO_MAX = 1.25

# BODY-paragraph: merge by estimated_font_size (≤1.5×), vertical_gap ≤ 2–3 char heights
BODY_PARAGRAPH_GAP_FONT_FACTOR = 2.5   # vertical_gap <= 2.5 * font_size (2–3 char heights)
BODY_FONT_SIZE_RATIO_MAX = 1.5         # estimated_font_size ratio ≤ 1.5×
BODY_OVERLAP_X_MIN = 0.6

# Region merge: merge all blocks inside text_region when gap ≤ this × median_font_size
REGION_MERGE_GAP_FONT_FACTOR = 2.5


def _median(values: List[float]) -> float:
    if not values:
        return 18.0
    vv = sorted(values)
    return float(vv[len(vv) // 2])

def _median_line_height(lines: List[Line]) -> float:
    """Median of line heights for stable gap threshold (use medians, not means)."""
    if not lines:
        return 18.0
    heights = [l.h for l in lines]
    heights.sort()
    n = len(heights)
    return float(heights[n // 2])


def _local_body_height(lines: List[Line]) -> float:
    """Median height of body lines only. Body defines scale; header/button are excluded."""
    body_heights = [l.h for l in lines if getattr(l, "role", None) == "body"]
    if not body_heights:
        return _median_line_height(lines)
    body_heights.sort()
    n = len(body_heights)
    return float(body_heights[n // 2])


def _line_font_size_px(line: Line) -> float:
    """Estimated font size (px) for merge; not raw line.h."""
    v = getattr(line, "estimated_font_size_px", None)
    return float(v) if v is not None and v > 0 else float(line.h)


def _local_body_font_size(lines: List[Line]) -> float:
    """Median estimated_font_size_px of body lines."""
    body_sizes = [_line_font_size_px(l) for l in lines if getattr(l, "role", None) == "body"]
    if not body_sizes:
        return _median([int(_line_font_size_px(l)) for l in lines])
    body_sizes.sort()
    return float(body_sizes[len(body_sizes) // 2])


def _horizontal_overlap_ratio(prev: Line, curr: Line) -> float:
    """Overlap length / min(prev.w, curr.w). 0 if no overlap."""
    ox1 = max(prev.x, curr.x)
    ox2 = min(prev.x + prev.w, curr.x + curr.w)
    overlap = max(0, ox2 - ox1)
    denom = min(prev.w, curr.w)
    return overlap / denom if denom > 0 else 0.0


def _has_divider_between(
    prev_bottom: float,
    curr_top: float,
    dividers: Sequence[HorizontalRule],
) -> bool:
    """True if any horizontal rule overlaps the vertical gap between prev line bottom and curr line top."""
    for d in dividers:
        if d.y_min < curr_top and d.y_max > prev_bottom:
            return True
    return False


def lines_to_blocks(
    lines: List[Line],
    gap_factor: float = 1.4,
    align_dx: Optional[int] = None,
    char_width_px: Optional[float] = None,
) -> List[TextBlock]:
    """
    Group lines into vertical TextBlocks (paragraphs). Tesseract/PDF-style.

    Heuristics:
    - gap <= gap_factor * median_line_height (medians for robustness)
    - left alignment: |x0_prev - x0_curr| < 0.2 * char_width (or align_dx if given)
    - width_ratio <= 3.0 (paragraph with varying line lengths)
    - Provenance: each block keeps list of lines, each line keeps list of words.
    """
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda l: l.y)
    median_h = _median_line_height(sorted_lines)
    # Left alignment: spec |x0_prev - x0_curr| < 0.2 * char_width
    if align_dx is not None:
        dx = align_dx
    elif char_width_px is not None:
        dx = max(1, int(ALIGN_CHAR_WIDTH_RATIO * char_width_px))
    else:
        dx = 24

    blocks: List[TextBlock] = []
    cur: List[Line] = [sorted_lines[0]]
    bx1, by1 = sorted_lines[0].x, sorted_lines[0].y
    bx2, by2 = sorted_lines[0].x + sorted_lines[0].w, sorted_lines[0].y + sorted_lines[0].h

    for ln in sorted_lines[1:]:
        prev = cur[-1]
        gap = ln.y - (prev.y + prev.h)
        same_x = abs(ln.x - prev.x) <= dx
        width_ratio = max(prev.w, ln.w) / max(1, min(prev.w, ln.w))
        same_block = (
            gap >= 0
            and gap <= gap_factor * median_h
            and same_x
            and width_ratio <= WIDTH_RATIO_MAX
        )
        if same_block:
            cur.append(ln)
            bx1 = min(bx1, ln.x)
            by1 = min(by1, ln.y)
            bx2 = max(bx2, ln.x + ln.w)
            by2 = max(by2, ln.y + ln.h)
        else:
            blocks.append(TextBlock(lines=list(cur), x=bx1, y=by1, w=bx2 - bx1, h=by2 - by1))
            cur = [ln]
            bx1, by1 = ln.x, ln.y
            bx2, by2 = ln.x + ln.w, ln.y + ln.h

    blocks.append(TextBlock(lines=list(cur), x=bx1, y=by1, w=bx2 - bx1, h=by2 - by1))
    return blocks


def lines_to_blocks_with_headers(
    lines: List[Line],
    char_width_px: float,
    dividers: Optional[Sequence[HorizontalRule]] = None,
) -> List[TextBlock]:
    """
    Lines → blocks. TEXT SPLIT BY ROLE HERE: header/body/button never merge; cards stay separate.

    - Body+body merge only: vertical_gap <= 2.5×font_size, overlap_x >= 0.6, font_ratio <= 1.5×, no divider.
    - Header and button each form their own block (no merge with body or each other).
    - Divider between lines → no merge. Role from line_classifier (not has_background fallback).
    """
    if not lines:
        return []

    rules = dividers or ()
    sorted_lines = sorted(lines, key=lambda l: l.y)
    align_dx = max(1, int(ALIGN_CHAR_WIDTH_RATIO * char_width_px))

    def _is_button_line(line: Line) -> bool:
        # Role only. Classification does not affect merge; button = tag from classifier.
        # No fallback to has_background — that would break body-merge.
        return getattr(line, "role", None) == "button"

    blocks: List[TextBlock] = []
    cur: List[Line] = [sorted_lines[0]]
    bx1, by1 = sorted_lines[0].x, sorted_lines[0].y
    bx2 = sorted_lines[0].x + sorted_lines[0].w
    by2 = sorted_lines[0].y + sorted_lines[0].h

    def flush(cur_lines: List[Line], is_header_block: bool) -> None:
        if not cur_lines:
            return
        x1 = min(l.x for l in cur_lines)
        y1 = min(l.y for l in cur_lines)
        x2 = max(l.x + l.w for l in cur_lines)
        y2 = max(l.y + l.h for l in cur_lines)
        med_h = _median_line_height(cur_lines)
        block_type: BlockType = "header" if is_header_block else ("standalone" if len(cur_lines) == 1 else "paragraph")
        stats = {"median_line_height": med_h, "font_size_class": "header" if is_header_block else "normal"}
        blocks.append(
            TextBlock(
                lines=list(cur_lines),
                x=x1,
                y=y1,
                w=x2 - x1,
                h=y2 - y1,
                block_type=block_type,
                stats=stats,
            )
        )

    for ln in sorted_lines[1:]:
        prev = cur[-1]
        gap = ln.y - (prev.y + prev.h)
        min_h = min(prev.h, ln.h)
        # §3 strict: gap > 2.0 * min_line_height → merge FORBIDDEN
        if gap > GAP_FORBID_FACTOR * min_h:
            flush(cur, prev.is_header)
            cur = [ln]
            bx1, by1 = ln.x, ln.y
            bx2, by2 = ln.x + ln.w, ln.y + ln.h
            continue
        # §2: headers never merge with others
        if ln.is_header:
            flush(cur, prev.is_header)
            cur = [ln]
            bx1, by1 = ln.x, ln.y
            bx2, by2 = ln.x + ln.w, ln.y + ln.h
            continue
        if prev.is_header:
            flush(cur, True)
            cur = [ln]
            bx1, by1 = ln.x, ln.y
            bx2, by2 = ln.x + ln.w, ln.y + ln.h
            continue
        # Button-lines → separate blocks (never merge with header/body)
        if _is_button_line(ln):
            flush(cur, False)
            cur = [ln]
            bx1, by1 = ln.x, ln.y
            bx2, by2 = ln.x + ln.w, ln.y + ln.h
            continue
        if _is_button_line(prev):
            flush(cur, False)
            cur = [ln]
            bx1, by1 = ln.x, ln.y
            bx2, by2 = ln.x + ln.w, ln.y + ln.h
            continue
        # §3D: divider between lines → no merge
        if _has_divider_between(prev.y + prev.h, ln.y, rules):
            flush(cur, False)
            cur = [ln]
            bx1, by1 = ln.x, ln.y
            bx2, by2 = ln.x + ln.w, ln.y + ln.h
            continue

        # Merge ONLY body with body. Use estimated_font_size (≤1.5×), vertical_gap ≤ 2–3 char heights.
        prev_role = getattr(prev, "role", None)
        ln_role = getattr(ln, "role", None)
        local_font_size = _local_body_font_size(sorted_lines)
        overlap_x = _horizontal_overlap_ratio(prev, ln)
        prev_fs = _line_font_size_px(prev)
        ln_fs = _line_font_size_px(ln)
        font_ratio = max(prev_fs, ln_fs) / max(1.0, min(prev_fs, ln_fs))
        if prev_role == "body" and ln_role == "body":
            if (
                gap <= BODY_PARAGRAPH_GAP_FONT_FACTOR * min(prev_fs, ln_fs)
                and font_ratio <= BODY_FONT_SIZE_RATIO_MAX
                and overlap_x >= BODY_OVERLAP_X_MIN
                and abs(ln.x - prev.x) <= align_dx
                and max(prev.w, ln.w) / max(1, min(prev.w, ln.w)) <= WIDTH_RATIO_MAX
            ):
                cur.append(ln)
                bx1 = min(bx1, ln.x)
                by1 = min(by1, ln.y)
                bx2 = max(bx2, ln.x + ln.w)
                by2 = max(by2, ln.y + ln.h)
                continue

        # Not body-body → new block (header/button/label/other never merge with body or each other)
        flush(cur, False)
        cur = [ln]
        bx1, by1 = ln.x, ln.y
        bx2, by2 = ln.x + ln.w, ln.y + ln.h

    flush(cur, cur[0].is_header if cur else False)
    return blocks


def lines_to_blocks_geometry_only(
    lines: List[Line],
    char_width_px: float,
    dividers: Optional[Sequence[HorizontalRule]] = None,
) -> List[TextBlock]:
    """
    Lines → blocks by GEOMETRY only. Role (button/header/body) is NOT used for merge.
    Merge consecutive lines if: vertical_gap ≤ 2.5×font_size, font_size ratio ≤ 1.5×,
    overlap_x ≥ 0.6, no horizontal_rule. Block type is set from line roles after (by caller).
    """
    if not lines:
        return []
    rules = dividers or ()
    sorted_lines = sorted(lines, key=lambda l: l.y)
    align_dx = max(1, int(ALIGN_CHAR_WIDTH_RATIO * char_width_px))
    median_font = _median([_line_font_size_px(l) for l in sorted_lines])

    blocks: List[TextBlock] = []
    cur: List[Line] = [sorted_lines[0]]
    bx1, by1 = sorted_lines[0].x, sorted_lines[0].y
    bx2 = sorted_lines[0].x + sorted_lines[0].w
    by2 = sorted_lines[0].y + sorted_lines[0].h

    def flush(cur_lines: List[Line]) -> None:
        if not cur_lines:
            return
        x1 = min(l.x for l in cur_lines)
        y1 = min(l.y for l in cur_lines)
        x2 = max(l.x + l.w for l in cur_lines)
        y2 = max(l.y + l.h for l in cur_lines)
        med_h = _median_line_height(cur_lines)
        blocks.append(
            TextBlock(
                lines=list(cur_lines),
                x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                block_type="paragraph",
                stats={"median_line_height": med_h},
            )
        )

    for ln in sorted_lines[1:]:
        prev = cur[-1]
        gap = ln.y - (prev.y + prev.h)
        prev_fs = _line_font_size_px(prev)
        ln_fs = _line_font_size_px(ln)
        min_fs = min(prev_fs, ln_fs)
        font_ratio = max(prev_fs, ln_fs) / max(1.0, min_fs)
        overlap_x = _horizontal_overlap_ratio(prev, ln)
        if (
            gap >= 0
            and gap <= BODY_PARAGRAPH_GAP_FONT_FACTOR * min_fs
            and font_ratio <= BODY_FONT_SIZE_RATIO_MAX
            and overlap_x >= BODY_OVERLAP_X_MIN
            and abs(ln.x - prev.x) <= align_dx
            and max(prev.w, ln.w) / max(1, min(prev.w, ln.w)) <= WIDTH_RATIO_MAX
            and not _has_divider_between(prev.y + prev.h, ln.y, rules)
        ):
            cur.append(ln)
            bx1 = min(bx1, ln.x)
            by1 = min(by1, ln.y)
            bx2 = max(bx2, ln.x + ln.w)
            by2 = max(by2, ln.y + ln.h)
        else:
            flush(cur)
            cur = [ln]
            bx1, by1 = ln.x, ln.y
            bx2, by2 = ln.x + ln.w, ln.y + ln.h
    flush(cur)
    return blocks


def merge_blocks_inside_region(
    region_bbox: Tuple[int, int, int, int],
    blocks: List[TextBlock],
    median_font_size_px: float,
    dividers: Optional[Sequence[HorizontalRule]] = None,
) -> List[TextBlock]:
    """
    Merge blocks inside one text_region when there is no horizontal_rule and
    vertical_gap ≤ 2.5 × median_font_size between consecutive blocks.
    Resulting block bbox is clipped to region. Region = (rx, ry, rw, rh).
    """
    if not blocks:
        return []
    rx, ry, rw, rh = region_bbox
    rules = dividers or ()
    sorted_blocks = sorted(blocks, key=lambda b: b.y)
    max_gap = REGION_MERGE_GAP_FONT_FACTOR * median_font_size_px
    merged: List[TextBlock] = []
    cur_lines: List[Line] = []
    cur_x1, cur_y1, cur_x2, cur_y2 = 0, 0, 0, 0

    def flush_merged() -> None:
        if not cur_lines:
            return
        x1 = min(l.x for l in cur_lines)
        y1 = min(l.y for l in cur_lines)
        x2 = max(l.x + l.w for l in cur_lines)
        y2 = max(l.y + l.h for l in cur_lines)
        x1 = max(x1, rx)
        y1 = max(y1, ry)
        x2 = min(x2, rx + rw)
        y2 = min(y2, ry + rh)
        if x2 <= x1 or y2 <= y1:
            return
        merged.append(
            TextBlock(
                lines=cur_lines,
                x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                block_type="paragraph",
                stats={},
            )
        )

    for b in sorted_blocks:
        if not b.lines:
            continue
        if not cur_lines:
            cur_lines = list(b.lines)
            cur_x1 = min(l.x for l in cur_lines)
            cur_y1 = min(l.y for l in cur_lines)
            cur_x2 = max(l.x + l.w for l in cur_lines)
            cur_y2 = max(l.y + l.h for l in cur_lines)
            continue
        prev_bottom = cur_y2
        curr_top = b.y
        gap = curr_top - prev_bottom
        has_rule = _has_divider_between(prev_bottom, curr_top, rules)
        if not has_rule and gap <= max_gap:
            cur_lines.extend(b.lines)
            cur_x1 = min(cur_x1, b.x)
            cur_y1 = min(cur_y1, b.y)
            cur_x2 = max(cur_x2, b.x + b.w)
            cur_y2 = max(cur_y2, b.y + b.h)
        else:
            flush_merged()
            cur_lines = list(b.lines)
            cur_x1 = min(l.x for l in cur_lines)
            cur_y1 = min(l.y for l in cur_lines)
            cur_x2 = max(l.x + l.w for l in cur_lines)
            cur_y2 = max(l.y + l.h for l in cur_lines)
    flush_merged()
    return merged


def lines_to_blocks_hdbscan(
    lines: List[Line],
    char_width_px: float = 10.0,
    gap_factor: float = 1.4,
    min_cluster_size: int = 1,
    min_samples: int = 1,
) -> List[TextBlock]:
    """
    Alternative: cluster lines with HDBSCAN, then merge lines per cluster into blocks.

    Points: (centroid_x, centroid_y, line_height). Distance = normalized vertical
    + normalized horizontal. min_cluster_size=1–2, min_samples=1 for small paragraphs.
    """
    try:
        import hdbscan
        import numpy as np
    except ImportError:
        # Fallback to heuristic if hdbscan not installed
        return lines_to_blocks(lines, gap_factor=gap_factor, char_width_px=char_width_px)

    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda l: l.y)
    # Features: centroid_x, centroid_y, line_height (normalize for distance)
    centroids_x = [l.x + l.w / 2.0 for l in sorted_lines]
    centroids_y = [l.y + l.h / 2.0 for l in sorted_lines]
    heights = [float(l.h) for l in sorted_lines]
    scale_y = max(1.0, max(centroids_y) - min(centroids_y)) if len(centroids_y) > 1 else 1.0
    scale_x = max(1.0, max(centroids_x) - min(centroids_x)) if len(centroids_x) > 1 else 1.0
    scale_h = max(1.0, max(heights))
    X = np.array([
        [cx / scale_x, cy / scale_y, h / scale_h]
        for cx, cy, h in zip(centroids_x, centroids_y, heights)
    ])
    if X.shape[0] < 2:
        return [TextBlock(lines=sorted_lines, x=sorted_lines[0].x, y=sorted_lines[0].y, w=sorted_lines[0].w, h=sorted_lines[0].h)]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(X)
    blocks: List[TextBlock] = []
    for label in set(labels):
        cluster_lines = [sorted_lines[i] for i in range(len(sorted_lines)) if labels[i] == label]
        if not cluster_lines:
            continue
        # noise (label -1): one block per line; else one block per cluster
        if label == -1:
            for l in cluster_lines:
                blocks.append(TextBlock(lines=[l], x=l.x, y=l.y, w=l.w, h=l.h))
        else:
            x1 = min(l.x for l in cluster_lines)
            y1 = min(l.y for l in cluster_lines)
            x2 = max(l.x + l.w for l in cluster_lines)
            y2 = max(l.y + l.h for l in cluster_lines)
            blocks.append(TextBlock(lines=cluster_lines, x=x1, y=y1, w=x2 - x1, h=y2 - y1))
    return blocks

