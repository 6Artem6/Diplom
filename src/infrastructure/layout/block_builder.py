from __future__ import annotations

from typing import List, Optional, Sequence

from .atoms import BlockType, Line, TextBlock, HorizontalRule

# Spec §3: |x0_prev - x0_curr| <= 0.3 * char_width
ALIGN_CHAR_WIDTH_RATIO = 0.3
WIDTH_RATIO_MAX = 3.0
# §3A: gap <= 1.4 * min(prev_line_height, curr_line_height); §3 strict: gap > 2.0 * min → FORBIDDEN
GAP_FACTOR_MAX = 1.4
GAP_FORBID_FACTOR = 2.0
# §3B: height_ratio <= 1.25 (different font size → no merge; header and body stay separate)
HEIGHT_RATIO_MAX = 1.25

# BODY-paragraph: merge body lines using LOCAL body height; horizontal overlap required
BODY_PARAGRAPH_GAP_FACTOR = 1.2    # vertical_gap <= 1.2 * local_body_h
BODY_HEIGHT_TOLERANCE_RATIO = 0.1  # abs(prev.h - curr.h) <= 0.1 * local_body_h
BODY_OVERLAP_X_MIN = 0.6           # horizontal overlap_ratio >= 0.6 (same column/block)


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
    Lines → blocks. Merge ONLY body with body.

    - prev.role == body AND curr.role == body AND vertical_gap <= 1.2*local_body_h
      AND |h1-h2| <= 0.1*local_body_h AND overlap_x >= 0.6 → merge.
    - Header (role=header) and button (role=button) never participate in body-merge;
      each forms its own block; they do not block body-merge when beside.
    - Divider between lines → no merge.
    - _is_button_line = role only (no has_background fallback).
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

        # Merge ONLY body with body. Button and header never participate; they don't block when beside (we flush).
        prev_role = getattr(prev, "role", None)
        ln_role = getattr(ln, "role", None)
        local_body_h = _local_body_height(sorted_lines)
        overlap_x = _horizontal_overlap_ratio(prev, ln)
        if prev_role == "body" and ln_role == "body":
            if (
                gap <= BODY_PARAGRAPH_GAP_FACTOR * local_body_h
                and abs(prev.h - ln.h) <= BODY_HEIGHT_TOLERANCE_RATIO * local_body_h
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

