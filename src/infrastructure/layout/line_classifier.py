"""
Line role: typographic segmentation (header / body / button / label).

Used ONLY as a tag after words_to_lines. Does NOT affect line aggregation or paragraph merge.
Body is the default; header and button are deviations that must be proven.
Button = line inside compact container; text color is NOT a filter (white/gray text must exist as Word).
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from .atoms import Line, LineRole, Word

logger = logging.getLogger(__name__)

# Local body height: exclude header-like (h > 1.3 * med_all)
HEADER_OUTLIER_RATIO = 1.3
# Header: typography only — tall, no background; font_weight optional reinforcing
HEADER_HEIGHT_BODY_RATIO = 1.3
HEADER_MAX_WORDS = 10
HEADER_NO_BG_RATIO = 0.5   # ratio_bg < this (no bg covering ≥50%)
HEADER_GAP_ABOVE_RATIO = 0.8
# Button = line inside compact container; bg + (text light/gray or aspect ≥ 1.5)
BUTTON_BG_RATIO_MIN = 0.8
BUTTON_HEIGHT_BODY_RATIO = 1.1
BUTTON_WIDTH_PAGE_RATIO = 0.6
BUTTON_MAX_WORDS = 5
BUTTON_MIN_WORDS = 1
BUTTON_MAX_CHARS = 30
BUTTON_ASPECT_MIN = 1.2  # line.w / line.h ≥ 1.2 or text light/gray (View details = valid)
BUTTON_GAP_ISOLATION_RATIO = 0.5  # gap above and below > this * local_body_h
BUTTON_MAX_HEIGHT_RATIO = 2.0     # if h >= 2*local_body_h → not button (large container)
LARGE_CONTAINER_HEIGHT_RATIO = 2.0  # container span >= this * body_h → not button
BODY_OVERLAP_NEAR_RATIO = 0.5     # neighbor with overlap and gap < this → inside card
BODY_NEIGHBOR_OVERLAP_X = 0.6     # neighbor with this overlap + body-like → not button
# Label
LABEL_MAX_WORDS = 3
LABEL_ASPECT_THIN = 4.0


def _median(values: List[float]) -> float:
    if not values:
        return 18.0
    vv = sorted(values)
    return float(vv[len(vv) // 2])


def _line_font_size_px(line: Line) -> float:
    """Estimated font size (px) for scale; not raw line.h."""
    v = getattr(line, "estimated_font_size_px", None)
    return float(v) if v is not None and v > 0 else float(line.h)


def _local_body_height(all_lines: List[Line]) -> float:
    """Median height of body-like lines (exclude header-like). Body defines scale."""
    if not all_lines:
        return 18.0
    heights = [float(l.h) for l in all_lines]
    med_all = _median(heights)
    body_heights = [h for h in heights if h <= HEADER_OUTLIER_RATIO * med_all]
    return _median(body_heights) if body_heights else med_all


def _is_button_like_line(line: Line) -> bool:
    """Button/badge line: exclude from body scale so body does not spread."""
    ratio_bg = _line_ratio_bg(line)
    wc = _line_word_count(line)
    tc = _line_total_char(line)
    return ratio_bg >= BUTTON_BG_RATIO_MIN and 1 <= wc <= BUTTON_MAX_WORDS and tc <= BUTTON_MAX_CHARS


def _local_body_font_size(all_lines: List[Line]) -> float:
    """Median estimated_font_size_px of body-like lines. Buttons/badges excluded."""
    if not all_lines:
        return 18.0
    sizes = [_line_font_size_px(l) for l in all_lines if not _is_button_like_line(l)]
    if not sizes:
        sizes = [_line_font_size_px(l) for l in all_lines]
    med_all = _median(sizes)
    body_sizes = [s for s in sizes if s <= HEADER_OUTLIER_RATIO * med_all]
    return _median(body_sizes) if body_sizes else med_all


def _line_font_weight(line: Line) -> float | None:
    """Median font_weight of words that have it. None if no word has font_weight."""
    weights = [w.font_weight for w in line.words if getattr(w, "font_weight", None) is not None]
    if not weights:
        return None
    return _median(weights)


def _median_body_weight(all_lines: List[Line], local_body_h: float) -> float | None:
    """Median font_weight of body-like lines (h <= header threshold). None if no data."""
    weights: List[float] = []
    for l in all_lines:
        if l.h > HEADER_OUTLIER_RATIO * local_body_h:
            continue
        w = _line_font_weight(l)
        if w is not None:
            weights.append(w)
    if not weights:
        return None
    return _median(weights)


def _gap_above(line: Line, sorted_by_y: List[Line]) -> float:
    """Vertical gap between this line and the line immediately above."""
    try:
        i = next(idx for idx, l in enumerate(sorted_by_y) if l is line or (l.y == line.y and l.x == line.x))
    except StopIteration:
        return 1e9
    if i <= 0:
        return 1e9
    prev = sorted_by_y[i - 1]
    return float(line.y - (prev.y + prev.h))


def _gap_below(line: Line, sorted_by_y: List[Line]) -> float:
    """Vertical gap between this line and the line immediately below."""
    try:
        i = next(idx for idx, l in enumerate(sorted_by_y) if l is line or (l.y == line.y and l.x == line.x))
    except StopIteration:
        return 1e9
    if i + 1 >= len(sorted_by_y):
        return 1e9
    nxt = sorted_by_y[i + 1]
    return float(nxt.y - (line.y + line.h))


def _horizontal_overlap_ratio(a: Line, b: Line) -> float:
    """Overlap length / min(width_a, width_b). 0 if no overlap."""
    ax1, ax2 = a.x, a.x + a.w
    bx1, bx2 = b.x, b.x + b.w
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    denom = min(a.w, b.w)
    return overlap / denom if denom > 0 else 0.0


def _vertical_gap(line: Line, other: Line) -> float:
    """Vertical gap between two lines (0 if overlapping)."""
    if other.y >= line.y + line.h:
        return float(other.y - (line.y + line.h))
    if line.y >= other.y + other.h:
        return float(line.y - (other.y + other.h))
    return 0.0


def _inside_card(line: Line, all_lines: List[Line], local_body_h: float) -> bool:
    """True if line has a neighbor with overlapping X and small vertical gap (inside card/list)."""
    for other in all_lines:
        if other is line or (other.y == line.y and other.x == line.x):
            continue
        overlap = _horizontal_overlap_ratio(line, other)
        if overlap < 0.5:
            continue
        gap = _vertical_gap(line, other)
        if gap < BODY_OVERLAP_NEAR_RATIO * local_body_h:
            return True
    return False


def _line_ratio_bg(line: Line) -> float:
    n = sum(1 for w in line.words if getattr(w, "has_background", False))
    return n / max(1, len(line.words))


def _line_word_count(line: Line) -> int:
    return len(line.words)


def _line_total_char(line: Line) -> int:
    return sum(len((w.text or "").strip()) for w in line.words)


def _line_has_light_or_gray(line: Line) -> bool:
    """True if any word has text_color_class light or gray (button text on colored bg)."""
    for w in line.words:
        c = getattr(w, "text_color_class", "dark") or "dark"
        if c in ("light", "gray"):
            return True
    return False


def _line_aspect(line: Line) -> float:
    """line.w / line.h. 0 if h is 0."""
    return (line.w / float(line.h)) if line.h and line.w else 0.0


def _is_likely_body(other: Line, local_body_h: float) -> bool:
    """Neighbor is body-like: not header (tall), not button (short+bg+short text)."""
    if other.h > HEADER_HEIGHT_BODY_RATIO * local_body_h:
        return False
    ratio_bg = _line_ratio_bg(other)
    wc = _line_word_count(other)
    tc = _line_total_char(other)
    if ratio_bg >= BUTTON_BG_RATIO_MIN and 1 <= wc <= BUTTON_MAX_WORDS and tc <= BUTTON_MAX_CHARS:
        return False
    return True


def _has_body_neighbor_same_x(
    line: Line, sorted_by_y: List[Line], all_lines: List[Line], local_body_h: float
) -> bool:
    """True if prev or next line has significant X-overlap and is body-like (→ not button)."""
    try:
        i = next(idx for idx, l in enumerate(sorted_by_y) if l is line or (l.y == line.y and l.x == line.x))
    except StopIteration:
        return False
    for idx in (i - 1, i + 1):
        if idx < 0 or idx >= len(sorted_by_y):
            continue
        other = sorted_by_y[idx]
        if _horizontal_overlap_ratio(line, other) < BODY_NEIGHBOR_OVERLAP_X:
            continue
        if _is_likely_body(other, local_body_h):
            return True
    return False


def _inside_large_container(line: Line, all_lines: List[Line], local_body_h: float) -> bool:
    """True if line is inside a background block with vertical span >= 2*local_body_h (card/navbar/form)."""
    if _line_ratio_bg(line) < BUTTON_BG_RATIO_MIN:
        return False
    # Collect lines that overlap in X with line, have bg, and are within small vertical gap
    line_y1, line_y2 = line.y, line.y + line.h
    group_y1, group_y2 = line_y1, line_y2
    for other in all_lines:
        if other is line or (other.y == line.y and other.x == line.x):
            continue
        if _horizontal_overlap_ratio(line, other) < 0.5:
            continue
        if _line_ratio_bg(other) < BUTTON_BG_RATIO_MIN:
            continue
        gap = _vertical_gap(line, other)
        if gap > BODY_OVERLAP_NEAR_RATIO * local_body_h:
            continue
        oy1, oy2 = other.y, other.y + other.h
        group_y1 = min(group_y1, oy1)
        group_y2 = max(group_y2, oy2)
    span = group_y2 - group_y1
    return span >= LARGE_CONTAINER_HEIGHT_RATIO * local_body_h


def classify_line(line: Line, all_lines: List[Line], page_width: float | None = None) -> LineRole:
    """
    Body by default. Header and button only if ALL conditions are met.
    Header = typography (tall, font_weight > body when available, no bg, gap above).
    Button = short CTA (small, bounded, isolated); NOT next to body with same X; NOT in large container.
    """
    words = line.words
    if not words:
        return "body"

    local_body_h = _local_body_height(all_lines)
    pw = page_width if page_width is not None and page_width > 0 else max((l.x + l.w for l in all_lines), default=1)
    sorted_by_y = sorted(all_lines, key=lambda l: (l.y, l.x))
    gap_above = _gap_above(line, sorted_by_y)
    gap_below = _gap_below(line, sorted_by_y)

    ratio_bg = _line_ratio_bg(line)
    word_count = _line_word_count(line)
    total_char = _line_total_char(line)
    local_body_fs = _local_body_font_size(all_lines)
    line_fs = _line_font_size_px(line)

    # HEADER only if ALL: tall (by font size), (font_weight > median_body when available), no background, gap above
    median_body_weight = _median_body_weight(all_lines, local_body_h)
    line_weight = _line_font_weight(line)
    header_height_ok = local_body_fs > 0 and line_fs >= HEADER_HEIGHT_BODY_RATIO * local_body_fs
    header_words_ok = word_count <= HEADER_MAX_WORDS
    header_no_bg = ratio_bg < HEADER_NO_BG_RATIO
    header_gap_ok = gap_above >= HEADER_GAP_ABOVE_RATIO * local_body_h
    # If font_weight available: header must be heavier than body. If not available, don't require (height+gap define).
    header_weight_ok = (median_body_weight is None or line_weight is None or
                        (line_weight is not None and median_body_weight is not None and line_weight > median_body_weight))

    if header_height_ok and header_words_ok and header_no_bg and header_gap_ok and header_weight_ok:
        return "header"

    # BUTTON only if ALL: short (by font size), bounded, bg, isolated; NOT next to body; NOT large container
    if local_body_fs > 0 and line_fs >= BUTTON_MAX_HEIGHT_RATIO * local_body_fs:
        pass  # large line → never button
    elif _inside_card(line, all_lines, local_body_h):
        pass  # inside card/list → not button
    elif _has_body_neighbor_same_x(line, sorted_by_y, all_lines, local_body_h):
        pass  # next to body with same X-range → not button
    elif _inside_large_container(line, all_lines, local_body_h):
        pass  # inside large background container (card/navbar/form) → not button
    elif (
        ratio_bg >= BUTTON_BG_RATIO_MIN
        and line_fs <= BUTTON_HEIGHT_BODY_RATIO * local_body_fs
        and line.w <= BUTTON_WIDTH_PAGE_RATIO * pw
        and BUTTON_MIN_WORDS <= word_count <= BUTTON_MAX_WORDS
        and total_char <= BUTTON_MAX_CHARS
        and gap_above >= BUTTON_GAP_ISOLATION_RATIO * local_body_h
        and gap_below >= BUTTON_GAP_ISOLATION_RATIO * local_body_h
        and (_line_has_light_or_gray(line) or _line_aspect(line) >= BUTTON_ASPECT_MIN)
    ):
        return "button"

    # LABEL: very short, narrow (e.g. "Name", "Description")
    if word_count <= LABEL_MAX_WORDS and line.w > 0 and line.h > 0:
        aspect = line.w / float(line.h)
        if aspect <= LABEL_ASPECT_THIN:
            return "label"

    return "body"


def classify_line_with_reason(
    line: Line, all_lines: List[Line], page_width: float | None = None
) -> Tuple[LineRole, str]:
    """
    Same as classify_line but returns (role, reason) for diagnostics.
    Reason is typographic/geometric only; never "only background" (that would be a bug).
    """
    words = line.words
    if not words:
        return "body", "default"

    local_body_h = _local_body_height(all_lines)
    pw = page_width if page_width is not None and page_width > 0 else max((l.x + l.w for l in all_lines), default=1)
    sorted_by_y = sorted(all_lines, key=lambda l: (l.y, l.x))
    gap_above = _gap_above(line, sorted_by_y)
    gap_below = _gap_below(line, sorted_by_y)
    ratio_bg = _line_ratio_bg(line)
    word_count = _line_word_count(line)
    total_char = _line_total_char(line)

    role = classify_line(line, all_lines, page_width=page_width)

    if role == "body":
        return "body", "default"
    if role == "label":
        return "label", "short_narrow"

    local_body_fs = _local_body_font_size(all_lines)
    line_fs = _line_font_size_px(line)
    if role == "header":
        reasons: List[str] = []
        if local_body_fs > 0 and line_fs >= HEADER_HEIGHT_BODY_RATIO * local_body_fs:
            reasons.append(f"font_size>={HEADER_HEIGHT_BODY_RATIO}*body_fs")
        if word_count <= HEADER_MAX_WORDS:
            reasons.append("word_count<=10")
        if ratio_bg < HEADER_NO_BG_RATIO:
            reasons.append("no_bg")
        if gap_above >= HEADER_GAP_ABOVE_RATIO * local_body_h:
            reasons.append("gap_above>=0.8*body_h")
        median_bw = _median_body_weight(all_lines, local_body_h)
        lw = _line_font_weight(line)
        if median_bw is not None and lw is not None and lw > median_bw:
            reasons.append("font_weight>median_body")
        reason_str = "; ".join(reasons) if reasons else "typography"
        logger.debug("Line role=header: %s", reason_str)
        return "header", reason_str

    if role == "button":
        reasons = []
        if BUTTON_MIN_WORDS <= word_count <= BUTTON_MAX_WORDS:
            reasons.append(f"word_count={word_count}")
        if total_char <= BUTTON_MAX_CHARS:
            reasons.append(f"chars={total_char}")
        if line_fs <= BUTTON_HEIGHT_BODY_RATIO * local_body_fs:
            reasons.append("font_size<=1.1*body_fs")
        if line.w <= BUTTON_WIDTH_PAGE_RATIO * pw:
            reasons.append("w<=0.6*page")
        if gap_above >= BUTTON_GAP_ISOLATION_RATIO * local_body_h:
            reasons.append("gap_above>=0.5*body_h")
        if gap_below >= BUTTON_GAP_ISOLATION_RATIO * local_body_h:
            reasons.append("gap_below>=0.5*body_h")
        if not _inside_card(line, all_lines, local_body_h):
            reasons.append("not_inside_card")
        if not _has_body_neighbor_same_x(line, sorted_by_y, all_lines, local_body_h):
            reasons.append("no_body_neighbor_same_x")
        if not _inside_large_container(line, all_lines, local_body_h):
            reasons.append("not_large_container")
        if _line_has_light_or_gray(line):
            reasons.append("text_light_or_gray")
        elif _line_aspect(line) >= BUTTON_ASPECT_MIN:
            reasons.append("aspect>=1.5")
        reason_str = "; ".join(reasons) if reasons else "short_bounded_isolated"
        logger.debug("Line role=button: %s", reason_str)
        return "button", reason_str

    return role, "default"
