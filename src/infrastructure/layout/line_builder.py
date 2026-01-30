"""
Words → Lines for web pages.

Y-bands (local median height) + X-islands (gap, height, color, rules).
Words with has_background do not merge with words without; text color compatible; rules break.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .atoms import HorizontalRule, Line, VerticalRule, Word

# Line grouping: |y_center − y_center_ref| ≤ 2×font_size_px, height_ratio ≤ 2
Y_CENTER_TOLERANCE_FONT_FACTOR = 2.0   # band: allow word if |y_center - band_y| ≤ 2×font_size
FONT_SIZE_RATIO_MAX = 2.0
# Horizontal gap: gap_x ≤ max(3×font_size_px, 2×char_width)
X_GAP_FONT_FACTOR = 3.0
X_GAP_CHAR_FACTOR = 2.0


def _median(values: List[int]) -> float:
    if not values:
        return 18.0
    vv = sorted(values)
    return float(vv[len(vv) // 2])


def _word_font_size_px(w: Word) -> float:
    """Estimated font size (px). Use for merge, not raw bbox h."""
    v = getattr(w, "estimated_font_size_px", None)
    return float(v) if v is not None and v > 0 else float(w.h)


def _word_baseline_y(w: Word) -> float:
    """Approximate baseline Y (cap height ~0.75 * font_size from top)."""
    fs = _word_font_size_px(w)
    return float(w.y) + fs * 0.75


def _word_center_y(w: Word) -> float:
    return float(w.y + w.h / 2.0)


def _word_avg_char_width(w: Word) -> float:
    """word.w / max(len(text), 1). Layout coords."""
    n = max(1, len((w.text or "").strip()))
    return float(w.w) / n


def _estimate_char_width(words: List[Word]) -> Optional[float]:
    """Median char width from word width / len(text). Layout coords."""
    vals: List[float] = []
    for w in words:
        t = (w.text or "").strip()
        if not t:
            continue
        n = len(t)
        if n <= 0:
            continue
        vals.append(w.w / float(n))
    if not vals:
        return None
    vals.sort()
    return float(vals[len(vals) // 2])


def _height_ratio(h1: float, h2: float) -> float:
    if h1 <= 0 or h2 <= 0:
        return 0.0
    return max(h1, h2) / min(h1, h2)


def _text_color_compatible(a: Word, b: Word) -> bool:
    """light↔light, gray↔gray, dark↔dark/gray. For button (both have bg) allow same or light/gray."""
    ca = getattr(a, "text_color_class", "dark") or "dark"
    cb = getattr(b, "text_color_class", "dark") or "dark"
    if ca == cb:
        return True
    if (ca == "dark" and cb == "gray") or (ca == "gray" and cb == "dark"):
        return True
    return False


def _has_rule_between_words(
    w1: Word,
    w2: Word,
    h_rules: Sequence[HorizontalRule],
    v_rules: Sequence[VerticalRule],
) -> bool:
    """True if any horizontal or vertical rule lies in the gap between the two words."""
    left = w1 if w1.x <= w2.x else w2
    right = w2 if w1.x <= w2.x else w1
    gap_x_low = left.x + left.w
    gap_x_high = right.x
    if gap_x_low >= gap_x_high:
        return False
    gap_y_low = min(w1.y, w2.y)
    gap_y_high = max(w1.y + w1.h, w2.y + w2.h)
    for r in v_rules:
        if r.x_min < gap_x_high and r.x_max > gap_x_low and r.y_min < gap_y_high and r.y_max > gap_y_low:
            return True
    for r in h_rules:
        if r.y_min < gap_y_high and r.y_max > gap_y_low and r.x_min < gap_x_high and r.x_max > gap_x_low:
            return True
    return False


def _group_into_local_y_bands(words: List[Word]) -> List[List[Word]]:
    """
    Y-bands by y_center: |y_center − y_center_ref| ≤ 2×font_size_px, height_ratio ≤ 2.
    Font size = median(word.h) per band; raw bbox h not used for membership.
    """
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (_word_center_y(w), w.x))
    bands: List[List[Word]] = []
    band_center_y: List[float] = []
    band_font_sizes: List[float] = []

    for w in sorted_words:
        cy = _word_center_y(w)
        fs = _word_font_size_px(w)
        best_idx: Optional[int] = None
        best_dist = 1e18
        for i, band_cy in enumerate(band_center_y):
            band_fs = band_font_sizes[i]
            font_ok = _height_ratio(fs, band_fs) <= FONT_SIZE_RATIO_MAX
            tol = Y_CENTER_TOLERANCE_FONT_FACTOR * max(fs, band_fs)
            dist = abs(cy - band_cy)
            if font_ok and dist <= tol and dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is None:
            bands.append([w])
            band_center_y.append(cy)
            band_font_sizes.append(fs)
            continue

        bands[best_idx].append(w)
        band_center_y[best_idx] = _median([int(_word_center_y(ww)) for ww in bands[best_idx]])
        band_font_sizes[best_idx] = _median([int(_word_font_size_px(ww)) for ww in bands[best_idx]])

    bands_with_cy = list(zip(bands, band_center_y))
    bands_with_cy.sort(key=lambda bc: bc[1])
    return [b for (b, _c) in bands_with_cy]


def _split_band_into_x_islands(
    band_words: List[Word],
    char_width_px: Optional[float],
    band_med_h: float,
    horizontal_rules: Sequence[HorizontalRule],
    vertical_rules: Sequence[VerticalRule],
) -> List[List[Word]]:
    """
    Inside one Y-band, split by X-gap and compatibility.
    New island if: gap_x > max(1.5*h_band, 2.5*char_width),
    or height_ratio > 2, or text color incompatible, or rule between, or has_background mixed.
    """
    if not band_words:
        return []

    ws = sorted(band_words, key=lambda w: w.x)
    islands: List[List[Word]] = []
    cur: List[Word] = [ws[0]]
    last_x2 = ws[0].x + ws[0].w

    for w in ws[1:]:
        gap_x = w.x - last_x2
        cur_font_size = _median([int(_word_font_size_px(ww)) for ww in cur])
        cur_has_bg = any(getattr(ww, "has_background", False) for ww in cur)
        w_has_bg = getattr(w, "has_background", False)

        # Words with has_background do not merge with words without
        if cur_has_bg != w_has_bg:
            islands.append(cur)
            cur = [w]
            last_x2 = w.x + w.w
            continue

        # gap_x ≤ max(3×font_size_px, 2×char_width)
        prev_w = cur[-1]
        avg_cw = (_word_avg_char_width(prev_w) + _word_avg_char_width(w)) / 2.0
        cw = float(char_width_px) if char_width_px is not None else avg_cw
        max_gap_x = max(X_GAP_FONT_FACTOR * cur_font_size, X_GAP_CHAR_FACTOR * cw)
        if gap_x > max_gap_x:
            islands.append(cur)
            cur = [w]
            last_x2 = w.x + w.w
            continue

        # estimated_font_size ratio ≤ 2
        if _height_ratio(_word_font_size_px(w), cur_font_size) > FONT_SIZE_RATIO_MAX:
            islands.append(cur)
            cur = [w]
            last_x2 = w.x + w.w
            continue

        # Text color compatible with last word
        if not _text_color_compatible(cur[-1], w):
            islands.append(cur)
            cur = [w]
            last_x2 = w.x + w.w
            continue

        # No separator between last word and w
        if _has_rule_between_words(cur[-1], w, horizontal_rules, vertical_rules):
            islands.append(cur)
            cur = [w]
            last_x2 = w.x + w.w
            continue

        cur.append(w)
        last_x2 = max(last_x2, w.x + w.w)

    islands.append(cur)
    return [sorted(isl, key=lambda w: w.x) for isl in islands]


def words_to_lines(
    words: List[Word],
    line_dy_px: int = 18,
    char_width_px: Optional[float] = None,
    horizontal_rules: Optional[Sequence[HorizontalRule]] = None,
    vertical_rules: Optional[Sequence[VerticalRule]] = None,
) -> List[Line]:
    """
    Words → Lines. TEXT MERGED HERE: same line = Y within 2×font_size, gap_x ≤ max(3×font, 2×cw).

    Y-bands: |y_center − y_ref| ≤ 2×font_size_px, height_ratio ≤ 2.
    X-islands: gap_x ≤ max(3×font_size_px, 2×char_width); color compatible; no rule between words;
    words with has_background do not merge with words without.
    """
    _ = line_dy_px
    if not words:
        return []
    h_rules = horizontal_rules or ()
    v_rules = vertical_rules or ()

    bands = _group_into_local_y_bands(words)
    lines: List[Line] = []

    for band in bands:
        band_font_size = _median([int(_word_font_size_px(w)) for w in band])
        for island in _split_band_into_x_islands(
            band,
            char_width_px=char_width_px,
            band_med_h=band_font_size,
            horizontal_rules=h_rules,
            vertical_rules=v_rules,
        ):
            ws = island
            x1 = min(w.x for w in ws)
            y1 = min(w.y for w in ws)
            x2 = max(w.x + w.w for w in ws)
            y2 = max(w.y + w.h for w in ws)
            line_font_size = _median([int(_word_font_size_px(w)) for w in ws])
            lines.append(
                Line(
                    words=ws,
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    estimated_font_size_px=float(line_font_size),
                )
            )

    lines.sort(key=lambda l: (l.y + l.h // 2, l.x))
    return lines
