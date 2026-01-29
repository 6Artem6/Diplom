"""
Words → Lines for web pages.

Y-bands (local median height) + X-islands (gap, height, color, rules).
Words with has_background do not merge with words without; text color compatible; rules break.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .atoms import HorizontalRule, Line, VerticalRule, Word

# Y-band: word joins band iff y_center within band AND height_ratio ≤ 2
Y_BAND_DY_RATIO = 0.55
HEIGHT_RATIO_MAX = 2.0  # height_ratio_word / median_height_band ≤ 2

# X-island: new cluster if gap_x > max(1.5 * h_band, 2.5 * char_width)
X_GAP_HEIGHT_FACTOR = 1.5
X_GAP_CHAR_FACTOR = 2.5


def _median(values: List[int]) -> float:
    if not values:
        return 18.0
    vv = sorted(values)
    return float(vv[len(vv) // 2])


def _word_center_y(w: Word) -> float:
    return float(w.y + w.h / 2.0)


def _estimate_char_width(words: List[Word]) -> Optional[float]:
    """Median char width from word width / len(text)."""
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
    Y-bands: words on same visual line. Word joins band iff:
    - y_center within 0.55 * local_median_h of band center
    - height_ratio(word, band) ≤ 2
    """
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (_word_center_y(w), w.x))
    bands: List[List[Word]] = []
    band_centers: List[float] = []
    band_med_h_list: List[float] = []

    for w in sorted_words:
        cy = _word_center_y(w)
        best_idx: Optional[int] = None
        best_dist = 1e18
        for i, center in enumerate(band_centers):
            dist = abs(cy - center)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is None:
            bands.append([w])
            band_centers.append(cy)
            band_med_h_list.append(float(w.h))
            continue

        local_med_h = band_med_h_list[best_idx]
        height_ok = _height_ratio(float(w.h), local_med_h) <= HEIGHT_RATIO_MAX
        if best_dist <= Y_BAND_DY_RATIO * local_med_h and height_ok:
            bands[best_idx].append(w)
            band_centers[best_idx] = _median([int(_word_center_y(ww)) for ww in bands[best_idx]])
            band_med_h_list[best_idx] = _median([ww.h for ww in bands[best_idx]])
        else:
            bands.append([w])
            band_centers.append(cy)
            band_med_h_list.append(float(w.h))

    bands_with_center = list(zip(bands, band_centers))
    bands_with_center.sort(key=lambda bc: bc[1])
    return [b for (b, _c) in bands_with_center]


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

    cw = char_width_px or _estimate_char_width(band_words) or (band_med_h * 0.5)
    max_gap_x = max(X_GAP_HEIGHT_FACTOR * band_med_h, X_GAP_CHAR_FACTOR * cw)

    ws = sorted(band_words, key=lambda w: w.x)
    islands: List[List[Word]] = []
    cur: List[Word] = [ws[0]]
    last_x2 = ws[0].x + ws[0].w

    for w in ws[1:]:
        gap_x = w.x - last_x2
        cur_med_h = _median([ww.h for ww in cur])
        cur_has_bg = any(getattr(ww, "has_background", False) for ww in cur)
        w_has_bg = getattr(w, "has_background", False)

        # Words with has_background do not merge with words without
        if cur_has_bg != w_has_bg:
            islands.append(cur)
            cur = [w]
            last_x2 = w.x + w.w
            continue

        if gap_x > max_gap_x:
            islands.append(cur)
            cur = [w]
            last_x2 = w.x + w.w
            continue

        # height_ratio ≤ 2
        if _height_ratio(float(w.h), cur_med_h) > HEIGHT_RATIO_MAX:
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
    Words → Lines. Y-bands (local median h, height_ratio ≤ 2) + X-islands.

    X-island: gap_x ≤ max(1.5*h_band, 2.5*char_width); height_ratio ≤ 2;
    text color compatible; no horizontal/vertical rule between words;
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
        band_med_h = _median([w.h for w in band])
        for island in _split_band_into_x_islands(
            band,
            char_width_px=char_width_px,
            band_med_h=band_med_h,
            horizontal_rules=h_rules,
            vertical_rules=v_rules,
        ):
            ws = island
            x1 = min(w.x for w in ws)
            y1 = min(w.y for w in ws)
            x2 = max(w.x + w.w for w in ws)
            y2 = max(w.y + w.h for w in ws)
            lines.append(Line(words=ws, x=x1, y=y1, w=x2 - x1, h=y2 - y1))

    lines.sort(key=lambda l: (l.y + l.h // 2, l.x))
    return lines
