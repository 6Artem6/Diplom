"""
Flow-based / implicit layout backend.

Goal: detect large implicit containers (messages/sections) on flow-based UIs (e.g. chat),
using OCR text lines + YOLO elements, without relying on VLMs.
Implements GUIDetectionService and returns GUIBlock-compatible output.

Architecture:
- Layout: OCR lines → vertical flows → implicit containers.
- Explicit elements: YOLO (buttons/inputs/images).
- Hierarchy: containers as parents, YOLO elements as children (bbox containment).
- OCR is used only as structural signal + text for containers, not as direct GUIBlock per line.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import logging

from PIL import Image

from src.domain.interfaces.gui_detection import GUIDetectionService
from src.domain.models.gui_block import GUIBlock
from src.infrastructure.layout import Word, Line
from src.infrastructure.layout.cv_prepass import run_cv_prepass
from src.infrastructure.layout.line_builder import words_to_lines
from src.infrastructure.layout.line_classifier import classify_line, classify_line_with_reason
from src.infrastructure.layout.block_builder import lines_to_blocks_with_headers

logger = logging.getLogger(__name__)

# Heuristic thresholds for flow grouping
LINE_DY_PX_DEFAULT = 18  # fallback when no words to estimate
# OCR→Word: do NOT filter by bbox size, color, or text length. If Tesseract returned bbox+text → Word always.
# Only skip invalid bbox (w<=0 or h<=0). Thin/light text must not be dropped.
MIN_BBOX_WIDTH = 1
MIN_BBOX_HEIGHT = 1
HORIZ_OVERLAP_RATIO = 0.4  # min horizontal overlap between consecutive lines
MAX_WIDTH_CHANGE_RATIO = 2.0  # width_j / width_i in [1/MAX, MAX]
# Gap between lines: ~1.2–1.5 × line height → new block (smaller = less over-merge, e.g. form fields)
BLOCK_GAP_LINE_HEIGHT_FACTOR = 1.4

# Sanitizer thresholds
MIN_AREA_RATIO = 0.0005  # blocks smaller этой доли от экрана считаются шумом (если ещё и пустые)
MIN_BLOCK_WIDTH = 20
MIN_BLOCK_HEIGHT = 12
IOU_DEDUP_THRESHOLD = 0.7
OVERLAP_SAME_TYPE_THRESHOLD = 0.8

INTERACTIVE_TYPES = {
    "button",
    "input",
    "checkbox",
    "dropdown",
    "list-item",
    "card",
    "image",
}


def _ocr_words_bbox(image_path: str, psm: int = 11) -> List[Tuple[str, int, int, int, int, Optional[float]]]:
    """Return list of (text, x, y, w, h, conf) using pytesseract.

    Invariant: if Tesseract returned bbox + non-empty text → Word is created.
    No filtering by color, font thickness, text length, or contrast.
    Only skip empty text or invalid bbox (w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT).
    """
    logger.info("FlowLayout: OCR start for %s (psm=%d)", image_path, psm)
    try:
        import pytesseract
    except ImportError as e:  # pragma: no cover - environment dependent
        # At INFO-level конфиге это единственный сигнал, почему words=0
        logger.warning("FlowLayout: pytesseract not available: %s", e)
        return []
    path = Path(image_path)
    if not path.exists():
        logger.warning("FlowLayout: image not found for OCR %s", image_path)
        return []
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:  # pragma: no cover - image IO
        logger.warning("FlowLayout: failed to load image %s: %s", image_path, e)
        return []
    try:
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT, config=f"--psm {psm}"
        )
    except Exception as e:  # pragma: no cover - OCR failure
        # Повышаем уровень, чтобы видеть реальные проблемы OCR в прод-логах
        logger.warning("FlowLayout: pytesseract.image_to_data failed for %s: %s", image_path, e)
        return []

    texts = data.get("text", []) or []
    n_raw = len(texts)
    n_non_empty = sum(1 for t in texts if (t or "").strip())
    logger.info(
        "FlowLayout: OCR image_to_data for %s returned n_raw=%d n_non_empty=%d (psm=%d)",
        image_path,
        n_raw,
        n_non_empty,
        psm,
    )

    out: List[Tuple[str, int, int, int, int, Optional[float]]] = []
    conf_list = data.get("conf", [])
    for i, t in enumerate(texts):
        t = (t or "").strip()
        if not t:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        if w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT:
            logger.debug("FlowLayout: OCR skipped invalid bbox %r w=%d h=%d", t, w, h)
            continue
        conf_val = conf_list[i] if conf_list and i < len(conf_list) else None
        conf: Optional[float] = None
        if conf_val is not None and conf_val != -1 and conf_val != "-1":
            try:
                conf = float(conf_val) / 100.0
            except (ValueError, TypeError):
                pass
        out.append((t, x, y, w, h, conf))
    logger.info("FlowLayout: OCR for %s words=%d (from %d non-empty)", image_path, len(out), n_non_empty)
    return out


def _group_words_into_lines(
    words: List[Tuple[str, int, int, int, int]], line_dy: int = LINE_DY_PX_DEFAULT
) -> List[List[Tuple[str, int, int, int, int]]]:
    """Group words by similar y (same visual line)."""
    if not words:
        return []
    # sort by center-y then x
    sorted_w = sorted(words, key=lambda w: (w[2] + w[4] // 2, w[1]))
    lines: List[List[Tuple[str, int, int, int, int]]] = []
    cur = [sorted_w[0]]
    y_prev = sorted_w[0][2] + sorted_w[0][4] // 2
    for w in sorted_w[1:]:
        y_c = w[2] + w[4] // 2
        if abs(y_c - y_prev) <= line_dy:
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
            y_prev = y_c
    if cur:
        lines.append(cur)
    return lines


def _line_bbox(line: List[Tuple[str, int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Union bbox of all words in line → (x, y, w, h)."""
    x1 = min(w[1] for w in line)
    y1 = min(w[2] for w in line)
    x2 = max(w[1] + w[3] for w in line)
    y2 = max(w[2] + w[4] for w in line)
    return x1, y1, x2 - x1, y2 - y1


def _estimate_line_height(words: List[Word]) -> float:
    """Estimate typical line (cap) height from word heights. Used for line/block grouping."""
    if not words:
        return float(LINE_DY_PX_DEFAULT)
    heights = [w.h for w in words]
    heights.sort()
    n = len(heights)
    return float(heights[n // 2]) if n else float(LINE_DY_PX_DEFAULT)


def _estimate_char_width(words: List[Word]) -> float:
    """Estimate average character width (px per char) for horizontal continuity. Used for width merge."""
    if not words:
        return 8.0
    per_char: List[float] = []
    for w in words:
        n = max(1, len(w.text))
        per_char.append(w.w / n)
    per_char.sort()
    n = len(per_char)
    return per_char[n // 2] if n else 8.0


class FlowLayoutDetectionService(GUIDetectionService):
    """
    Flow-based backend (step 0–1):
    WORD → LINE → TEXT_BLOCK → GUIBlock(text_block)

    No YOLO, no regions, no roles, no patterns.
    """

    def __init__(self, *_, **__) -> None:
        # yolo_detector may be passed from DI, but is intentionally unused at this stage.
        pass

    async def detect_gui_blocks(
        self, screenshot_path: str, ocr_text: str
    ) -> List[GUIBlock]:
        path = Path(screenshot_path)
        screenshot_id = path.stem

        # 1) OCR → Words (psm 11 = sparse text, better vertical separation for lists/cards)
        raw_words = _ocr_words_bbox(screenshot_path, psm=11)
        words: List[Word] = [
            Word(text=t, x=x, y=y, w=w, h=h, conf=conf)
            for (t, x, y, w, h, conf) in raw_words
        ]

        # 2) CV prepass: visual context (background, text color, separators) for layout invariants
        words, horizontal_rules, vertical_rules = run_cv_prepass(screenshot_path, words)

        # 3) Font-size–dependent estimates from words (for line/block grouping)
        line_height_px = _estimate_line_height(words)
        line_dy_px = max(10, int(line_height_px * 0.8))  # same-line tolerance ~ height of capitals
        char_width_px = _estimate_char_width(words)  # horizontal continuity: gap up to a few chars
        logger.debug(
            "FlowLayout: line_height=%.0f line_dy_px=%d char_width=%.1f for %s",
            line_height_px,
            line_dy_px,
            char_width_px,
            screenshot_path,
        )

        # 4) Words → Lines (Y-band + X-islands; height_ratio ≤ 2; color/rules/bg compatibility)
        lines = words_to_lines(
            words,
            line_dy_px=line_dy_px,
            char_width_px=char_width_px,
            horizontal_rules=horizontal_rules,
            vertical_rules=vertical_rules,
        )

        # 5) Line classification: role (body/header/button/label); body default
        lines_with_roles: List[Line] = []
        for idx, ln in enumerate(lines):
            role = classify_line(ln, lines)
            lines_with_roles.append(
                Line(
                    words=ln.words,
                    x=ln.x,
                    y=ln.y,
                    w=ln.w,
                    h=ln.h,
                    is_header=(role == "header"),
                    role=role,
                )
            )
            if logger.isEnabledFor(logging.DEBUG):
                _, reason = classify_line_with_reason(ln, lines, page_width=None)
                logger.debug(
                    "Line(id=%s): words=[%s] font_px=%s role=%s reason=%s",
                    idx,
                    ",".join((w.text or "").strip() or "?" for w in ln.words),
                    ln.h,
                    role,
                    reason,
                )
        n_headers = sum(1 for l in lines_with_roles if l.role == "header")
        n_buttons = sum(1 for l in lines_with_roles if l.role == "button")
        logger.info(
            "FlowLayout: headers=%d buttons=%d of %d lines for %s",
            n_headers,
            n_buttons,
            len(lines_with_roles),
            screenshot_path,
        )

        # 6) Lines → TextBlocks: role + dividers; button/header never merge with body
        text_blocks = lines_to_blocks_with_headers(
            lines_with_roles,
            char_width_px=char_width_px,
            dividers=horizontal_rules,
        )

        logger.info(
            "FlowLayout: words=%d, lines=%d, text_blocks=%d for %s",
            len(words),
            len(lines),
            len(text_blocks),
            screenshot_path,
        )

        # 7) Map TextBlocks → GUIBlock (§6: type header|paragraph|standalone|button)
        gui_blocks: List[GUIBlock] = []
        for i, tb in enumerate(text_blocks):
            text = "\n".join(ln.text for ln in tb.lines)
            etypes = ["text_block"]
            if tb.block_type == "header":
                etypes = ["header", "text_block"]
            elif any(getattr(ln, "role", None) == "button" for ln in tb.lines):
                etypes = ["button", "text_block"]
            gui_blocks.append(
                GUIBlock(
                    id=f"{screenshot_id}_tb_{i}",
                    screenshot_id=screenshot_id,
                    screenshot_path=screenshot_path,
                    bounding_box={
                        "x": tb.x,
                        "y": tb.y,
                        "width": tb.w,
                        "height": tb.h,
                        "x1": tb.x,
                        "y1": tb.y,
                        "x2": tb.x + tb.w,
                        "y2": tb.y + tb.h,
                    },
                    element_types=etypes,
                    ocr_text=text,
                    visual_features=[],
                )
            )

        return gui_blocks


def _horizontal_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Horizontal overlap length / min(width_a, width_b)."""
    ax1, ax2 = a["x"], a["x"] + a["w"]
    bx1, bx2 = b["x"], b["x"] + b["w"]
    ix1 = max(ax1, bx1)
    ix2 = min(ax2, bx2)
    if ix2 <= ix1:
        return 0.0
    overlap = ix2 - ix1
    return overlap / max(1, min(a["w"], b["w"]))


def _width_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """max(width_a, width_b) / min(width_a, width_b)."""
    wa, wb = a["w"], b["w"]
    mn = min(wa, wb)
    mx = max(wa, wb)
    if mn <= 0:
        return MAX_WIDTH_CHANGE_RATIO + 1.0
    return mx / mn


def _build_hierarchy(
    containers: List[GUIBlock], explicit: List[GUIBlock]
) -> List[GUIBlock]:
    """Assign explicit blocks as children of nearest containing container.

    - Containers: implicit flow blocks (section/text_block).
    - Explicit: YOLO elements (button, input, image, etc.).
    - Children are attached via bbox containment; roots are containers.
    """
    roots = containers.copy()

    for blk in explicit:
        parent = _find_smallest_container(blk, containers)
        if parent is None:
            # No container; treat as separate root
            roots.append(blk)
        else:
            children = list(parent.children or [])
            children.append(blk)
            parent.children = children

    return roots


def _find_smallest_container(
    child: GUIBlock, containers: List[GUIBlock]
) -> GUIBlock | None:
    """Return smallest container whose bbox fully contains child bbox."""
    cb = child.bounding_box
    cx1, cy1, cx2, cy2 = cb["x1"], cb["y1"], cb["x2"], cb["y2"]
    candidates: List[Tuple[float, GUIBlock]] = []
    for cont in containers:
        b = cont.bounding_box
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        if x1 <= cx1 and y1 <= cy1 and x2 >= cx2 and y2 >= cy2:
            area = (x2 - x1) * (y2 - y1)
            candidates.append((area, cont))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def _sanitize_layout(roots: List[GUIBlock], img_w: int, img_h: int) -> List[GUIBlock]:
    """Remove obviously noisy blocks and merge near-duplicates on the same level.

    Rules (approximate implementation of LayoutSanitizer spec):
    - Удаляем блок, если выполняются ≥2 условий:
      * площадь сильно мала относительно экрана,
      * пустой текст,
      * нет детей,
      * экстремальные пропорции (очень узкий/низкий),
      * сильно перекрывается с более крупным блоком того же типа.
    - Не возвращаем блоки без текста, детей или интерактивных element_types.
    - На одном уровне выполняем merge для пар с большим IoU и совместимыми типами.
    """

    image_area = max(1, img_w * img_h)

    def is_interactive(block: GUIBlock) -> bool:
        return any(t in INTERACTIVE_TYPES for t in block.element_types)

    def block_area(b: GUIBlock) -> int:
        bb = b.bounding_box
        return max(0, (bb["x2"] - bb["x1"]) * (bb["y2"] - bb["y1"]))

    def iou(a: GUIBlock, b: GUIBlock) -> float:
        ba, bb = a.bounding_box, b.bounding_box
        ax1, ay1, ax2, ay2 = ba["x1"], ba["y1"], ba["x2"], ba["y2"]
        bx1, by1, bx2, by2 = bb["x1"], bb["y1"], bb["x2"], bb["y2"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        ua = block_area(a)
        ub = block_area(b)
        union = ua + ub - inter
        return inter / union if union > 0 else 0.0

    def same_type(a: GUIBlock, b: GUIBlock) -> bool:
        return bool(a.element_types and b.element_types and a.element_types[0] == b.element_types[0])

    def compatible_types(a: GUIBlock, b: GUIBlock) -> bool:
        t1 = a.element_types[0] if a.element_types else ""
        t2 = b.element_types[0] if b.element_types else ""
        pair = {t1, t2}
        # button + text, input + text, card + text_block, etc.
        compatible_sets = [
            {"button", "text"},
            {"input", "text"},
            {"card", "text_block"},
            {"card", "text"},
        ]
        if pair in compatible_sets:
            return True
        # identical type: allow merge if bbox почти совпадает
        return t1 == t2

    def merge_blocks(a: GUIBlock, b: GUIBlock) -> GUIBlock:
        """Union bbox, choose dominant type, merge text и children."""
        ba, bb = a.bounding_box, b.bounding_box
        x1 = min(ba["x1"], bb["x1"])
        y1 = min(ba["y1"], bb["y1"])
        x2 = max(ba["x2"], bb["x2"])
        y2 = max(ba["y2"], bb["y2"])
        w = x2 - x1
        h = y2 - y1

        def dominant_type(types: List[str]) -> str:
            order = ["button", "input", "card", "list-item", "image", "text_block", "section", "text"]
            for t in order:
                if t in types:
                    return t
            return types[0] if types else "text_block"

        types = list({*(a.element_types or []), *(b.element_types or [])})
        etype = dominant_type(types)

        texts = []
        for t in [a.ocr_text, b.ocr_text]:
            t_norm = (t or "").strip()
            if t_norm and t_norm not in texts:
                texts.append(t_norm)
        merged_text = "\n".join(texts)

        children = []
        if a.children:
            children.extend(a.children)
        if b.children:
            children.extend(b.children)

        return GUIBlock(
            id=a.id,  # keep one id; upstream uses ids only as local identifiers
            screenshot_id=a.screenshot_id,
            screenshot_path=a.screenshot_path,
            bounding_box={
                "x": x1,
                "y": y1,
                "width": w,
                "height": h,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            element_types=[etype],
            ocr_text=merged_text,
            visual_features=[],
            children=children or None,
        )

    def prune_and_dedup(level_blocks: List[GUIBlock]) -> List[GUIBlock]:
        # 1) рекурсивно обработать детей
        for b in level_blocks:
            if b.children:
                b.children = prune_and_dedup(list(b.children)) or None

        # 2) первичное удаление шумовых блоков
        kept: List[GUIBlock] = []
        for b in level_blocks:
            bb = b.bounding_box
            w = bb["x2"] - bb["x1"]
            h = bb["y2"] - bb["y1"]
            area = block_area(b)
            area_ratio = area / image_area
            empty_text = not (b.ocr_text or "").strip()
            no_children = not b.children
            tiny = area_ratio < MIN_AREA_RATIO
            bad_shape = w < MIN_BLOCK_WIDTH or h < MIN_BLOCK_HEIGHT
            # не интерактивный и без детей/текста
            non_interactive = not is_interactive(b)

            # считаем кол-во "плохих" признаков
            bad_flags = [
                tiny,
                empty_text,
                no_children,
                bad_shape,
            ]
            bad_count = sum(1 for f in bad_flags if f)

            # Жёсткое правило: не возвращаем блоки без текста, детей и интерактива
            if empty_text and no_children and not is_interactive(b):
                continue

            # Удаляем блок, если ≥2 плохих признаков и он не интерактивный
            if bad_count >= 2 and non_interactive:
                continue

            kept.append(b)

        # 3) удаление сильно перекрывающихся блоков одинакового типа
        pruned: List[GUIBlock] = []
        for b in kept:
            drop = False
            for other in kept:
                if other is b:
                    continue
                if same_type(b, other) and block_area(other) > block_area(b):
                    if iou(b, other) >= OVERLAP_SAME_TYPE_THRESHOLD:
                        drop = True
                        break
            if not drop:
                pruned.append(b)

        # 4) merge-проход: объединяем пары с большим IoU и совместимыми типами
        merged: List[GUIBlock] = []
        for b in pruned:
            merged_into_existing = False
            for i, m in enumerate(merged):
                if compatible_types(b, m) and iou(b, m) >= IOU_DEDUP_THRESHOLD:
                    merged[i] = merge_blocks(m, b)
                    merged_into_existing = True
                    break
            if not merged_into_existing:
                merged.append(b)

        return merged

    sanitized = prune_and_dedup(roots)
    # Простая диагностика пересечений на верхнем уровне
    intersections = 0
    for i, a in enumerate(sanitized):
        for b in sanitized[i + 1 :]:
            if iou(a, b) > 0.0:
                intersections += 1
    if intersections:
        logger.info("FlowLayout: sanitizer detected %d intersections at root level", intersections)
    return sanitized


def _log_group_signatures(roots: List[GUIBlock], img_w: int, img_h: int) -> None:
    """Extract coarse signatures for large containers to enable pattern mining later.

    Signature features (per container):
    - normalized_bbox_ratio (area / image_area)
    - children_types_sequence (e.g., ['text', 'text', 'button'])
    - text_density (len(text) / area)
    - has_interaction (any child interactive)
    """

    image_area = max(1, img_w * img_h)

    def is_large_container(b: GUIBlock) -> bool:
        et = (b.element_types or [""])[0]
        return et in {"section", "text_block", "card", "list-item", "message"}

    def walk(blocks: List[GUIBlock]) -> List[GUIBlock]:
        acc: List[GUIBlock] = []
        for b in blocks:
            if is_large_container(b):
                acc.append(b)
            if b.children:
                acc.extend(walk(list(b.children)))
        return acc

    containers = walk(roots)
    if not containers:
        return

    signatures: Dict[Tuple, int] = {}
    for b in containers:
        bb = b.bounding_box
        area = max(1, (bb["x2"] - bb["x1"]) * (bb["y2"] - bb["y1"]))
        norm_ratio = area / image_area
        text = (b.ocr_text or "").strip()
        text_density = len(text) / area
        child_types = []
        has_interaction = False
        for ch in (b.children or []):
            et = (ch.element_types or [""])[0]
            child_types.append(et)
            if et in INTERACTIVE_TYPES:
                has_interaction = True
        key = (
            round(norm_ratio, 3),
            tuple(child_types),
            round(text_density, 3),
            has_interaction,
        )
        signatures[key] = signatures.get(key, 0) + 1

    # Логируем только повторяющиеся паттерны (count > 1)
    repeated = {k: c for k, c in signatures.items() if c > 1}
    if repeated:
        logger.info("FlowLayout: repeated container patterns (pattern_id -> count):")
        for idx, (sig, count) in enumerate(repeated.items(), start=1):
            norm_ratio, child_seq, density, has_inter = sig
            logger.info(
                "  pattern_%d: norm_ratio=%.3f, children=%s, text_density=%.3f, has_interaction=%s, count=%d",
                idx,
                norm_ratio,
                list(child_seq),
                density,
                has_inter,
                count,
            )

