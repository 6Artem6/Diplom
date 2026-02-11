"""
Строгий семантический классификатор строк (RowSemanticClassifier).

Запускается между этапом CV-якорей и SlotDetector. CV — единственный источник геометрии строк;
классификатор меняет только row_type, input_bbox (обнуление), action_bbox, label_bbox — не границы строк.

Жёсткие правила:
- HEADER: крупный OCR (> baseline*1.35), нет рамки поля, нет placeholder, нет label сверху.
- ACTION: bbox шире 60% контейнера, OCR ≤3 слова, центрирован, нет placeholder/label сверху.
- TEXT: нет визуального прямоугольника поля, нет placeholder.
- FIELD допускается только при: визуальная рамка + высота bbox близка к median_input_height + (placeholder или label).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.infrastructure.atoms_v2.experimental_v2.models import FormRow

logger = logging.getLogger(__name__)

# Жёсткие пороги (без мягких эвристик). Приоритет: HEADER → TEXT → ACTION → FIELD
HEADER_OCR_HEIGHT_RATIO = 1.35
HEADER_TOP_ZONE_RATIO = 0.2   # строка в верхних 20% контейнера
HEADER_HEIGHT_MAX_RATIO = 0.9  # bbox height < 0.9×median для альтернативы HEADER
ACTION_ROW_WIDTH_RATIO = 0.4  # ширина bbox ≥ 0.4 контейнера
ACTION_HEIGHT_MAX_RATIO = 1.2  # bbox height ≤ 1.2×median
ACTION_TOP_ZONE_RATIO = 0.25  # ACTION не в верхних 25%
ACTION_MAX_WORDS = 3
CENTER_TOLERANCE_RATIO = 0.35
FIELD_HEIGHT_LO_RATIO = 0.7   # ±30% от median
FIELD_HEIGHT_HI_RATIO = 1.3
LABEL_NOT_INPUT_HEIGHT_RATIO = 0.7  # bbox < 0.7*median без рамки → не input
LABEL_ABOVE_MAX_GAP_PX = 50
MIN_INPUT_HEIGHT_PX = 28


def _ocr_in_row(ob: Dict[str, Any], row: FormRow) -> bool:
    b = ob.get("bbox") or []
    if len(b) < 4:
        return False
    cy = (b[1] + b[3]) / 2
    return row.y_min <= cy <= row.y_max and row.x_min <= (b[0] + b[2]) / 2 <= row.x_max


def _row_has_visual_box(row: FormRow) -> bool:
    """Строка имеет визуальный прямоугольник поля (input_bbox от CV, не вся строка)."""
    ib = getattr(row, "input_bbox", None)
    if not ib or len(ib) < 4:
        return False
    row_h = row.y_max - row.y_min
    ib_h = ib[3] - ib[1]
    if row_h < 1:
        return True
    return ib_h >= min(MIN_INPUT_HEIGHT_PX, row_h * 0.5)


def _x_overlap_ratio(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix1 = max(a[0], b[0])
    ix2 = min(a[2], b[2])
    if ix2 <= ix1:
        return 0.0
    wa = a[2] - a[0]
    return (ix2 - ix1) / wa if wa > 0 else 0.0


def _bbox_fully_inside(inner: List[float], outer: List[float]) -> bool:
    if len(inner) < 4 or len(outer) < 4:
        return False
    return (
        outer[0] <= inner[0]
        and inner[2] <= outer[2]
        and outer[1] <= inner[1]
        and inner[3] <= outer[3]
    )


def _has_placeholder_in_row(ocr_in_row: List[Dict[str, Any]], input_bbox: Optional[List[float]]) -> bool:
    if not input_bbox or len(input_bbox) < 4:
        return False
    for ob in ocr_in_row:
        b = ob.get("bbox") or []
        if len(b) < 4:
            continue
        if _x_overlap_ratio(b, input_bbox) < 0.8:
            continue
        if not _bbox_fully_inside(b, input_bbox):
            continue
        txt = (ob.get("text") or "").strip()
        if len(txt) < 25:
            return True
    return False


def _is_centered(b: List[float], container_bbox: List[float]) -> bool:
    if len(b) < 4 or len(container_bbox) < 4:
        return False
    cx = (b[0] + b[2]) / 2
    c_w = container_bbox[2] - container_bbox[0]
    cont_cx = (container_bbox[0] + container_bbox[2]) / 2
    return c_w > 0 and abs(cx - cont_cx) / c_w <= CENTER_TOLERANCE_RATIO


def _has_label_above(
    row_index: int,
    rows: List[FormRow],
    layout_ocr: List[Dict[str, Any]],
) -> bool:
    """Есть ли текст (label) непосредственно над этой строкой в пределах зазора."""
    if row_index <= 0:
        return False
    prev = rows[row_index - 1]
    r = rows[row_index]
    gap = r.y_min - prev.y_max
    if gap > LABEL_ABOVE_MAX_GAP_PX or gap < 0:
        return False
    ocr_prev = [ob for ob in layout_ocr if _ocr_in_row(ob, prev)]
    if not ocr_prev:
        return False
    normalized = re.sub(r"\s+", " ", " ".join((ob.get("text") or "").strip() for ob in ocr_prev))
    letters = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9]", "", normalized)
    return len(normalized) >= 2 and len(letters) >= 1


def _word_count(text: str) -> int:
    return len(re.sub(r"\s+", " ", text.strip()).split()) if text else 0


def _max_ocr_height_in_row(ocr_in_row: List[Dict[str, Any]]) -> float:
    out = 0.0
    for ob in ocr_in_row:
        b = ob.get("bbox") or []
        if len(b) >= 4:
            out = max(out, b[3] - b[1])
    return out


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[len(s) // 2])


def _bbox_has_visible_frame_stub(image_path: Optional[str], bbox: List[float], ratio: float = 0.6) -> bool:
    """Проверка рамки по изображению (минимальная реализация, без импорта form_inner_layout)."""
    if not image_path or not bbox or len(bbox) < 4:
        return True
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(image_path))
        if img is None:
            return True
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if x2 <= x1 or y2 <= y1:
            return True
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return True
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        w = edges.shape[1]
        if w < 2:
            return True
        h = edges.shape[0]
        band = max(2, h // 6)
        top_band = edges[:band, :]
        bottom_band = edges[-band:, :] if h > band else edges
        cols_top = np.sum(np.sum(top_band > 0, axis=0) > 0)
        cols_bottom = np.sum(np.sum(bottom_band > 0, axis=0) > 0)
        return cols_top >= w * ratio or cols_bottom >= w * ratio
    except Exception:
        return True


def _bbox_has_low_color_variance_stub(image_path: Optional[str], bbox: List[float], max_std: float = 45.0) -> bool:
    if not image_path or not bbox or len(bbox) < 4:
        return False
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if x2 <= x1 or y2 <= y1:
            return False
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray)) <= max_std
    except Exception:
        return False


def classify_rows(
    rows: List[FormRow],
    layout_ocr: List[Dict[str, Any]],
    container_bbox: List[float],
    baseline: Dict[str, Any],
    image_path: Optional[str] = None,
) -> None:
    """
    Строгий семантический классификатор. Приоритет: HEADER → TEXT → ACTION → FIELD.
    Изменяет только row_type и input_bbox (обнуление). Границы строк не меняются.
    """
    if len(container_bbox) < 4 or not rows:
        return
    c_x_min, c_y_min, c_x_max, c_y_max = (
        container_bbox[0], container_bbox[1], container_bbox[2], container_bbox[3],
    )
    container_w = c_x_max - c_x_min
    container_h = c_y_max - c_y_min
    baseline_font = float(baseline.get("median_font_height", 20.0))
    header_font_min = baseline_font * HEADER_OCR_HEIGHT_RATIO
    header_zone_bottom = c_y_min + container_h * HEADER_TOP_ZONE_RATIO if container_h > 0 else c_y_min
    action_zone_bottom = c_y_min + container_h * ACTION_TOP_ZONE_RATIO if container_h > 0 else c_y_min

    # Медиана высоты input по строкам с визуальным box (для правила FIELD)
    input_heights: List[float] = []
    for r in rows:
        if getattr(r, "input_bbox", None) and len(r.input_bbox) >= 4:
            if r.row_type in ("TEXTAREA", "ACTION", "HEADER", "TEXT"):
                continue
            input_heights.append(r.input_bbox[3] - r.input_bbox[1])
    median_input_height = _median(input_heights) if input_heights else 40.0

    for i, r in enumerate(rows):
        if r.row_type == "TEXTAREA":
            continue
        ocr_here = [ob for ob in layout_ocr if _ocr_in_row(ob, r)]
        row_w = r.x_max - r.x_min
        input_bbox = getattr(r, "input_bbox", None)
        has_visual_box = _row_has_visual_box(r)
        has_placeholder = _has_placeholder_in_row(
            ocr_here,
            input_bbox if (input_bbox and len(input_bbox) >= 4) else None,
        )
        has_label_above = _has_label_above(i, rows, layout_ocr)
        max_ocr_h = _max_ocr_height_in_row(ocr_here)

        # 1. HEADER: строка в верхних 20%, нет рамки, нет placeholder; и (bbox height < 0.9×median OR OCR > baseline×1.35)
        row_center_y = (r.y_min + r.y_max) / 2
        in_header_zone = row_center_y <= header_zone_bottom
        ih = (input_bbox[3] - input_bbox[1]) if (input_bbox and len(input_bbox) >= 4) else 0.0
        header_by_height = median_input_height > 0 and ih > 0 and ih < median_input_height * HEADER_HEIGHT_MAX_RATIO
        header_by_ocr = max_ocr_h > header_font_min
        if (
            in_header_zone
            and not has_visual_box
            and not has_placeholder
            and not has_label_above
            and (header_by_height or header_by_ocr)
        ):
            r.row_type = "HEADER"
            r.input_bbox = None
            if getattr(r, "input_bboxes", None):
                r.input_bboxes = None
            r.column_count = 0
            continue

        # 2. TEXT: нет визуального прямоугольника, нет placeholder (до ACTION)
        if not has_visual_box and not has_placeholder:
            r.row_type = "TEXT"
            r.input_bbox = None
            if getattr(r, "input_bboxes", None):
                r.input_bboxes = None
            continue

        # 3. ACTION: width ≥ 0.4 container, height ≤ 1.2×median, не в верхних 25%, (visible_frame OR low_variance)
        not_in_action_excluded_zone = row_center_y > action_zone_bottom
        action_height_ok = (not input_bbox or len(input_bbox) < 4 or median_input_height <= 0 or
                            (input_bbox[3] - input_bbox[1]) <= median_input_height * ACTION_HEIGHT_MAX_RATIO)
        action_frame_or_variance = True
        if image_path and input_bbox and len(input_bbox) >= 4:
            action_frame_or_variance = (_bbox_has_visible_frame_stub(image_path, input_bbox, 0.6)
                                        or _bbox_has_low_color_variance_stub(image_path, input_bbox))
        if (container_w > 0 and row_w >= container_w * ACTION_ROW_WIDTH_RATIO
                and not_in_action_excluded_zone and action_height_ok and action_frame_or_variance):
            combined_text = " ".join((ob.get("text") or "").strip() for ob in ocr_here).strip()
            words = _word_count(combined_text)
            any_centered = any(_is_centered(ob.get("bbox", []), container_bbox) for ob in ocr_here if ob.get("bbox"))
            if (
                words <= ACTION_MAX_WORDS
                and any_centered
                and not has_placeholder
                and not has_label_above
            ):
                r.row_type = "ACTION"
                r.input_bbox = None
                if getattr(r, "input_bboxes", None):
                    r.input_bboxes = None
                continue

        # 3b. Защита «label стал input»: bbox высотой < 0.7*median, только текст, нет рамки → TEXT/HEADER
        if input_bbox and len(input_bbox) >= 4 and median_input_height > 0:
            ih = input_bbox[3] - input_bbox[1]
            if ih < median_input_height * LABEL_NOT_INPUT_HEIGHT_RATIO and not has_visual_box:
                r.row_type = "HEADER" if (in_header_zone and max_ocr_h > header_font_min) else "TEXT"
                r.input_bbox = None
                if getattr(r, "input_bboxes", None):
                    r.input_bboxes = None
                continue

        # 4. FIELD допускается только при: визуальная рамка + высота ≈ median ±30% + (placeholder или label)
        if r.row_type in ("FIELD", "FIELD_HORIZONTAL", "FIELD_VERTICAL", "FIELD_INPUT_ONLY"):
            if not has_visual_box:
                r.row_type = "TEXT"
                r.input_bbox = None
                if getattr(r, "input_bboxes", None):
                    r.input_bboxes = None
                continue
            if not (has_placeholder or getattr(r, "label_bbox", None) or has_label_above):
                r.row_type = "TEXT"
                r.input_bbox = None
                if getattr(r, "input_bboxes", None):
                    r.input_bboxes = None
                continue
            if input_bbox and len(input_bbox) >= 4 and median_input_height > 0:
                ih = input_bbox[3] - input_bbox[1]
                if ih < median_input_height * FIELD_HEIGHT_LO_RATIO or ih > median_input_height * FIELD_HEIGHT_HI_RATIO:
                    if max_ocr_h > header_font_min and in_header_zone:
                        r.row_type = "HEADER"
                        r.column_count = 0
                    else:
                        r.row_type = "TEXT"
                    r.input_bbox = None
                    if getattr(r, "input_bboxes", None):
                        r.input_bboxes = None
                    continue
