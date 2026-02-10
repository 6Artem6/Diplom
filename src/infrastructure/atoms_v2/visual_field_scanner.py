"""
VisualFieldScanner — ранний визуальный скан полей ввода без OCR bootstrap.

Работает только внутри form_region. Ищет прямоугольные области по:
- цветовой разнице границы и фона,
- замкнутому контуру,
- одинаковой толщине границ (вытянутый прямоугольник),
- вытянутой горизонтальной форме (aspect > 4).

Поддерживает светлую и тёмную тему: адаптивный threshold, усиление слабых границ (dilate / morph close).
Возвращает сырые field_bbox без классификации (не input/textarea). Не использует OCR как seed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Геометрические ограничения поля ввода
VISUAL_FIELD_ASPECT_MIN = 4.0  # ширина / высота > 4 (вытянутый горизонтально)
VISUAL_FIELD_HEIGHT_MIN_PX = 24
VISUAL_FIELD_HEIGHT_MAX_PX = 85
VISUAL_FIELD_WIDTH_MIN_PX = 80
VISUAL_FIELD_MIN_AREA = 400
# Контур: прямоугольность (approx polygon 4 вершины)
CONTOUR_EPSILON_RATIO = 0.05  # approxPolyDP от периметра
# Canny / threshold: светлая vs тёмная тема
DARK_THEME_LUMINANCE = 128
CANNY_LOW_LIGHT, CANNY_HIGH_LIGHT = 30, 100
CANNY_LOW_DARK, CANNY_HIGH_DARK = 50, 150
ADAPTIVE_BLOCK = 11
MORPH_KERNEL_SIZE = (3, 3)
DILATE_ITERATIONS = 1
# Обрезка по form_region с отступом
FORM_CROP_MARGIN_PX = 2


def _bbox_intersection(a: List[float], b: List[float]) -> Optional[List[float]]:
    if len(a) < 4 or len(b) < 4:
        return None
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _is_dark_theme(gray) -> bool:
    import numpy as np
    return float(gray.mean()) < DARK_THEME_LUMINANCE


def _scan_form_region_roi(
    crop: "Any",  # np.ndarray BGR
    form_bbox: List[float],
    roi_x0: int,
    roi_y0: int,
    dark_theme: bool,
) -> List[List[float]]:
    """
    Сканирует один crop (форма уже обрезана). Возвращает bbox в координатах crop (нужно + roi_x0, roi_y0).
    """
    import cv2
    import numpy as np

    if crop is None or crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = dark_theme or _is_dark_theme(gray)

    # Адаптивный порог: светлая тема — обычный, тёмная — инвертировать или другой порог
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK, 2
    )
    if dark:
        thresh = cv2.bitwise_not(thresh)
    low, high = (CANNY_LOW_DARK, CANNY_HIGH_DARK) if dark else (CANNY_LOW_LIGHT, CANNY_HIGH_LIGHT)
    edges = cv2.Canny(thresh, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    edges = cv2.dilate(edges, kernel, iterations=DILATE_ITERATIONS)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: List[List[float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < VISUAL_FIELD_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h <= 0:
            continue
        aspect = w / h
        if aspect < VISUAL_FIELD_ASPECT_MIN:
            continue
        if h < VISUAL_FIELD_HEIGHT_MIN_PX or h > VISUAL_FIELD_HEIGHT_MAX_PX:
            continue
        if w < VISUAL_FIELD_WIDTH_MIN_PX:
            continue
        # Проверка на прямоугольность (замкнутый контур, ~4 вершины)
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, CONTOUR_EPSILON_RATIO * peri, True)
        if len(approx) < 4:
            continue
        # Глобальные координаты
        global_bbox = [
            float(roi_x0 + x),
            float(roi_y0 + y),
            float(roi_x0 + x + w),
            float(roi_y0 + y + h),
        ]
        result.append(global_bbox)
    return result


def scan_form_regions(
    image_path: str,
    form_regions: List[Dict[str, Any]],
    dark_theme: bool = False,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Сканирует изображение только внутри переданных form_region.
    Возвращает (list of field_bbox в координатах изображения, stats для логов).
    """
    import cv2
    import numpy as np

    stats: Dict[str, Any] = {
        "form_regions_scanned": 0,
        "field_bbox_found": 0,
        "by_form": [],
    }
    all_bboxes: List[List[float]] = []
    if not form_regions or not image_path:
        return all_bboxes, stats

    img = cv2.imread(str(image_path))
    if img is None:
        logger.debug("visual_field_scanner: image not read %s", image_path)
        return all_bboxes, stats
    img_h, img_w = img.shape[:2]
    if not dark_theme:
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark_theme = _is_dark_theme(gray_full)

    for fr in form_regions:
        bbox = fr.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        x1 = max(0, int(bbox[0]) - FORM_CROP_MARGIN_PX)
        y1 = max(0, int(bbox[1]) - FORM_CROP_MARGIN_PX)
        x2 = min(img_w, int(bbox[2]) + FORM_CROP_MARGIN_PX)
        y2 = min(img_h, int(bbox[3]) + FORM_CROP_MARGIN_PX)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        stats["form_regions_scanned"] += 1
        local_bboxes = _scan_form_region_roi(crop, bbox, x1, y1, dark_theme)
        # Клип к границам form_region (не вылезать за пределы формы)
        for lb in local_bboxes:
            inter = _bbox_intersection(lb, bbox)
            if inter is not None:
                all_bboxes.append(inter)
        stats["by_form"].append(len(local_bboxes))

    stats["field_bbox_found"] = len(all_bboxes)
    return all_bboxes, stats


def run_visual_field_scan(
    image_path: Optional[str],
    form_regions: Optional[List[Dict[str, Any]]],
    dark_theme: bool = False,
) -> Tuple[List[List[float]], List[str]]:
    """
    Точка входа: запуск визуального сканера по form_region.
    Возвращает (field_bboxes в координатах изображения, log_lines).
    """
    log_lines: List[str] = []
    if not image_path or not form_regions:
        return [], log_lines
    bboxes, stats = scan_form_regions(image_path, form_regions, dark_theme)
    log_lines.append(
        "visual_field_scanner: form_regions=%d field_bbox_found=%d"
        % (stats["form_regions_scanned"], stats["field_bbox_found"])
    )
    if stats.get("by_form"):
        log_lines.append("visual_field_scanner by_form: %s" % stats["by_form"])
    return bboxes, log_lines
