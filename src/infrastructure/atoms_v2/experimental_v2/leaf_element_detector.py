"""
LeafElementDetection — диагностический слой (Stage 1).

Работает строго внутри row.bbox. Не меняет:
- row_type
- input_bbox / input_bboxes
- row.y_min / row.y_max / x_min / x_max
- slots
- container
- graph

Только добавляет metadata и логирует результаты для анализа конфликтов root vs leaf.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Пороги для детекторов
BUTTON_MIN_WIDTH_RATIO = 0.4  # ширина ≥ 0.4 × container
BUTTON_MAX_COLOR_STD = 50.0   # низкая цветовая дисперсия
BUTTON_CENTER_TOLERANCE = 0.35

CHECKBOX_SIZE_MIN = 12
CHECKBOX_SIZE_MAX = 28
CHECKBOX_ASPECT_LO = 0.8
CHECKBOX_ASPECT_HI = 1.2
CHECKBOX_OCR_X_OVERLAP = 0.3

RADIO_DIAMETER_MIN = 12
RADIO_DIAMETER_MAX = 28
RADIO_CIRCULARITY_MIN = 0.7

TEXTAREA_HEIGHT_RATIO = 1.8
TEXTAREA_WIDTH_RATIO = 0.5
TEXTAREA_FRAME_RATIO = 0.5


def _crop_roi(image_path: str, bbox: List[float]) -> Optional[Any]:
    """Вырезает ROI из изображения по bbox. Возвращает numpy array или None."""
    if not image_path or len(bbox) < 4:
        return None
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if x2 <= x1 or y2 <= y1:
            return None
        roi = img[y1:y2, x1:x2]
        return roi if roi.size > 0 else None
    except Exception:
        return None


def _color_variance(roi: Any) -> float:
    """Стандартное отклонение по серому каналу (мера однородности цвета)."""
    try:
        import cv2
        import numpy as np
        if roi is None or roi.size == 0:
            return 999.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))
    except Exception:
        return 999.0


def _ocr_in_bbox(ocr_boxes: List[Dict[str, Any]], bbox: List[float]) -> List[Dict[str, Any]]:
    """OCR-блоки с центром внутри bbox."""
    if len(bbox) < 4:
        return []
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    out = []
    for ob in ocr_boxes:
        b = ob.get("bbox") or []
        if len(b) < 4:
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            out.append(ob)
    return out


def _is_ocr_centered(ocr_boxes: List[Dict[str, Any]], bbox: List[float]) -> bool:
    """Хотя бы один OCR центрирован по X в bbox."""
    if len(bbox) < 4 or not ocr_boxes:
        return False
    bbox_cx = (bbox[0] + bbox[2]) / 2
    bbox_w = bbox[2] - bbox[0]
    if bbox_w <= 0:
        return False
    for ob in ocr_boxes:
        b = ob.get("bbox") or []
        if len(b) < 4:
            continue
        ocr_cx = (b[0] + b[2]) / 2
        if abs(ocr_cx - bbox_cx) / bbox_w <= BUTTON_CENTER_TOLERANCE:
            return True
    return False


def detect_button_like(
    roi: Any,
    row_bbox: List[float],
    container_bbox: List[float],
    ocr_in_row: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Детектор button-like элемента.
    Условия: ширина ≥ 0.4 контейнера, низкая цветовая дисперсия, OCR внутри и центрирован.
    """
    result = {"type": "button", "confidence": 0.0, "reason": ""}
    if roi is None or len(row_bbox) < 4 or len(container_bbox) < 4:
        result["reason"] = "invalid_input"
        return result

    container_w = container_bbox[2] - container_bbox[0]
    row_w = row_bbox[2] - row_bbox[0]
    if container_w <= 0:
        result["reason"] = "zero_container_width"
        return result

    # Ширина ≥ 0.4 контейнера
    width_ok = row_w >= container_w * BUTTON_MIN_WIDTH_RATIO
    if not width_ok:
        result["reason"] = "width_too_small"
        return result

    # Низкая цветовая дисперсия
    color_std = _color_variance(roi)
    color_ok = color_std <= BUTTON_MAX_COLOR_STD
    if not color_ok:
        result["reason"] = f"high_color_variance={color_std:.1f}"
        return result

    # OCR внутри
    ocr_inside = _ocr_in_bbox(ocr_in_row, row_bbox)
    if not ocr_inside:
        result["reason"] = "no_ocr_inside"
        return result

    # OCR центрирован
    centered = _is_ocr_centered(ocr_inside, row_bbox)
    if not centered:
        result["reason"] = "ocr_not_centered"
        return result

    # Confidence: базовый 0.5 + бонусы
    conf = 0.5
    conf += 0.2 * min(1.0, (BUTTON_MAX_COLOR_STD - color_std) / BUTTON_MAX_COLOR_STD)
    conf += 0.15 if len(ocr_inside) == 1 else 0.05
    conf += 0.15 if row_w >= container_w * 0.5 else 0.0
    result["confidence"] = min(1.0, conf)
    result["reason"] = "detected"
    return result


def detect_checkbox_like(
    roi: Any,
    row_bbox: List[float],
    ocr_in_row: List[Dict[str, Any]],
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Детектор checkbox-like элемента.
    Условия: маленький квадрат (aspect ~1), размер 12–28px, рядом OCR (X-overlap > 30%).
    """
    result = {"type": "checkbox", "confidence": 0.0, "reason": "", "candidates": []}
    if roi is None or len(row_bbox) < 4:
        result["reason"] = "invalid_input"
        return result

    try:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < CHECKBOX_SIZE_MIN or w > CHECKBOX_SIZE_MAX:
                continue
            if h < CHECKBOX_SIZE_MIN or h > CHECKBOX_SIZE_MAX:
                continue
            aspect = w / max(1, h)
            if aspect < CHECKBOX_ASPECT_LO or aspect > CHECKBOX_ASPECT_HI:
                continue
            # Глобальные координаты
            gx1 = row_bbox[0] + x
            gy1 = row_bbox[1] + y
            gx2 = gx1 + w
            gy2 = gy1 + h
            candidates.append([gx1, gy1, gx2, gy2])

        if not candidates:
            result["reason"] = "no_square_found"
            return result

        # Проверка: рядом с кандидатом есть OCR (X-overlap > 30%)
        best_conf = 0.0
        for cand in candidates:
            cand_w = cand[2] - cand[0]
            for ob in ocr_in_row:
                b = ob.get("bbox") or []
                if len(b) < 4:
                    continue
                # X-overlap
                ix1 = max(cand[0], b[0])
                ix2 = min(cand[2], b[2])
                overlap = max(0, ix2 - ix1)
                ratio = overlap / cand_w if cand_w > 0 else 0.0
                if ratio >= CHECKBOX_OCR_X_OVERLAP or (cand[2] < b[0] and b[0] - cand[2] < 20):
                    conf = 0.6 + 0.2 * min(1.0, ratio)
                    best_conf = max(best_conf, conf)

        result["candidates"] = candidates
        if best_conf > 0:
            result["confidence"] = min(1.0, best_conf)
            result["reason"] = "detected"
        else:
            result["reason"] = "no_adjacent_ocr"
        return result
    except Exception as e:
        result["reason"] = f"error={e}"
        return result


def detect_radio_like(
    roi: Any,
    row_bbox: List[float],
) -> Dict[str, Any]:
    """
    Детектор radio-like элемента.
    Условия: маленький круг, диаметр 12–28px, высокая циркулярность.
    """
    result = {"type": "radio", "confidence": 0.0, "reason": "", "candidates": []}
    if roi is None or len(row_bbox) < 4:
        result["reason"] = "invalid_input"
        return result

    try:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < RADIO_CIRCULARITY_MIN:
                continue
            x, y, w, h = cv2.boundingRect(c)
            diameter = (w + h) / 2
            if diameter < RADIO_DIAMETER_MIN or diameter > RADIO_DIAMETER_MAX:
                continue
            gx1 = row_bbox[0] + x
            gy1 = row_bbox[1] + y
            gx2 = gx1 + w
            gy2 = gy1 + h
            candidates.append({"bbox": [gx1, gy1, gx2, gy2], "circularity": circularity})

        if not candidates:
            result["reason"] = "no_circle_found"
            return result

        best = max(candidates, key=lambda c: c["circularity"])
        result["candidates"] = [c["bbox"] for c in candidates]
        result["confidence"] = min(1.0, 0.5 + 0.5 * best["circularity"])
        result["reason"] = "detected"
        return result
    except Exception as e:
        result["reason"] = f"error={e}"
        return result


def detect_textarea_like(
    roi: Any,
    row_bbox: List[float],
    container_bbox: List[float],
    median_input_height: float,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Детектор textarea-like элемента.
    Условия: height ≥ 1.8×median, width ≥ 0.5 container, есть рамка/контур.
    """
    result = {"type": "textarea", "confidence": 0.0, "reason": ""}
    if roi is None or len(row_bbox) < 4 or len(container_bbox) < 4:
        result["reason"] = "invalid_input"
        return result

    row_h = row_bbox[3] - row_bbox[1]
    row_w = row_bbox[2] - row_bbox[0]
    container_w = container_bbox[2] - container_bbox[0]

    if median_input_height <= 0:
        median_input_height = 40.0

    # Высота ≥ 1.8×median
    height_ok = row_h >= median_input_height * TEXTAREA_HEIGHT_RATIO
    if not height_ok:
        result["reason"] = f"height_too_small={row_h:.0f}"
        return result

    # Ширина ≥ 0.5 container
    width_ok = container_w > 0 and row_w >= container_w * TEXTAREA_WIDTH_RATIO
    if not width_ok:
        result["reason"] = f"width_too_small={row_w:.0f}"
        return result

    # Проверка рамки/контура
    try:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        if h < 2 or w < 2:
            result["reason"] = "roi_too_small"
            return result
        band = max(2, h // 6)
        top_band = edges[:band, :]
        bottom_band = edges[-band:, :]
        cols_top = np.sum(np.sum(top_band > 0, axis=0) > 0)
        cols_bottom = np.sum(np.sum(bottom_band > 0, axis=0) > 0)
        frame_ratio = max(cols_top, cols_bottom) / w
        if frame_ratio < TEXTAREA_FRAME_RATIO:
            result["reason"] = f"no_frame={frame_ratio:.2f}"
            return result

        conf = 0.5 + 0.3 * min(1.0, (row_h / median_input_height) / 3.0)
        conf += 0.2 * frame_ratio
        result["confidence"] = min(1.0, conf)
        result["reason"] = "detected"
        return result
    except Exception as e:
        result["reason"] = f"error={e}"
        return result


def run_leaf_element_detection(
    rows: List[Any],
    image_path: str,
    raw_ocr_boxes: List[Dict[str, Any]],
    container_bbox: List[float],
    median_input_height: Optional[float] = None,
) -> None:
    """
    Stage 1: Диагностический слой LeafElementDetection.

    Для каждой строки:
    - Детектирует element-like паттерны внутри row.bbox
    - НЕ меняет row_type
    - НЕ меняет geometry (y_min, y_max, x_min, x_max)
    - НЕ меняет input_bbox / input_bboxes
    - НЕ меняет slots
    - Только добавляет metadata и логирует

    Args:
        rows: список FormRow
        image_path: путь к изображению
        raw_ocr_boxes: все OCR-блоки
        container_bbox: bbox контейнера формы
        median_input_height: медианная высота input (для textarea)
    """
    if not rows or not image_path or len(container_bbox) < 4:
        return

    # Вычислить median_input_height если не задан
    if median_input_height is None or median_input_height <= 0:
        heights = []
        for r in rows:
            ib = getattr(r, "input_bbox", None)
            if ib and len(ib) >= 4:
                heights.append(ib[3] - ib[1])
        median_input_height = sorted(heights)[len(heights) // 2] if heights else 40.0

    for r in rows:
        row_bbox = [r.x_min, r.y_min, r.x_max, r.y_max]
        roi = _crop_roi(image_path, row_bbox)

        # OCR в строке
        ocr_in_row = _ocr_in_bbox(raw_ocr_boxes, row_bbox)

        # Запуск детекторов
        button_res = detect_button_like(roi, row_bbox, container_bbox, ocr_in_row)
        checkbox_res = detect_checkbox_like(roi, row_bbox, ocr_in_row, image_path)
        radio_res = detect_radio_like(roi, row_bbox)
        textarea_res = detect_textarea_like(roi, row_bbox, container_bbox, median_input_height, image_path)

        # Собрать кандидатов с confidence > 0
        candidates = []
        for res in [button_res, checkbox_res, radio_res, textarea_res]:
            if res["confidence"] > 0:
                candidates.append({"type": res["type"], "confidence": res["confidence"]})

        # Debug info
        debug_info = {
            "roi_shape": roi.shape if roi is not None else None,
            "mean_color_variance": _color_variance(roi),
            "ocr_count": len(ocr_in_row),
            "detectors": {
                "button": {"confidence": button_res["confidence"], "reason": button_res["reason"]},
                "checkbox": {"confidence": checkbox_res["confidence"], "reason": checkbox_res["reason"]},
                "radio": {"confidence": radio_res["confidence"], "reason": radio_res["reason"]},
                "textarea": {"confidence": textarea_res["confidence"], "reason": textarea_res["reason"]},
            },
        }

        # Записать в metadata (НЕ меняя структуру строки)
        if not hasattr(r, "metadata") or r.metadata is None:
            r.metadata = {}
        r.metadata["leaf_candidates"] = candidates
        r.metadata["leaf_debug"] = debug_info

        # Логирование
        if candidates:
            cand_str = ", ".join(f"{c['type']}({c['confidence']:.2f})" for c in candidates)
            logger.debug("[LEAF] row_index=%d candidates=[%s]", r.row_index, cand_str)
        else:
            logger.debug("[LEAF] row_index=%d no candidates", r.row_index)
