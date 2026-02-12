"""
ContainerLeafDetection — диагностический слой (Stage 1.1).

Работает ДО row segmentation. Ищет UI-элементы внутри container_bbox,
которые могут быть потеряны при построении строк.

НЕ создаёт rows.
НЕ меняет container_bbox.
НЕ меняет layout.
НЕ влияет на downstream логику.

Только собирает диагностическую информацию.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Флаг диагностического режима
DEBUG_CONTAINER_LEAF = True

# Ослабленные пороги для container-level (диагностика)
CONTAINER_BUTTON_MIN_WIDTH_RATIO = 0.25  # было 0.4
CONTAINER_BUTTON_MAX_COLOR_STD = 65.0    # было 50 (+30%)
CONTAINER_BUTTON_CENTER_TOLERANCE = 0.5  # ослаблено

CONTAINER_CHECKBOX_SIZE_MIN = 10         # было 12
CONTAINER_CHECKBOX_SIZE_MAX = 32         # было 28

CONTAINER_RADIO_DIAMETER_MIN = 10        # было 12
CONTAINER_RADIO_DIAMETER_MAX = 32        # было 28

CONTAINER_TEXTAREA_HEIGHT_RATIO = 1.5    # было 1.8
CONTAINER_TEXTAREA_WIDTH_RATIO = 0.4     # было 0.5

# Порог IoU для определения пересечения с rows
IOU_THRESHOLD_INSIDE_ROW = 0.5

# Пороги для фильтрации OCR-областей (чтобы буквы не классифицировались как UI)
OCR_CONFIDENCE_THRESHOLD = 0.6   # минимальная уверенность OCR для фильтрации
OCR_BBOX_PADDING = 3             # расширение OCR bbox на N пикселей
IOU_THRESHOLD_OCR_GENERAL = 0.3  # компонент отклоняется если IoU с OCR >= 0.3
IOU_THRESHOLD_OCR_CHECKBOX_RADIO = 0.2  # для checkbox/radio строже: IoU >= 0.2

# Геометрические пороги для checkbox/radio (отличие от букв)
# Буквы плотные, checkbox/radio — пустые внутри
FILL_RATIO_MAX = 0.45            # foreground_pixels / bbox_area < 0.45
INNER_CONTOURS_MAX = 2           # количество внутренних контуров ≤ 2
EDGE_DENSITY_MAX = 0.25          # edge_pixels / bbox_area < 0.25

# Дополнительные геометрические требования для checkbox/radio
ASPECT_RATIO_MIN = 0.85          # checkbox/radio почти квадратные
ASPECT_RATIO_MAX = 1.15
SYMMETRY_SCORE_MIN = 0.8         # checkbox/radio симметричны
CHECKBOX_RADIO_MAX_SIZE = 40     # максимальный размер checkbox/radio в пикселях


def _crop_roi(image_path: str, bbox: List[float]) -> Optional[Any]:
    """Вырезает ROI из изображения по bbox."""
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
    """Стандартное отклонение по серому каналу."""
    try:
        import cv2
        import numpy as np
        if roi is None or roi.size == 0:
            return 999.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))
    except Exception:
        return 999.0


def _iou(box1: List[float], box2: List[float]) -> float:
    """Intersection over Union для двух bbox."""
    if len(box1) < 4 or len(box2) < 4:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


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


def _get_reliable_ocr_bboxes(
    ocr_boxes: List[Dict[str, Any]],
    min_confidence: float = OCR_CONFIDENCE_THRESHOLD,
    padding: int = OCR_BBOX_PADDING,
) -> List[List[float]]:
    """
    Возвращает bbox'ы надёжных OCR-блоков (confidence > порог) с расширением.
    Используется для исключения текстовых областей из анализа.
    """
    result = []
    for ob in ocr_boxes:
        conf = ob.get("confidence", 0.0)
        if conf < min_confidence:
            continue
        b = ob.get("bbox") or []
        if len(b) < 4:
            continue
        # Расширить bbox на padding пикселей
        padded = [
            b[0] - padding,
            b[1] - padding,
            b[2] + padding,
            b[3] + padding,
        ]
        result.append(padded)
    return result


def _max_iou_with_ocr(
    candidate_bbox: List[float],
    ocr_bboxes: List[List[float]],
) -> float:
    """Возвращает максимальный IoU кандидата с любым OCR bbox."""
    max_iou = 0.0
    for ocr_bbox in ocr_bboxes:
        iou = _iou(candidate_bbox, ocr_bbox)
        if iou > max_iou:
            max_iou = iou
    return max_iou


def _is_text_region(
    candidate_bbox: List[float],
    ocr_bboxes: List[List[float]],
    threshold: float = IOU_THRESHOLD_OCR_GENERAL,
) -> bool:
    """
    Проверяет, является ли кандидат текстовой областью.
    Возвращает True если IoU с любым OCR >= threshold.
    """
    return _max_iou_with_ocr(candidate_bbox, ocr_bboxes) >= threshold


def _compute_geometry_metrics(roi: Any) -> Dict[str, float]:
    """
    Вычисляет геометрические метрики для отличия UI-элементов от букв.

    Returns:
        {
            "fill_ratio": foreground_pixels / bbox_area,
            "inner_contours": количество внутренних контуров,
            "edge_density": edge_pixels / bbox_area,
            "aspect_ratio": width / height,
            "symmetry_score": мера симметричности (0-1),
            "max_dimension": max(width, height),
        }
    """
    result = {
        "fill_ratio": 1.0,
        "inner_contours": 99,
        "edge_density": 1.0,
        "aspect_ratio": 0.0,
        "symmetry_score": 0.0,
        "max_dimension": 999,
    }
    if roi is None:
        return result

    try:
        import cv2
        import numpy as np

        h, w = roi.shape[:2]
        bbox_area = h * w
        if bbox_area <= 0:
            return result

        # Aspect ratio
        result["aspect_ratio"] = w / h if h > 0 else 0.0
        result["max_dimension"] = max(w, h)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Fill ratio — плотность foreground
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        foreground_pixels = np.sum(th > 0)
        result["fill_ratio"] = foreground_pixels / bbox_area

        # 2. Inner contours — количество внутренних контуров
        # RETR_TREE даёт иерархию контуров
        contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None and len(hierarchy) > 0:
            # hierarchy[0][i] = [next, prev, child, parent]
            # parent == -1 означает внешний контур
            # parent != -1 означает внутренний (дочерний) контур
            inner_count = sum(1 for i in range(len(hierarchy[0])) if hierarchy[0][i][3] != -1)
            result["inner_contours"] = inner_count
        else:
            result["inner_contours"] = 0

        # 3. Edge density — плотность рёбер
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        result["edge_density"] = edge_pixels / bbox_area

        # 4. Symmetry score — горизонтальная и вертикальная симметрия
        # Сравниваем левую/правую и верхнюю/нижнюю части
        try:
            # Горизонтальная симметрия (left vs right flipped)
            mid_w = w // 2
            if mid_w > 0:
                left = th[:, :mid_w]
                right = th[:, -mid_w:]
                right_flipped = cv2.flip(right, 1)
                # Нормализуем размеры
                min_w = min(left.shape[1], right_flipped.shape[1])
                left = left[:, :min_w]
                right_flipped = right_flipped[:, :min_w]
                h_diff = np.sum(np.abs(left.astype(float) - right_flipped.astype(float)))
                h_symmetry = 1.0 - (h_diff / (255.0 * left.size)) if left.size > 0 else 0.0
            else:
                h_symmetry = 0.0

            # Вертикальная симметрия (top vs bottom flipped)
            mid_h = h // 2
            if mid_h > 0:
                top = th[:mid_h, :]
                bottom = th[-mid_h:, :]
                bottom_flipped = cv2.flip(bottom, 0)
                min_h = min(top.shape[0], bottom_flipped.shape[0])
                top = top[:min_h, :]
                bottom_flipped = bottom_flipped[:min_h, :]
                v_diff = np.sum(np.abs(top.astype(float) - bottom_flipped.astype(float)))
                v_symmetry = 1.0 - (v_diff / (255.0 * top.size)) if top.size > 0 else 0.0
            else:
                v_symmetry = 0.0

            # Среднее симметрии
            result["symmetry_score"] = (h_symmetry + v_symmetry) / 2.0
        except Exception:
            result["symmetry_score"] = 0.0

        return result
    except Exception:
        return result


def _is_letter_like(metrics: Dict[str, float]) -> Tuple[bool, str]:
    """
    Проверяет, похож ли объект на букву (а не на checkbox/radio).

    Checkbox/radio должны быть:
    - Низкий fill_ratio (пустые внутри)
    - Мало внутренних контуров
    - Низкая edge_density
    - Aspect ratio близок к 1.0 (квадрат/круг)
    - Высокая симметрия
    - Ограниченный размер

    Returns:
        (is_letter, reason)
    """
    fill_ratio = metrics.get("fill_ratio", 1.0)
    inner_contours = metrics.get("inner_contours", 99)
    edge_density = metrics.get("edge_density", 1.0)
    aspect_ratio = metrics.get("aspect_ratio", 0.0)
    symmetry_score = metrics.get("symmetry_score", 0.0)
    max_dimension = metrics.get("max_dimension", 999)

    # 1. Fill ratio — буквы плотные
    if fill_ratio > FILL_RATIO_MAX:
        return True, f"fill_ratio={fill_ratio:.2f}>{FILL_RATIO_MAX}"

    # 2. Inner contours — буквы имеют много внутренних контуров
    if inner_contours > INNER_CONTOURS_MAX:
        return True, f"inner_contours={inner_contours}>{INNER_CONTOURS_MAX}"

    # 3. Edge density — буквы дают плотную сетку рёбер
    if edge_density > EDGE_DENSITY_MAX:
        return True, f"edge_density={edge_density:.2f}>{EDGE_DENSITY_MAX}"

    # 4. Aspect ratio — checkbox/radio почти квадратные (0.85-1.15)
    if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
        return True, f"aspect_ratio={aspect_ratio:.2f} not in [{ASPECT_RATIO_MIN},{ASPECT_RATIO_MAX}]"

    # 5. Symmetry — checkbox/radio симметричны
    if symmetry_score < SYMMETRY_SCORE_MIN:
        return True, f"symmetry_score={symmetry_score:.2f}<{SYMMETRY_SCORE_MIN}"

    # 6. Size — checkbox/radio имеют ограниченный размер
    if max_dimension > CHECKBOX_RADIO_MAX_SIZE:
        return True, f"max_dimension={max_dimension}>{CHECKBOX_RADIO_MAX_SIZE}"

    return False, ""


def _find_connected_components(
    image_path: str,
    container_bbox: List[float],
    min_area: int = 100,
    max_area: int = 50000,
) -> List[List[float]]:
    """
    Находит connected components (контуры) внутри container_bbox.
    Возвращает список bbox [x1, y1, x2, y2] в глобальных координатах.
    """
    if not image_path or len(container_bbox) < 4:
        return []
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            return []

        cx1, cy1, cx2, cy2 = (
            int(container_bbox[0]),
            int(container_bbox[1]),
            int(container_bbox[2]),
            int(container_bbox[3]),
        )
        roi = img[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Адаптивный threshold для обнаружения UI-элементов
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Также попробуем edges
        edges = cv2.Canny(gray, 50, 150)
        combined = cv2.bitwise_or(th, edges)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # Фильтр по aspect ratio (не слишком узкие)
            aspect = w / max(1, h)
            if aspect < 0.1 or aspect > 10:
                continue
            # Глобальные координаты
            gx1, gy1 = cx1 + x, cy1 + y
            gx2, gy2 = gx1 + w, gy1 + h
            bboxes.append([gx1, gy1, gx2, gy2])

        return bboxes
    except Exception as e:
        logger.debug("[CONTAINER_LEAF] _find_connected_components error: %s", e)
        return []


def _detect_button_like_container(
    roi: Any,
    bbox: List[float],
    container_bbox: List[float],
    ocr_in_bbox_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Детектор button-like с ослабленными порогами."""
    result = {"type": "button", "confidence": 0.0, "bbox": bbox, "reason": ""}
    if roi is None or len(bbox) < 4 or len(container_bbox) < 4:
        result["reason"] = "invalid_input"
        return result

    container_w = container_bbox[2] - container_bbox[0]
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    if container_w <= 0:
        result["reason"] = "zero_container_width"
        return result

    # Ширина ≥ 0.25 контейнера (ослаблено)
    if bbox_w < container_w * CONTAINER_BUTTON_MIN_WIDTH_RATIO:
        result["reason"] = "width_too_small"
        return result

    # Aspect ratio: кнопки обычно широкие и невысокие
    if bbox_h > 0 and bbox_w / bbox_h < 1.5:
        result["reason"] = "aspect_not_button_like"
        return result

    # Низкая цветовая дисперсия (ослаблено)
    color_std = _color_variance(roi)
    if color_std > CONTAINER_BUTTON_MAX_COLOR_STD:
        result["reason"] = f"high_color_variance={color_std:.1f}"
        return result

    # OCR внутри (не строго центрирован)
    if not ocr_in_bbox_list:
        result["reason"] = "no_ocr_inside"
        return result

    # Confidence
    conf = 0.4
    conf += 0.2 * min(1.0, (CONTAINER_BUTTON_MAX_COLOR_STD - color_std) / CONTAINER_BUTTON_MAX_COLOR_STD)
    conf += 0.2 if len(ocr_in_bbox_list) == 1 else 0.1
    conf += 0.2 if bbox_w >= container_w * 0.4 else 0.0
    result["confidence"] = min(1.0, conf)
    result["reason"] = "detected"
    return result


def _detect_checkbox_like_container(
    roi: Any,
    bbox: List[float],
    ocr_boxes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Детектор checkbox-like с геометрическими фильтрами."""
    result = {"type": "checkbox", "confidence": 0.0, "bbox": bbox, "reason": ""}
    if roi is None or len(bbox) < 4:
        result["reason"] = "invalid_input"
        return result

    try:
        import cv2
        import numpy as np

        # 1. Геометрическая проверка — исключить буквы
        metrics = _compute_geometry_metrics(roi)
        is_letter, letter_reason = _is_letter_like(metrics)
        if is_letter:
            result["reason"] = f"letter_like:{letter_reason}"
            return result

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < CONTAINER_CHECKBOX_SIZE_MIN or w > CONTAINER_CHECKBOX_SIZE_MAX:
                continue
            if h < CONTAINER_CHECKBOX_SIZE_MIN or h > CONTAINER_CHECKBOX_SIZE_MAX:
                continue
            aspect = w / max(1, h)
            if aspect < 0.7 or aspect > 1.3:
                continue

            # 2. Проверить геометрию кандидата (вырезанного контура)
            cand_roi = roi[y:y+h, x:x+w] if y+h <= roi.shape[0] and x+w <= roi.shape[1] else None
            if cand_roi is not None and cand_roi.size > 0:
                cand_metrics = _compute_geometry_metrics(cand_roi)
                cand_is_letter, cand_reason = _is_letter_like(cand_metrics)
                if cand_is_letter:
                    continue  # этот контур — буква, пропустить

            # Найден квадрат — проверить OCR рядом
            gx1 = bbox[0] + x
            gy1 = bbox[1] + y
            for ob in ocr_boxes:
                b = ob.get("bbox") or []
                if len(b) < 4:
                    continue
                # OCR справа от checkbox
                if b[0] > gx1 and b[0] - (gx1 + w) < 30:
                    result["confidence"] = 0.65
                    result["reason"] = "detected"
                    return result

        result["reason"] = "no_checkbox_pattern"
        return result
    except Exception as e:
        result["reason"] = f"error={e}"
        return result


def _detect_radio_like_container(
    roi: Any,
    bbox: List[float],
) -> Dict[str, Any]:
    """Детектор radio-like с геометрическими фильтрами."""
    result = {"type": "radio", "confidence": 0.0, "bbox": bbox, "reason": ""}
    if roi is None or len(bbox) < 4:
        result["reason"] = "invalid_input"
        return result

    try:
        import cv2
        import numpy as np

        # 1. Геометрическая проверка — исключить буквы
        metrics = _compute_geometry_metrics(roi)
        is_letter, letter_reason = _is_letter_like(metrics)
        if is_letter:
            result["reason"] = f"letter_like:{letter_reason}"
            return result

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.6:  # ослаблено с 0.7
                continue
            x, y, w, h = cv2.boundingRect(c)
            diameter = (w + h) / 2
            if diameter < CONTAINER_RADIO_DIAMETER_MIN or diameter > CONTAINER_RADIO_DIAMETER_MAX:
                continue

            # 2. Проверить геометрию кандидата (вырезанного контура)
            cand_roi = roi[y:y+h, x:x+w] if y+h <= roi.shape[0] and x+w <= roi.shape[1] else None
            if cand_roi is not None and cand_roi.size > 0:
                cand_metrics = _compute_geometry_metrics(cand_roi)
                cand_is_letter, cand_reason = _is_letter_like(cand_metrics)
                if cand_is_letter:
                    continue  # этот контур — буква, пропустить

            result["confidence"] = min(1.0, 0.4 + 0.5 * circularity)
            result["reason"] = "detected"
            return result

        result["reason"] = "no_circle_found"
        return result
    except Exception as e:
        result["reason"] = f"error={e}"
        return result


def _detect_textarea_like_container(
    roi: Any,
    bbox: List[float],
    container_bbox: List[float],
    median_input_height: float,
) -> Dict[str, Any]:
    """Детектор textarea-like с ослабленными порогами."""
    result = {"type": "textarea", "confidence": 0.0, "bbox": bbox, "reason": ""}
    if roi is None or len(bbox) < 4 or len(container_bbox) < 4:
        result["reason"] = "invalid_input"
        return result

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]
    container_w = container_bbox[2] - container_bbox[0]

    if median_input_height <= 0:
        median_input_height = 40.0

    # Высота ≥ 1.5×median (ослаблено)
    if bbox_h < median_input_height * CONTAINER_TEXTAREA_HEIGHT_RATIO:
        result["reason"] = f"height_too_small={bbox_h:.0f}"
        return result

    # Ширина ≥ 0.4 container (ослаблено)
    if container_w > 0 and bbox_w < container_w * CONTAINER_TEXTAREA_WIDTH_RATIO:
        result["reason"] = f"width_too_small={bbox_w:.0f}"
        return result

    # Проверка рамки
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
        if frame_ratio < 0.4:  # ослаблено
            result["reason"] = f"no_frame={frame_ratio:.2f}"
            return result

        conf = 0.4 + 0.3 * min(1.0, (bbox_h / median_input_height) / 3.0)
        conf += 0.2 * frame_ratio
        result["confidence"] = min(1.0, conf)
        result["reason"] = "detected"
        return result
    except Exception as e:
        result["reason"] = f"error={e}"
        return result


def run_container_leaf_detection(
    container_bbox: List[float],
    image_path: str,
    raw_ocr_boxes: List[Dict[str, Any]],
    rows: Optional[List[Any]] = None,
    median_input_height: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Stage 1.1: ContainerLeafDetection — диагностический слой.

    Ищет UI-элементы внутри container_bbox ДО row segmentation.
    НЕ создаёт rows, НЕ меняет container, НЕ влияет на downstream.

    Args:
        container_bbox: bbox контейнера [x1, y1, x2, y2]
        image_path: путь к изображению
        raw_ocr_boxes: все OCR-блоки
        rows: список FormRow (для определения inside/outside rows)
        median_input_height: медианная высота input (для textarea)

    Returns:
        {
            "all_candidates": [...],
            "inside_rows": [...],
            "outside_rows": [...],
        }
    """
    result = {
        "all_candidates": [],
        "inside_rows": [],
        "outside_rows": [],
    }

    if not image_path or len(container_bbox) < 4:
        logger.debug("[CONTAINER_LEAF] no candidates (invalid input)")
        return result

    if not DEBUG_CONTAINER_LEAF:
        return result

    # Вычислить median_input_height если не задан
    if median_input_height is None or median_input_height <= 0:
        median_input_height = 40.0

    # Найти connected components внутри container
    components = _find_connected_components(image_path, container_bbox)
    if not components:
        logger.debug("[CONTAINER_LEAF] no candidates (no components)")
        return result

    # OCR внутри container
    ocr_in_container = _ocr_in_bbox(raw_ocr_boxes, container_bbox)

    # Построить список надёжных OCR bbox для исключения текстовых областей
    reliable_ocr_bboxes = _get_reliable_ocr_bboxes(raw_ocr_boxes)

    all_candidates = []
    skipped_as_text = 0

    for comp_bbox in components:
        # 1. Исключить компоненты с высоким IoU с OCR (это текст, не UI)
        if _is_text_region(comp_bbox, reliable_ocr_bboxes, IOU_THRESHOLD_OCR_GENERAL):
            skipped_as_text += 1
            continue

        roi = _crop_roi(image_path, comp_bbox)
        if roi is None:
            continue

        ocr_in_comp = _ocr_in_bbox(ocr_in_container, comp_bbox)

        # Запустить все детекторы
        detections = [
            _detect_button_like_container(roi, comp_bbox, container_bbox, ocr_in_comp),
            _detect_checkbox_like_container(roi, comp_bbox, ocr_in_container),
            _detect_radio_like_container(roi, comp_bbox),
            _detect_textarea_like_container(roi, comp_bbox, container_bbox, median_input_height),
        ]

        # Взять лучший результат
        best = max(detections, key=lambda d: d.get("confidence", 0))
        if best["confidence"] > 0:
            # 2. Дополнительная проверка для checkbox/radio — строже по OCR
            if best["type"] in ("checkbox", "radio"):
                if _is_text_region(comp_bbox, reliable_ocr_bboxes, IOU_THRESHOLD_OCR_CHECKBOX_RADIO):
                    skipped_as_text += 1
                    continue

            all_candidates.append({
                "type": best["type"],
                "confidence": best["confidence"],
                "bbox": comp_bbox,
                "reason": best["reason"],
            })

    if skipped_as_text > 0:
        logger.debug("[CONTAINER_LEAF] skipped %d components as text regions", skipped_as_text)

    result["all_candidates"] = all_candidates

    # Разделить на inside_rows / outside_rows
    if rows:
        row_bboxes = []
        for r in rows:
            if hasattr(r, "x_min") and hasattr(r, "y_min"):
                row_bboxes.append([r.x_min, r.y_min, r.x_max, r.y_max])

        for cand in all_candidates:
            cand_bbox = cand.get("bbox", [])
            is_inside = False
            for rb in row_bboxes:
                if _iou(cand_bbox, rb) >= IOU_THRESHOLD_INSIDE_ROW:
                    is_inside = True
                    break
            if is_inside:
                result["inside_rows"].append(cand)
            else:
                result["outside_rows"].append(cand)
    else:
        # Без rows все кандидаты считаются outside
        result["outside_rows"] = all_candidates

    # Логирование
    if result["outside_rows"]:
        types_str = ", ".join(
            f"{c['type']}({c['confidence']:.2f})" for c in result["outside_rows"]
        )
        logger.debug(
            "[CONTAINER_LEAF] outside_rows=%d types=[%s]",
            len(result["outside_rows"]),
            types_str,
        )
    elif all_candidates:
        logger.debug(
            "[CONTAINER_LEAF] all_inside_rows=%d",
            len(result["inside_rows"]),
        )
    else:
        logger.debug("[CONTAINER_LEAF] no candidates")

    return result


def update_container_leaf_with_rows(
    container_leaf_result: Dict[str, Any],
    rows: List[Any],
) -> Dict[str, Any]:
    """
    Обновляет результат ContainerLeafDetection после построения rows.
    Пересчитывает inside_rows / outside_rows.

    Вызывается ПОСЛЕ FormInnerLayout, чтобы определить какие элементы
    были вырезаны layout.
    """
    if not container_leaf_result or not rows:
        return container_leaf_result

    all_candidates = container_leaf_result.get("all_candidates", [])
    if not all_candidates:
        return container_leaf_result

    row_bboxes = []
    for r in rows:
        if hasattr(r, "x_min") and hasattr(r, "y_min"):
            row_bboxes.append([r.x_min, r.y_min, r.x_max, r.y_max])

    inside_rows = []
    outside_rows = []

    for cand in all_candidates:
        cand_bbox = cand.get("bbox", [])
        is_inside = False
        for rb in row_bboxes:
            if _iou(cand_bbox, rb) >= IOU_THRESHOLD_INSIDE_ROW:
                is_inside = True
                break
        if is_inside:
            inside_rows.append(cand)
        else:
            outside_rows.append(cand)

    container_leaf_result["inside_rows"] = inside_rows
    container_leaf_result["outside_rows"] = outside_rows

    # Логирование обновления
    if outside_rows:
        types_str = ", ".join(
            f"{c['type']}({c['confidence']:.2f})" for c in outside_rows
        )
        logger.debug(
            "[CONTAINER_LEAF] updated: outside_rows=%d types=[%s]",
            len(outside_rows),
            types_str,
        )

    return container_leaf_result
