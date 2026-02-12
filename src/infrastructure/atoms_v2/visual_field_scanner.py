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

# Расширенный режим (detect_all) — ищет все визуальные элементы, не только input fields
VISUAL_ELEMENT_MIN_AREA = 150
VISUAL_ELEMENT_HEIGHT_MIN = 16
VISUAL_ELEMENT_HEIGHT_MAX = 250
VISUAL_ELEMENT_WIDTH_MIN = 30
VISUAL_ELEMENT_ASPECT_MAX = 15.0  # не слишком вытянутые

# Checkbox/Radio detection
CHECKBOX_RADIO_SIZE_MIN = 12
CHECKBOX_RADIO_SIZE_MAX = 32
CHECKBOX_RADIO_ASPECT_TOLERANCE = 0.3  # aspect должен быть близок к 1.0

# Textarea vs Input threshold
TEXTAREA_HEIGHT_MIN = 60
INPUT_HEIGHT_TYPICAL = 28
INPUT_HEIGHT_MAX = 50
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


def _classify_element_type(w: int, h: int, crop_w: int, is_colored: bool, has_border: bool) -> Tuple[str, float]:
    """
    Классификация типа элемента по геометрии и характеристикам.
    
    Приоритет:
    1. checkbox/radio — маленькие квадратные
    2. button — цветные, небольшие
    3. textarea — высокие (>80px), без вложенных элементов
    4. input — горизонтально вытянутые, высота 28-60px
    5. section — большие области
    """
    aspect = w / max(1, h)
    width_ratio = w / max(1, crop_w)
    
    # 1. Checkbox/Radio: маленькие квадратные элементы (ВЫСШИЙ ПРИОРИТЕТ)
    if (CHECKBOX_RADIO_SIZE_MIN <= w <= CHECKBOX_RADIO_SIZE_MAX and 
        CHECKBOX_RADIO_SIZE_MIN <= h <= CHECKBOX_RADIO_SIZE_MAX and
        abs(aspect - 1.0) <= CHECKBOX_RADIO_ASPECT_TOLERANCE):
        return "checkbox", 0.85
    
    # 2. Button: цветной элемент, высота 25-75px, aspect 1.5-6
    if is_colored and 25 <= h <= 75 and 1.5 <= aspect <= 6.0 and width_ratio <= 0.5:
        return "button", 0.8
    
    # 3. Textarea: высокий элемент (>80px), aspect < 4, широкий
    if h >= 80 and aspect < 4.0 and width_ratio >= 0.25:
        return "textarea", 0.75
    
    # 4. Input: горизонтально вытянутый, высота 28-60px
    if 28 <= h <= 60 and aspect >= 3.0:
        return "input", 0.75
    
    # Дополнительно: широкое поле с рамкой
    if has_border and aspect >= 2.5 and width_ratio >= 0.3:
        if h >= 80:
            return "textarea", 0.65
        if 25 <= h <= 65:
            return "input", 0.65
    
    # 5. Button fallback: средний aspect, цветной или с рамкой
    if 1.5 <= aspect <= 5.0 and 25 <= h <= 60 and (is_colored or has_border) and width_ratio <= 0.4:
        return "button", 0.5
    
    # 6. Section: большие прямоугольные области (контейнеры)
    if w >= 100 and h >= 60 and aspect < 3.0 and width_ratio >= 0.4:
        return "section", 0.4
    
    # 7. Label: широкие низкие элементы без рамки
    if aspect > 4.0 and h < 30 and width_ratio >= 0.3:
        return "label", 0.3
    
    return "element", 0.25


def _scan_all_visual_elements_roi(
    crop: "Any",  # np.ndarray BGR
    roi_x0: int,
    roi_y0: int,
    dark_theme: bool,
) -> List[Dict[str, Any]]:
    """
    Расширенный скан — находит все визуальные элементы (кнопки, поля, секции, checkbox/radio).
    Возвращает list of {bbox, element_type, confidence}.
    """
    import cv2
    import numpy as np

    if crop is None or crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = dark_theme or _is_dark_theme(gray)
    crop_h, crop_w = gray.shape[:2]

    results: List[Dict[str, Any]] = []
    
    def _is_duplicate(bbox: List[float], threshold: float = 0.5) -> bool:
        for existing in results:
            eb = existing["bbox"]
            ix1 = max(bbox[0], eb[0])
            iy1 = max(bbox[1], eb[1])
            ix2 = min(bbox[2], eb[2])
            iy2 = min(bbox[3], eb[3])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_a = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area_b = (eb[2] - eb[0]) * (eb[3] - eb[1])
                iou = inter / max(1, area_a + area_b - inter)
                if iou > threshold:
                    return True
        return False

    # 0) Детекция checkbox/radio (маленькие квадратные элементы)
    # Ищем в бинарном изображении
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if dark:
        binary = cv2.bitwise_not(binary)
    
    # Находим маленькие контуры
    contours_small, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_small:
        x, y, w, h = cv2.boundingRect(c)
        if (CHECKBOX_RADIO_SIZE_MIN <= w <= CHECKBOX_RADIO_SIZE_MAX and
            CHECKBOX_RADIO_SIZE_MIN <= h <= CHECKBOX_RADIO_SIZE_MAX):
            aspect = w / max(1, h)
            if abs(aspect - 1.0) <= CHECKBOX_RADIO_ASPECT_TOLERANCE:
                # Проверяем, что это не часть текста (должен быть относительно изолирован)
                bbox = [float(roi_x0 + x), float(roi_y0 + y), float(roi_x0 + x + w), float(roi_y0 + y + h)]
                if not _is_duplicate(bbox, 0.3):
                    results.append({
                        "bbox": bbox,
                        "element_type": "checkbox",
                        "confidence": 0.6,
                        "source": "checkbox_detection",
                    })

    # 1) Поиск через цветовую сегментацию (контрастные области — кнопки)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    # Цветные области (saturation > 25) — кнопки, иконки
    _, color_mask = cv2.threshold(saturation, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    color_closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    contours_color, _ = cv2.findContours(color_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours_color:
        area = cv2.contourArea(c)
        if area < VISUAL_ELEMENT_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h < VISUAL_ELEMENT_HEIGHT_MIN or h > VISUAL_ELEMENT_HEIGHT_MAX:
            continue
        if w < VISUAL_ELEMENT_WIDTH_MIN:
            continue
        aspect = w / max(1, h)
        if aspect > VISUAL_ELEMENT_ASPECT_MAX:
            continue
        
        elem_type, conf = _classify_element_type(w, h, crop_w, is_colored=True, has_border=False)
        bbox = [float(roi_x0 + x), float(roi_y0 + y), float(roi_x0 + x + w), float(roi_y0 + y + h)]
        
        if not _is_duplicate(bbox):
            results.append({
                "bbox": bbox,
                "element_type": elem_type,
                "confidence": conf,
                "source": "color_segmentation",
            })

    # 2) Поиск через границы (контуры) — поля ввода, textarea, секции
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours_edges, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours_edges:
        area = cv2.contourArea(c)
        if area < VISUAL_ELEMENT_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h < VISUAL_ELEMENT_HEIGHT_MIN or h > VISUAL_ELEMENT_HEIGHT_MAX:
            continue
        if w < VISUAL_ELEMENT_WIDTH_MIN:
            continue
        aspect = w / max(1, h)
        if aspect > VISUAL_ELEMENT_ASPECT_MAX:
            continue
        
        # Проверка на прямоугольность
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) < 4:
            continue
        
        elem_type, conf = _classify_element_type(w, h, crop_w, is_colored=False, has_border=True)
        bbox = [float(roi_x0 + x), float(roi_y0 + y), float(roi_x0 + x + w), float(roi_y0 + y + h)]
        
        if not _is_duplicate(bbox):
            results.append({
                "bbox": bbox,
                "element_type": elem_type,
                "confidence": conf,
                "source": "edge_detection",
            })

    # 3) Дополнительно: поиск секций через адаптивный порог
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )
    if dark:
        thresh = cv2.bitwise_not(thresh)
    
    # Морфологическое закрытие для объединения близких элементов
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large)
    
    contours_thresh, _ = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours_thresh:
        area = cv2.contourArea(c)
        if area < 500:  # секции обычно больше
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h < 30 or h > crop_h * 0.8:  # не слишком маленькие и не вся форма
            continue
        if w < crop_w * 0.3:  # секции обычно широкие
            continue
        
        aspect = w / max(1, h)
        if aspect > 10:  # не слишком вытянутые
            continue
        
        elem_type, conf = _classify_element_type(w, h, crop_w, is_colored=False, has_border=True)
        bbox = [float(roi_x0 + x), float(roi_y0 + y), float(roi_x0 + x + w), float(roi_y0 + y + h)]
        
        if not _is_duplicate(bbox):
            results.append({
                "bbox": bbox,
                "element_type": elem_type,
                "confidence": conf,
                "source": "adaptive_threshold",
            })

    return results


def scan_all_visual_elements(
    image_path: str,
    container_bbox: List[float],
    dark_theme: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Сканирует все визуальные элементы внутри container_bbox.
    Возвращает (list of element dicts, stats).
    """
    import cv2

    stats: Dict[str, Any] = {"elements_found": 0}
    if not image_path or len(container_bbox) < 4:
        return [], stats

    img = cv2.imread(str(image_path))
    if img is None:
        return [], stats
    img_h, img_w = img.shape[:2]
    
    if not dark_theme:
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark_theme = _is_dark_theme(gray_full)

    x1 = max(0, int(container_bbox[0]))
    y1 = max(0, int(container_bbox[1]))
    x2 = min(img_w, int(container_bbox[2]))
    y2 = min(img_h, int(container_bbox[3]))
    
    if x2 <= x1 or y2 <= y1:
        return [], stats
    
    crop = img[y1:y2, x1:x2]
    elements = _scan_all_visual_elements_roi(crop, x1, y1, dark_theme)
    
    # Фильтруем элементы внутри container_bbox
    valid_elements = []
    for elem in elements:
        b = elem["bbox"]
        if b[0] >= container_bbox[0] and b[2] <= container_bbox[2]:
            if b[1] >= container_bbox[1] and b[3] <= container_bbox[3]:
                valid_elements.append(elem)
    
    stats["elements_found"] = len(valid_elements)
    return valid_elements, stats


def postprocess_visual_elements(
    elements: List[Dict[str, Any]],
    container_bbox: List[float],
    median_text_height: float = 20.0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Пост-обработка визуальных элементов:
    1. Разделяет контейнеры (содержащие другие элементы) от leaf
    2. Убирает вложенные label
    3. Корректирует классификацию textarea vs input
    4. Применяет NMS с логированием
    
    Returns:
        processed_elements: обработанные элементы
        log_lines: диагностические логи
    """
    if not elements:
        return [], []
    
    log_lines = []
    container_w = container_bbox[2] - container_bbox[0] if len(container_bbox) >= 4 else 1000
    
    # 1. Помечаем элементы, которые содержат другие элементы (контейнеры)
    # ВАЖНО: только большие элементы могут быть контейнерами
    # Input, textarea, button, checkbox НЕ должны быть контейнерами
    container_indices = set()
    MIN_CONTAINER_AREA = 10000  # минимальная площадь контейнера (100x100)
    NON_CONTAINER_TYPES = {"input", "textarea", "button", "checkbox", "radio"}
    
    for i, e1 in enumerate(elements):
        b1 = e1.get("bbox", [])
        if len(b1) < 4:
            continue
        
        # Элементы конкретных типов не могут быть контейнерами
        e1_type = e1.get("element_type", "")
        if e1_type in NON_CONTAINER_TYPES:
            continue
        
        # Маленькие элементы не могут быть контейнерами
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        if area1 < MIN_CONTAINER_AREA:
            continue
        
        for j, e2 in enumerate(elements):
            if i == j:
                continue
            b2 = e2.get("bbox", [])
            if len(b2) < 4:
                continue
            
            # Контейнер должен быть значительно больше содержимого
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            if area1 < area2 * 1.5:  # контейнер минимум в 1.5 раза больше
                continue
            
            # Проверяем, содержит ли b1 bbox b2 (>80% площади b2 внутри b1)
            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2])
            iy2 = min(b1[3], b2[3])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / max(1, area2) >= 0.85:  # повышен порог с 0.8 до 0.85
                    container_indices.add(i)
                    e1["is_container"] = True
                    log_lines.append(f"Element [{i}] ({e1_type}) contains [{j}] -> marked as container")
                    break
    
    # 2. Убираем вложенные элементы, которые являются label и покрывают input/textarea
    filtered = []
    removed_indices = set()
    
    for i, elem in enumerate(elements):
        if i in removed_indices:
            continue
        
        b = elem.get("bbox", [])
        if len(b) < 4:
            continue
        
        etype = elem.get("element_type", "")
        h = b[3] - b[1]
        w = b[2] - b[0]
        
        # Проверка на слишком высокий label (>2x median_text_height)
        if etype == "label" and h > 2 * median_text_height:
            log_lines.append(f"Label [{i}] too tall ({h:.0f} > {2*median_text_height:.0f}), checking for nested elements")
            # Проверяем, есть ли внутри реальные элементы
            has_nested = False
            for j, e2 in enumerate(elements):
                if i == j or j in removed_indices:
                    continue
                b2 = e2.get("bbox", [])
                if len(b2) < 4:
                    continue
                # Проверяем, что b2 внутри b
                if b[0] <= b2[0] and b2[2] <= b[2] and b[1] <= b2[1] and b2[3] <= b[3]:
                    has_nested = True
                    break
            if has_nested:
                # Это контейнер строки, не label
                elem["element_type"] = "row_container"
                elem["is_container"] = True
                log_lines.append(f"Label [{i}] reclassified as row_container")
        
        # 3. Если section содержит другие элементы — это контейнер
        if etype == "section" and i in container_indices:
            elem["element_type"] = "section_container"
            elem["is_container"] = True
        
        # 4. Textarea с вложенными элементами — это контейнер
        if etype == "textarea" and i in container_indices:
            # Проверяем, что внутри есть реальные leaf-элементы
            has_leaf_inside = False
            for j, e2 in enumerate(elements):
                if i == j or j in removed_indices:
                    continue
                if j not in container_indices:
                    b2 = e2.get("bbox", [])
                    if len(b2) >= 4:
                        if b[0] <= b2[0] and b2[2] <= b[2] and b[1] <= b2[1] and b2[3] <= b[3]:
                            has_leaf_inside = True
                            break
            if has_leaf_inside:
                elem["element_type"] = "textarea_container"
                elem["is_container"] = True
                log_lines.append(f"Textarea [{i}] has leaf inside -> textarea_container")
        
        filtered.append(elem)
    
    # 5. NMS с логированием
    # Сортируем по confidence и размеру (приоритет маленьким leaf-элементам)
    def sort_key(e):
        is_cont = 1 if e.get("is_container") else 0
        conf = e.get("confidence", 0)
        b = e.get("bbox", [0, 0, 0, 0])
        area = (b[2] - b[0]) * (b[3] - b[1]) if len(b) >= 4 else 0
        return (is_cont, -conf, area)  # контейнеры последние, высокий confidence первый, маленькие первые
    
    sorted_elements = sorted(filtered, key=sort_key)
    
    final = []
    suppressed = []
    
    for elem in sorted_elements:
        b = elem.get("bbox", [])
        if len(b) < 4:
            continue
        
        # Проверяем перекрытие с уже добавленными
        is_suppressed = False
        for kept in final:
            kb = kept.get("bbox", [])
            if len(kb) < 4:
                continue
            
            # IoU
            ix1 = max(b[0], kb[0])
            iy1 = max(b[1], kb[1])
            ix2 = min(b[2], kb[2])
            iy2 = min(b[3], kb[3])
            
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_a = (b[2] - b[0]) * (b[3] - b[1])
                area_b = (kb[2] - kb[0]) * (kb[3] - kb[1])
                iou = inter / max(1, area_a + area_b - inter)
                
                # Более строгий порог для разных типов
                threshold = 0.3 if elem.get("element_type") in ("checkbox", "radio", "button") else 0.5
                
                if iou > threshold:
                    is_suppressed = True
                    suppressed.append(
                        f"Suppressed {elem.get('element_type', '?')} by {kept.get('element_type', '?')} (iou={iou:.2f})"
                    )
                    break
        
        if not is_suppressed:
            final.append(elem)
    
    log_lines.extend(suppressed)
    log_lines.append(f"Postprocess: {len(elements)} -> {len(final)} elements ({len(suppressed)} suppressed)")
    
    return final, log_lines


def detect_checkbox_radio_priority(
    image_path: str,
    container_bbox: List[float],
    dark_theme: bool = False,
) -> List[Dict[str, Any]]:
    """
    Приоритетная детекция checkbox/radio перед другими элементами.
    Находит маленькие квадратные элементы (12-32px).
    """
    import cv2
    import numpy as np
    
    if not image_path or len(container_bbox) < 4:
        return []
    
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    img_h, img_w = img.shape[:2]
    x1 = max(0, int(container_bbox[0]))
    y1 = max(0, int(container_bbox[1]))
    x2 = min(img_w, int(container_bbox[2]))
    y2 = min(img_h, int(container_bbox[3]))
    
    if x2 <= x1 or y2 <= y1:
        return []
    
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Бинаризация
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if dark_theme:
        binary = cv2.bitwise_not(binary)
    
    results = []
    
    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Проверяем размер (12-32px)
        if not (CHECKBOX_RADIO_SIZE_MIN <= w <= CHECKBOX_RADIO_SIZE_MAX):
            continue
        if not (CHECKBOX_RADIO_SIZE_MIN <= h <= CHECKBOX_RADIO_SIZE_MAX):
            continue
        
        # Проверяем aspect (близок к 1)
        aspect = w / max(1, h)
        if abs(aspect - 1.0) > CHECKBOX_RADIO_ASPECT_TOLERANCE:
            continue
        
        # Проверяем заполненность (checkbox/radio не полностью заполнены)
        area = cv2.contourArea(c)
        rect_area = w * h
        fill_ratio = area / max(1, rect_area)
        
        # Checkbox обычно имеет fill_ratio < 0.5, radio — круглый контур
        if fill_ratio > 0.7:
            continue  # Скорее всего буква
        
        # Определяем тип: checkbox (прямоугольный) или radio (круглый)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / max(1, perimeter ** 2) if perimeter > 0 else 0
        
        elem_type = "radio" if circularity > 0.7 else "checkbox"
        
        bbox = [float(x1 + x), float(y1 + y), float(x1 + x + w), float(y1 + y + h)]
        results.append({
            "bbox": bbox,
            "element_type": elem_type,
            "confidence": 0.8,
            "source": "checkbox_radio_detection",
            "fill_ratio": fill_ratio,
            "circularity": circularity,
        })
    
    return results
