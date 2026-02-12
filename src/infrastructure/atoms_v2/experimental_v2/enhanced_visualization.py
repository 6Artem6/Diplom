"""
Усиленная контрастная визуализация для отладки детекции.

Каждый тип элемента имеет уникальный стиль:
- HEADER: толстая синяя рамка 4px, без заливки
- INPUT: зелёная заливка 40%, толстая граница
- TEXTAREA: оранжевая заливка 40%, пунктирная рамка
- ACTION: красная заливка 60%, белый текст
- CHECKBOX: рамка + крестик внутри
- RADIO: рамка + кружок внутри
- LABEL: тонкая голубая рамка
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Цвета (BGR)
COLOR_HEADER = (255, 100, 0)      # Ярко-синий
COLOR_INPUT = (0, 200, 0)         # Ярко-зелёный
COLOR_TEXTAREA = (0, 140, 255)    # Ярко-оранжевый
COLOR_ACTION = (0, 0, 255)        # Ярко-красный
COLOR_CHECKBOX = (200, 0, 200)    # Пурпурный
COLOR_RADIO = (200, 0, 200)       # Пурпурный
COLOR_LABEL = (255, 200, 100)     # Голубой светлый
COLOR_SECTION = (150, 150, 150)   # Серый
COLOR_UNKNOWN = (100, 100, 100)   # Тёмно-серый

# Заливка (прозрачность)
FILL_ALPHA_INPUT = 0.4
FILL_ALPHA_TEXTAREA = 0.4
FILL_ALPHA_ACTION = 0.6
FILL_ALPHA_CHECKBOX = 0.3

# Толщина линий
THICKNESS_HEADER = 4
THICKNESS_INPUT = 3
THICKNESS_TEXTAREA = 2
THICKNESS_ACTION = 3
THICKNESS_CHECKBOX = 2
THICKNESS_DEFAULT = 2


def get_color_and_style(element_type: str, row_type: str = "") -> Tuple[Tuple[int, int, int], int, float, str]:
    """
    Возвращает (color, thickness, fill_alpha, style) для типа элемента.
    style: 'solid', 'dashed', 'dotted'
    """
    etype = element_type.lower() if element_type else ""
    rtype = row_type.upper() if row_type else ""
    
    # Приоритет по element_type
    if etype in ("checkbox", "radio"):
        return COLOR_CHECKBOX, THICKNESS_CHECKBOX, FILL_ALPHA_CHECKBOX, "solid"
    
    if etype == "button" or rtype == "ACTION":
        return COLOR_ACTION, THICKNESS_ACTION, FILL_ALPHA_ACTION, "solid"
    
    if etype == "textarea" or rtype == "TEXTAREA":
        return COLOR_TEXTAREA, THICKNESS_TEXTAREA, FILL_ALPHA_TEXTAREA, "dashed"
    
    if etype in ("input", "field") or rtype.startswith("FIELD"):
        return COLOR_INPUT, THICKNESS_INPUT, FILL_ALPHA_INPUT, "solid"
    
    if rtype == "HEADER" or etype == "header":
        return COLOR_HEADER, THICKNESS_HEADER, 0.0, "solid"
    
    if etype == "label" or rtype == "TEXT":
        return COLOR_LABEL, 1, 0.0, "solid"
    
    if etype == "section":
        return COLOR_SECTION, 2, 0.1, "solid"
    
    return COLOR_UNKNOWN, THICKNESS_DEFAULT, 0.0, "solid"


def draw_filled_rect(img, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], alpha: float):
    """Рисует прямоугольник с заливкой и прозрачностью."""
    import cv2
    if alpha <= 0:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_dashed_rect(img, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int, dash_length: int = 10):
    """Рисует пунктирный прямоугольник."""
    import cv2
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Горизонтальные линии
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    
    # Вертикальные линии
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


def draw_checkbox_marker(img, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int]):
    """Рисует крестик внутри bbox (маркер checkbox)."""
    import cv2
    x1, y1 = pt1
    x2, y2 = pt2
    pad = 3
    cv2.line(img, (x1 + pad, y1 + pad), (x2 - pad, y2 - pad), color, 2)
    cv2.line(img, (x1 + pad, y2 - pad), (x2 - pad, y1 + pad), color, 2)


def draw_radio_marker(img, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int]):
    """Рисует кружок внутри bbox (маркер radio)."""
    import cv2
    cx = (pt1[0] + pt2[0]) // 2
    cy = (pt1[1] + pt2[1]) // 2
    radius = min(pt2[0] - pt1[0], pt2[1] - pt1[1]) // 3
    cv2.circle(img, (cx, cy), radius, color, 2)


def draw_element(
    img,
    bbox: List[float],
    element_type: str,
    row_type: str = "",
    confidence: float = 0.0,
    source: str = "",
    label: str = "",
):
    """
    Рисует элемент с полной визуализацией.
    """
    import cv2
    
    if len(bbox) < 4:
        return
    
    pt1 = (int(bbox[0]), int(bbox[1]))
    pt2 = (int(bbox[2]), int(bbox[3]))
    
    color, thickness, fill_alpha, style = get_color_and_style(element_type, row_type)
    
    # Заливка
    if fill_alpha > 0:
        draw_filled_rect(img, pt1, pt2, color, fill_alpha)
    
    # Рамка
    if style == "dashed":
        draw_dashed_rect(img, pt1, pt2, color, thickness)
    else:
        cv2.rectangle(img, pt1, pt2, color, thickness)
    
    # Специальные маркеры
    etype = element_type.lower() if element_type else ""
    if etype == "checkbox":
        draw_checkbox_marker(img, pt1, pt2, color)
    elif etype == "radio":
        draw_radio_marker(img, pt1, pt2, color)
    
    # Подпись
    text_parts = []
    if element_type:
        text_parts.append(element_type)
    if row_type and row_type != element_type:
        text_parts.append(f"[{row_type}]")
    if confidence > 0:
        text_parts.append(f"{confidence:.2f}")
    if source:
        text_parts.append(f"({source[:6]})")
    if label:
        text_parts.append(label)
    
    text = " ".join(text_parts)
    if text:
        # Белый текст с чёрной обводкой
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_y = max(pt1[1] - 5, 15)
        
        # Тень/обводка
        cv2.putText(img, text, (pt1[0] + 1, text_y + 1), font, font_scale, (0, 0, 0), font_thickness + 1)
        # Основной текст
        cv2.putText(img, text, (pt1[0], text_y), font, font_scale, (255, 255, 255), font_thickness)


def visualize_elements_enhanced(
    image_path: str,
    elements: List[Dict[str, Any]],
    output_path: str,
    title: str = "",
) -> None:
    """
    Создаёт усиленную визуализацию всех элементов.
    
    Args:
        image_path: путь к исходному изображению
        elements: список элементов с полями:
            - bbox: [x1, y1, x2, y2]
            - element_type: тип элемента
            - row_type: тип строки (опционально)
            - confidence: уверенность (опционально)
            - source: источник детекции (опционально)
        output_path: путь для сохранения
        title: заголовок (опционально)
    """
    import cv2
    
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return
    
    out = img.copy()
    
    # Сортируем по площади (большие сначала, чтобы маленькие были сверху)
    sorted_elements = sorted(
        elements,
        key=lambda e: -(e.get("bbox", [0, 0, 0, 0])[2] - e.get("bbox", [0, 0, 0, 0])[0]) *
                       (e.get("bbox", [0, 0, 0, 0])[3] - e.get("bbox", [0, 0, 0, 0])[1])
    )
    
    for e in sorted_elements:
        draw_element(
            out,
            e.get("bbox", []),
            e.get("element_type", ""),
            e.get("row_type", ""),
            e.get("confidence", 0.0),
            e.get("source", ""),
        )
    
    # Заголовок
    if title:
        cv2.putText(out, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Легенда
    legend_y = 50
    legend_items = [
        ("HEADER", COLOR_HEADER),
        ("INPUT", COLOR_INPUT),
        ("TEXTAREA", COLOR_TEXTAREA),
        ("ACTION", COLOR_ACTION),
        ("CHECKBOX", COLOR_CHECKBOX),
        ("LABEL", COLOR_LABEL),
    ]
    for name, color in legend_items:
        cv2.rectangle(out, (10, legend_y), (25, legend_y + 12), color, -1)
        cv2.putText(out, name, (30, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        legend_y += 18
    
    cv2.imwrite(output_path, out)
    logger.debug(f"Enhanced visualization saved: {output_path}")


def visualize_nesting(
    image_path: str,
    elements: List[Dict[str, Any]],
    nested_pairs: List[Tuple[int, int]],
    output_path: str,
) -> None:
    """
    Визуализирует вложенность элементов стрелками.
    """
    import cv2
    
    img = cv2.imread(image_path)
    if img is None:
        return
    
    out = img.copy()
    
    # Рисуем все элементы бледно
    for e in elements:
        bbox = e.get("bbox", [])
        if len(bbox) >= 4:
            cv2.rectangle(out, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (200, 200, 200), 1)
    
    # Рисуем вложенные пары яркими стрелками
    for parent_idx, child_idx in nested_pairs:
        if parent_idx >= len(elements) or child_idx >= len(elements):
            continue
        
        parent = elements[parent_idx].get("bbox", [])
        child = elements[child_idx].get("bbox", [])
        
        if len(parent) < 4 or len(child) < 4:
            continue
        
        # Parent — красная рамка
        cv2.rectangle(out, (int(parent[0]), int(parent[1])), (int(parent[2]), int(parent[3])), (0, 0, 255), 3)
        
        # Child — зелёная рамка
        cv2.rectangle(out, (int(child[0]), int(child[1])), (int(child[2]), int(child[3])), (0, 255, 0), 2)
        
        # Стрелка от центра parent к центру child
        p_center = (int((parent[0] + parent[2]) / 2), int((parent[1] + parent[3]) / 2))
        c_center = (int((child[0] + child[2]) / 2), int((child[1] + child[3]) / 2))
        cv2.arrowedLine(out, p_center, c_center, (255, 0, 255), 2, tipLength=0.3)
    
    cv2.imwrite(output_path, out)
    logger.debug(f"Nesting visualization saved: {output_path}")


def visualize_overlaps(
    image_path: str,
    elements: List[Dict[str, Any]],
    overlapping_pairs: List[Tuple[int, int, float]],
    output_path: str,
) -> None:
    """
    Визуализирует пересекающиеся элементы.
    """
    import cv2
    
    img = cv2.imread(image_path)
    if img is None:
        return
    
    out = img.copy()
    
    for idx1, idx2, iou in overlapping_pairs:
        if idx1 >= len(elements) or idx2 >= len(elements):
            continue
        
        e1 = elements[idx1].get("bbox", [])
        e2 = elements[idx2].get("bbox", [])
        
        if len(e1) < 4 or len(e2) < 4:
            continue
        
        # Оба элемента разными цветами
        cv2.rectangle(out, (int(e1[0]), int(e1[1])), (int(e1[2]), int(e1[3])), (0, 0, 255), 2)
        cv2.rectangle(out, (int(e2[0]), int(e2[1])), (int(e2[2]), int(e2[3])), (0, 255, 0), 2)
        
        # Область пересечения
        x1 = max(e1[0], e2[0])
        y1 = max(e1[1], e2[1])
        x2 = min(e1[2], e2[2])
        y2 = min(e1[3], e2[3])
        
        if x2 > x1 and y2 > y1:
            overlay = out.copy()
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
            
            # Текст с IoU
            cv2.putText(out, f"IoU={iou:.2f}", (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(out, f"IoU={iou:.2f}", (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    cv2.imwrite(output_path, out)
    logger.debug(f"Overlaps visualization saved: {output_path}")
