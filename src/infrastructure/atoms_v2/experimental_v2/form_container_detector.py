"""
Уровень 0 (ТЗ) — FormContainerDetector.

Находит замкнутый контейнер формы (card/panel) по геометрии:
замкнутый прямоугольник, светлый фон на тёмном, border/shadow, разумный aspect, визуальный центр.
Не использует OCR для bbox формы. Не «остаток между header и footer».
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.models import FormContainer

logger = logging.getLogger(__name__)

# Главный инвариант контейнера: относительные размеры (не зависят от цвета фона)
MIN_CONTAINER_WIDTH_RATIO = 0.35   # width >= 0.35 * page_width
MIN_CONTAINER_HEIGHT_RATIO = 0.25  # height >= 0.25 * page_height
MIN_CONTAINER_AREA_RATIO = 0.10    # площадь >= 0.1 * page_area
# Отсечение согласовано с минимумами: не контейнер если уже не прошёл min (чтобы не отбрасывать 0.35–0.4 ширины)
REJECT_WIDTH_RATIO = 0.35
REJECT_HEIGHT_RATIO = 0.25
# Абсолютные минимумы (fallback при отсутствии размеров страницы)
MIN_CONTAINER_WIDTH = 320
MIN_CONTAINER_HEIGHT = 120
MIN_CONTAINER_AREA = 8000
MAX_ASPECT = 2.5
MIN_ASPECT = 0.6
MAX_CONTAINER_AREA_RATIO = 0.90
MIN_MARGIN_PX = 10
# Кнопка никогда не контейнер
BUTTON_CONTAINER_HEIGHT_MIN = 28
BUTTON_CONTAINER_HEIGHT_MAX = 64
BUTTON_CONTAINER_WIDTH_MIN = 60
BUTTON_CONTAINER_ASPECT_MIN = 1.8
# Порог «светлее фона»: разница средних (inside - outside)
LIGHT_ON_DARK_MIN_DIFF = 15
# Минимальный confidence для кандидата
MIN_CONFIDENCE = 0.3


def _score_container(
    img_global: Any,
    bbox: List[float],
    page_mean: float,
) -> float:
    """Скор кандидата: светлый фон на тёмном, близость к центру страницы."""
    import cv2
    h, w = img_global.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = img_global[y1:y2, x1:x2]
    inside_mean = float(cv2.mean(crop)[0])
    # Среднее по рамке вокруг bbox (внешность)
    pad = max(5, min(w, h) // 20)
    band_top = img_global[max(0, y1 - pad):y1, x1:x2] if y1 - pad < y1 else None
    band_bot = img_global[y2:min(h, y2 + pad), x1:x2] if y2 + pad <= h else None
    band_left = img_global[y1:y2, max(0, x1 - pad):x1] if x1 - pad < x1 else None
    band_right = img_global[y1:y2, x2:min(w, x2 + pad)] if x2 + pad <= w else None
    outer_vals = []
    for band in (band_top, band_bot, band_left, band_right):
        if band is not None and band.size > 0:
            outer_vals.append(float(cv2.mean(band)[0]))
    outside_mean = sum(outer_vals) / len(outer_vals) if outer_vals else page_mean
    light_on_dark = inside_mean - outside_mean
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    page_cx = w / 2
    page_cy = h / 2
    dist = ((center_x - page_cx) ** 2 + (center_y - page_cy) ** 2) ** 0.5
    max_dist = ((w / 2) ** 2 + (h / 2) ** 2) ** 0.5
    center_score = 1.0 - min(1.0, dist / max(1, max_dist)) * 0.5
    contrast_score = min(1.0, max(0.0, light_on_dark / 80.0)) if light_on_dark > 0 else 0.0
    return 0.4 * center_score + 0.6 * contrast_score


def detect_form_containers(
    image_path: str,
    detectron_regions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[FormContainer], Dict[str, Any]]:
    """
    Находит кандидатов контейнера формы по визуальным примитивам (edges, contours).
    Возвращает список FormContainer, отсортированный по confidence (лучший первым).
    """
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    if img is None:
        return [], {"error": "image_not_read"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    page_area = h * w
    page_mean = float(gray.mean())

    candidates: List[Tuple[float, List[float]]] = []

    # 1) Контуры из бинаризации: замкнутые прямоугольники
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTAINER_AREA or area > page_area * MAX_CONTAINER_AREA_RATIO:
            continue
        x, y, rw, rh = cv2.boundingRect(c)
        x2, y2 = x + rw, y + rh
        if x < MIN_MARGIN_PX or y < MIN_MARGIN_PX or w - x2 < MIN_MARGIN_PX or h - y2 < MIN_MARGIN_PX:
            continue
        if rh <= 0:
            continue
        aspect = rw / rh
        if aspect > MAX_ASPECT or aspect < MIN_ASPECT:
            continue
        bbox = [float(x), float(y), float(x2), float(y2)]
        sc = _score_container(gray, bbox, page_mean)
        if sc >= MIN_CONFIDENCE:
            candidates.append((sc, bbox))

    # 2) Контуры из Canny (границы карточки)
    edges = cv2.Canny(blurred, 50, 150)
    contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours2:
        area = cv2.contourArea(c)
        if area < MIN_CONTAINER_AREA or area > page_area * MAX_CONTAINER_AREA_RATIO:
            continue
        x, y, rw, rh = cv2.boundingRect(c)
        x2, y2 = x + rw, y + rh
        if rw < 50 or rh < 50:
            continue
        if x < MIN_MARGIN_PX or y < MIN_MARGIN_PX or w - x2 < MIN_MARGIN_PX or h - y2 < MIN_MARGIN_PX:
            continue
        aspect = rw / rh if rh else 0
        if aspect > MAX_ASPECT or aspect < MIN_ASPECT:
            continue
        bbox = [float(x), float(y), float(x2), float(y2)]
        sc = _score_container(gray, bbox, page_mean)
        if sc >= MIN_CONFIDENCE:
            candidates.append((sc, bbox))

    # 3) Опционально: регионы от Detectron (card/panel)
    if detectron_regions:
        for r in detectron_regions:
            b = r.get("bbox", [])
            if len(b) < 4:
                continue
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area < MIN_CONTAINER_AREA or area > page_area * MAX_CONTAINER_AREA_RATIO:
                continue
            sc = _score_container(gray, b, page_mean) * 0.9
            if sc >= MIN_CONFIDENCE:
                candidates.append((sc, list(b)))

    # Дедуп по IoU
    def _iou(a: List[float], b: List[float]) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / max(1e-9, u)

    candidates.sort(key=lambda x: -x[0])
    unique: List[Tuple[float, List[float]]] = []
    for sc, bbox in candidates:
        if any(_iou(bbox, u[1]) >= 0.7 for u in unique):
            continue
        unique.append((sc, bbox))

    # Как в исходной версии (04cd37d): контейнеры из unique без жёсткой геометрии.
    # Отсев кнопок и мелких bbox делается в validate_container_with_visual (≥3 строк, ≥2 полей).
    containers = [FormContainer(bbox=b, confidence=float(sc), metadata={"source": "form_container_detector"}) for sc, b in unique]
    diag = {"candidates": len(candidates), "after_dedup": len(unique), "after_geometry": len(containers), "image_h": h, "image_w": w}
    return containers, diag


def get_best_container(
    containers: List[FormContainer],
    demo_mode: bool = False,
) -> Optional[FormContainer]:
    """Возвращает один лучший контейнер. В demo_mode — по площади (самый большой bbox)."""
    if not containers:
        return None
    if demo_mode:
        return max(
            containers,
            key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]) if len(c.bbox) >= 4 else 0,
        )
    return max(containers, key=lambda c: c.confidence)


def visualize_container(
    image_path: str,
    container: FormContainer,
    output_path: str,
) -> None:
    """Сохраняет container_bbox.png."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    from src.infrastructure.debug_draw import putText_visible, rectangle_visible

    out = img.copy()
    b = container.bbox
    if len(b) >= 4:
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        color_bgr = (0, 180, 0)
        rectangle_visible(out, (x1, y1), (x2, y2), color_bgr, 2)
        putText_visible(
            out, "FormContainer %.2f" % container.confidence, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), (0, 0, 0), 1,
        )
    cv2.imwrite(output_path, out)
    logger.debug("form_container_detector: saved %s", output_path)
