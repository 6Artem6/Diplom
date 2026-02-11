"""
Уровень 0 — Глобальная ориентация страницы.

Определяет вертикальный ритм, основные цветовые фоны, повторяющиеся контейнеры.
Сегментация по цвету, плотности контента, отступам. Detectron2 не используется.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.models import PageSegment

logger = logging.getLogger(__name__)

# Размер сетки для анализа (пиксели на ячейку)
GRID_CELL_H = 32
GRID_CELL_W = 64
# Порог плотности краёв для "контент" vs "пустота"
CONTENT_DENSITY_THRESHOLD = 0.08
# Минимальная высота сегмента (объединение полос)
MIN_SEGMENT_HEIGHT_PX = 24
# K-means: число доминантных цветов фона
N_DOMINANT_COLORS = 4


def _ensure_cv2_numpy():
    import cv2
    import numpy as np
    return cv2, np


def build_page_orientation_context(
    image_path: str,
) -> Tuple[List[PageSegment], Dict[str, Any]]:
    """
    Строит контекст ориентации страницы по изображению.
    Возвращает список сегментов (крупные области по цвету/плотности) и диагностику.
    """
    cv2, np = _ensure_cv2_numpy()
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("page_orientation_context: image not read %s", image_path)
        return [], {"error": "image_not_read"}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Вертикальный ритм: плотность контента по горизонтальным полосам
    strip_h = max(8, h // 40)
    density_per_strip: List[float] = []
    for y0 in range(0, h, strip_h):
        y1 = min(y0 + strip_h, h)
        strip = gray[y0:y1, :]
        edges = cv2.Canny(strip, 50, 150)
        density_per_strip.append(float(edges.sum()) / max(1, strip.size))

    # Нормализация плотности к 0..1
    if density_per_strip:
        d_max = max(density_per_strip)
        d_min = min(density_per_strip)
        if d_max > d_min:
            density_norm = [(d - d_min) / (d_max - d_min) for d in density_per_strip]
        else:
            density_norm = [0.5] * len(density_per_strip)
    else:
        density_norm = []

    # Доминантный цвет фона: по нижней части (часто фон) и по сетке
    sample_points: List[Tuple[int, int, int, int]] = []
    for by in range(0, h, GRID_CELL_H):
        for bx in range(0, w, GRID_CELL_W):
            by2 = min(by + GRID_CELL_H, h)
            bx2 = min(bx + GRID_CELL_W, w)
            if by2 > by and bx2 > bx:
                sample_points.append((bx, by, bx2, by2))
    colors_bgr: List[Tuple[int, int, int]] = []
    for (x1, y1, x2, y2) in sample_points[:200]:
        cell = img[y1:y2, x1:x2]
        if cell.size > 0:
            m = cell.reshape(-1, 3).mean(axis=0)
            colors_bgr.append((int(m[0]), int(m[1]), int(m[2])))

    if not colors_bgr:
        dominant_bgr = (255, 255, 255)
    else:
        # Упрощённый "доминантный": медиана по каналам
        import statistics
        dominant_bgr = (
            int(statistics.median([c[0] for c in colors_bgr])),
            int(statistics.median([c[1] for c in colors_bgr])),
            int(statistics.median([c[2] for c in colors_bgr])),
        )

    # Сегменты: объединяем полосы с похожей плотностью в крупные области
    segments: List[PageSegment] = []
    i = 0
    while i < len(density_norm):
        d = density_norm[i]
        y_start = i * strip_h
        y_end = min((i + 1) * strip_h, h)
        is_gap = d < CONTENT_DENSITY_THRESHOLD
        j = i + 1
        while j < len(density_norm) and (density_norm[j] < CONTENT_DENSITY_THRESHOLD) == is_gap:
            y_end = min((j + 1) * strip_h, h)
            j += 1
        seg_h = y_end - y_start
        if seg_h >= MIN_SEGMENT_HEIGHT_PX or not segments:
            seg_density = sum(density_norm[i:j]) / max(1, j - i) if j > i else d
            segments.append(PageSegment(
                y_min=float(y_start),
                y_max=float(y_end),
                x_min=0.0,
                x_max=float(w),
                dominant_bgr=dominant_bgr,
                content_density=seg_density,
                is_likely_gap=is_gap,
            ))
        i = j

    diagnostics: Dict[str, Any] = {
        "image_h": h,
        "image_w": w,
        "strip_h": strip_h,
        "n_strips": len(density_per_strip),
        "dominant_bgr": list(dominant_bgr),
        "n_segments": len(segments),
    }
    return segments, diagnostics


def visualize_page_orientation(
    image_path: str,
    segments: List[PageSegment],
    output_path: str,
) -> None:
    """Сохраняет визуализацию сегментов страницы (уровень 0)."""
    cv2, np = _ensure_cv2_numpy()
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    from src.infrastructure.debug_draw import putText_visible, rectangle_visible

    for seg in segments:
        y1, y2 = int(seg.y_min), int(seg.y_max)
        color = (0, 180, 0) if seg.is_likely_gap else (0, 0, 180)
        rectangle_visible(out, (0, y1), (out.shape[1], y2), color, 1)
        putText_visible(
            out, "gap" if seg.is_likely_gap else "content", (5, y1 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), (0, 0, 0), 1,
        )
    cv2.imwrite(output_path, out)
    logger.debug("page_orientation_context: saved %s", output_path)
