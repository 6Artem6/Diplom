"""
Геометрия из изображения: полосы строк и колонки из визуальных кандидатов и OCR.

Источник истины — CV и OCR; фиксированные ratio только fallback.
- Row bands: кластеризация Y (visual_candidates + OCR baseline) → полосы по рядам.
- Column boundaries: кластеризация X центров визуальных кандидатов → вертикальные оси.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.form_schema_models import FormSchema

logger = logging.getLogger(__name__)

# Кластеризация: max расстояние по Y для одного ряда (px)
ROW_CLUSTER_TOLERANCE_PX = 20
# Минимум якорей для использования (иначе fallback)
MIN_ANCHORS_FOR_ROW_BANDS = 1
# Доля высоты блока на полосу, если используем только центр кластера
ROW_BAND_HALF_RATIO = 0.04


@dataclass
class RowBand:
    y_min_ratio: float
    y_max_ratio: float


def _collect_y_anchors(
    visual_candidates: List[List[float]],
    ocr_inside: List[Dict[str, Any]],
    card_bbox: List[float],
) -> List[float]:
    """Y-координаты: центры визуальных кандидатов и baseline OCR (низ строки текста)."""
    anchors: List[float] = []
    card_y1, card_y2 = card_bbox[1], card_bbox[3]
    for b in visual_candidates:
        if len(b) < 4:
            continue
        cy = (b[1] + b[3]) / 2
        if card_y1 <= cy <= card_y2:
            anchors.append(cy)
    for ob in ocr_inside:
        b = ob.get("bbox", [0, 0, 0, 0])
        if len(b) < 4:
            continue
        baseline_y = b[3]
        if card_y1 <= baseline_y <= card_y2:
            anchors.append(baseline_y)
    return anchors


def _cluster_y(anchors: List[float], tolerance_px: float) -> List[float]:
    """Кластеризация по Y; возвращаем отсортированный список Y-центров кластеров."""
    if not anchors:
        return []
    sorted_y = sorted(anchors)
    clusters: List[List[float]] = []
    for y in sorted_y:
        placed = False
        for c in clusters:
            if abs(y - statistics.median(c)) <= tolerance_px:
                c.append(y)
                placed = True
                break
        if not placed:
            clusters.append([y])
    return [statistics.median(c) for c in clusters]


def row_bands_from_anchors(
    card_bbox: List[float],
    schema: FormSchema,
    visual_candidates: List[List[float]],
    ocr_inside: List[Dict[str, Any]],
    fallback_fixed_ratios: bool = True,
) -> List[RowBand]:
    """
    Полосы строк, привязанные к изображению: Y из визуальных кандидатов и OCR baseline.
    Если якорей мало — fallback на фиксированные ratio (если fallback_fixed_ratios=True).
    """
    card_h = card_bbox[3] - card_bbox[1]
    card_y1 = card_bbox[1]
    n = len(schema.rows)
    field_row_indices = [
        i for i in range(n)
        if schema.rows[i].role == "field_row"
        and (schema.flow_end_row_index is None or i <= schema.flow_end_row_index)
    ]
    if not field_row_indices:
        return [RowBand(0.0, 0.0) for _ in range(n)]

    anchors = _collect_y_anchors(visual_candidates, ocr_inside, card_bbox)
    row_centers = _cluster_y(anchors, ROW_CLUSTER_TOLERANCE_PX) if anchors else []
    row_centers.sort()
    if len(row_centers) >= MIN_ANCHORS_FOR_ROW_BANDS:
        band_half = max(card_h * ROW_BAND_HALF_RATIO, 8.0)
        row_bands_by_field: List[RowBand] = []
        for idx in range(len(field_row_indices)):
            if idx < len(row_centers):
                yc = row_centers[idx]
                y_min_ratio = (yc - band_half - card_y1) / card_h if card_h > 0 else 0.0
                y_max_ratio = (yc + band_half - card_y1) / card_h if card_h > 0 else 1.0
                y_min_ratio = max(0.0, min(1.0, y_min_ratio))
                y_max_ratio = max(0.0, min(1.0, y_max_ratio))
                if y_max_ratio <= y_min_ratio:
                    y_max_ratio = y_min_ratio + 0.05
                row_bands_by_field.append(RowBand(y_min_ratio=y_min_ratio, y_max_ratio=y_max_ratio))
            else:
                y_cur = row_bands_by_field[-1].y_max_ratio + 0.02 if row_bands_by_field else 0.0
                row_bands_by_field.append(RowBand(y_min_ratio=y_cur, y_max_ratio=min(1.0, y_cur + 0.065)))
        result: List[RowBand] = []
        fi = 0
        for i in range(n):
            if schema.rows[i].role != "field_row" or (schema.flow_end_row_index is not None and i > schema.flow_end_row_index):
                result.append(RowBand(0.0, 0.0))
            elif fi < len(row_bands_by_field):
                result.append(row_bands_by_field[fi])
                fi += 1
            else:
                result.append(RowBand(0.0, 0.0))
        while len(result) < n:
            result.append(RowBand(0.0, 0.0))
        return result

    if not fallback_fixed_ratios:
        return [RowBand(0.0, 0.0) for _ in range(n)]
    y_cur = 0.0
    result = []
    for i in range(n):
        row = schema.rows[i]
        if row.role != "field_row" or (schema.flow_end_row_index is not None and i > schema.flow_end_row_index):
            result.append(RowBand(0.0, 0.0))
            continue
        is_textarea = any(s.expected_input_type == "textarea" for s in row.slots)
        h_ratio = 0.12 if is_textarea else 0.065
        y_max = min(1.0, y_cur + h_ratio)
        result.append(RowBand(y_min_ratio=y_cur, y_max_ratio=y_max))
        y_cur = y_max + 0.02
    while len(result) < n:
        result.append(RowBand(0.0, 0.0))
    return result


def column_boundaries_from_visual(
    visual_candidates: List[List[float]],
    card_bbox: List[float],
    tolerance_ratio: float = 0.15,
) -> List[Tuple[float, float]]:
    """
    Границы колонок из кластеризации x_center визуальных кандидатов.
    Возвращает [(x1, x2), ...] в координатах изображения.
    """
    if not visual_candidates or len(card_bbox) < 4:
        return []
    card_w = card_bbox[2] - card_bbox[0]
    card_x1 = card_bbox[0]
    x_centers = [(b[0] + b[2]) / 2 for b in visual_candidates if len(b) >= 4 and card_bbox[0] <= (b[0]+b[2])/2 <= card_bbox[2]]
    if not x_centers:
        return []
    tol = max(20, card_w * tolerance_ratio)
    sorted_x = sorted(x_centers)
    clusters: List[List[float]] = []
    for x in sorted_x:
        placed = False
        for c in clusters:
            if abs(x - statistics.median(c)) <= tol:
                c.append(x)
                placed = True
                break
        if not placed:
            clusters.append([x])
    boundaries: List[Tuple[float, float]] = []
    for c in clusters:
        cx = statistics.median(c)
        half = (card_w / max(len(clusters), 1)) / 2
        x1 = max(card_x1, cx - half)
        x2 = min(card_bbox[2], cx + half)
        if x2 > x1:
            boundaries.append((x1, x2))
    return sorted(boundaries, key=lambda b: b[0])
