"""
Уровень 2 — Скелет формы (вершина графа).

Работает только внутри form_area. Строит rows, columns, card groups.
Использует визуальные границы, фоновый цвет, интервалы между строками, выравнивание по X.
OCR только вспомогательный сигнал.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.models import FormRow, FormSkeleton, PageSegment, Region

logger = logging.getLogger(__name__)

ROW_STRIP_HEIGHT = 24
MIN_ROW_HEIGHT = 16
MAX_ROW_GAP = 40
# Кластеризация X-центров для определения колонок
X_CLUSTER_TOLERANCE_RATIO = 0.12


def build_form_skeleton(
    image_path: str,
    form_region: Region,
    segments: List[PageSegment],
    ocr_boxes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[FormSkeleton], Dict[str, Any]]:
    """
    Строит скелет формы внутри form_region: строки и при необходимости колонки.
    """
    import cv2
    import statistics

    img = cv2.imread(str(image_path))
    if img is None:
        return None, {"error": "image_not_read"}

    bbox = form_region.bbox
    if len(bbox) < 4:
        return None, {"error": "invalid_form_region"}
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, {"error": "empty_crop"}

    h_crop, w_crop = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Горизонтальные линии / ритм: проекция краёв по Y
    row_edges = edges.sum(axis=1)
    # Сглаживание
    kernel = min(5, max(3, h_crop // 30))
    if kernel % 2 == 0:
        kernel += 1
    try:
        import numpy as np
        row_edges_smooth = np.convolve(row_edges, np.ones(kernel) / kernel, mode="same")
    except Exception:
        row_edges_smooth = row_edges

    # Порог: полосы с контентом vs промежутки
    try:
        threshold = float(max(row_edges_smooth)) * 0.15
    except (TypeError, ValueError):
        threshold = 0.0
    n_smooth = len(row_edges_smooth) if hasattr(row_edges_smooth, "__len__") else 0
    if n_smooth == 0:
        n_smooth = len(row_edges)

    # Найти границы строк (переходы низкая активность → высокая и обратно)
    in_row = False
    row_starts: List[int] = []
    row_ends: List[int] = []
    for i in range(1, n_smooth):
        val = float(row_edges_smooth[i]) if i < len(row_edges_smooth) else 0
        if not in_row and val > threshold:
            in_row = True
            row_starts.append(i)
        elif in_row and val <= threshold:
            in_row = False
            row_ends.append(i)

    if in_row:
        row_ends.append(n_smooth)

    # Объединить в пары start–end, отфильтровать по минимальной высоте
    row_ranges: List[Tuple[int, int]] = []
    for i in range(min(len(row_starts), len(row_ends))):
        a, b = row_starts[i], row_ends[i]
        if b - a >= MIN_ROW_HEIGHT:
            row_ranges.append((a, b))

    # Если по краям ничего не нашли — разбить form_area на равные полосы по умолчанию
    if not row_ranges:
        n_default = max(2, h_crop // 50)
        step = h_crop / n_default
        for i in range(n_default):
            row_ranges.append((int(i * step), int((i + 1) * step)))

    rows: List[FormRow] = []
    for idx, (ry0, ry1) in enumerate(row_ranges):
        rows.append(FormRow(
            row_index=idx,
            y_min=float(y1 + ry0),
            y_max=float(y1 + ry1),
            x_min=float(x1),
            x_max=float(x2),
            column_count=1,
        ))

    # Колонки: по X-центрам контуров в crop (или по OCR)
    x_centers: List[float] = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 200:
            continue
        rx, _, rw, _ = cv2.boundingRect(c)
        x_centers.append(x1 + rx + rw / 2.0)
    if ocr_boxes:
        for ob in ocr_boxes:
            b = ob.get("bbox", [])
            if len(b) >= 4 and b[1] >= y1 and b[3] <= y2 and b[0] >= x1 and b[2] <= x2:
                x_centers.append((b[0] + b[2]) / 2)

    column_boundaries: Optional[List[Tuple[float, float]]] = None
    if x_centers:
        x_centers.sort()
        tol = w_crop * X_CLUSTER_TOLERANCE_RATIO
        clusters: List[List[float]] = []
        for x in x_centers:
            placed = False
            for cl in clusters:
                if abs(x - statistics.median(cl)) <= tol:
                    cl.append(x)
                    placed = True
                    break
            if not placed:
                clusters.append([x])
        if len(clusters) >= 2:
            column_boundaries = []
            for cl in clusters:
                cx = statistics.median(cl)
                half = w_crop / max(len(clusters), 1) / 2
                column_boundaries.append((x1 + max(0, cx - half), x1 + min(w_crop, cx + half)))
            column_boundaries.sort(key=lambda p: p[0])
            for row in rows:
                row.column_count = len(column_boundaries)

    layout_type: str = "vertical"
    if column_boundaries and len(column_boundaries) > 1:
        layout_type = "grid" if all(r.column_count > 1 for r in rows) else "mixed"

    skeleton = FormSkeleton(
        form_region=form_region,
        rows=rows,
        column_boundaries=column_boundaries,
        layout_type=layout_type,
    )
    diag = {"n_rows": len(rows), "layout_type": layout_type, "n_columns": len(column_boundaries) if column_boundaries else 0}
    return skeleton, diag


def visualize_form_skeleton(
    image_path: str,
    skeleton: FormSkeleton,
    output_path: str,
) -> None:
    """Визуализация скелета формы: строки и колонки (уровень 2)."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    from src.infrastructure.debug_draw import line_visible, putText_visible, rectangle_visible

    out = img.copy()
    for row in skeleton.rows:
        y1, y2 = int(row.y_min), int(row.y_max)
        rectangle_visible(out, (int(row.x_min), y1), (int(row.x_max), y2), (0, 180, 180), 1)
        putText_visible(
            out, "R%d" % row.row_index, (int(row.x_min) + 2, y1 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), (0, 0, 0), 1,
        )
    if skeleton.column_boundaries:
        for (cx1, cx2) in skeleton.column_boundaries:
            line_visible(
                out,
                (int(cx1), int(skeleton.rows[0].y_min)),
                (int(cx1), int(skeleton.rows[-1].y_max)),
                (0, 140, 200), 1,
            )
    cv2.imwrite(output_path, out)
    logger.debug("form_skeleton_builder: saved %s", output_path)
