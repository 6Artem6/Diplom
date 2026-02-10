"""
Уровень 1 (ТЗ) — FormInnerLayout: строки и колонки строго внутри FormContainer.bbox.

RowDetector: типы строк HEADER/FIELD/TEXTAREA/ACTION/SPACER; OCR как сигнал (текст сверху + поле снизу = одна строка);
height_mode fixed/adaptive. ColumnDetector только при layout_type != vertical.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.models import (
    FormColumn,
    FormContainer,
    FormRow,
    FormSkeleton,
    RowType,
    HeightMode,
)

logger = logging.getLogger(__name__)

MIN_ROW_HEIGHT = 14
TEXTAREA_HEIGHT_MEDIAN_K = 1.8  # row height > median * k → TEXTAREA
ACTION_WORDS = frozenset({
    "save", "submit", "search", "send", "add", "ok", "login", "cancel", "apply",
    "отправить", "сохранить", "войти", "далее",
})
X_CLUSTER_TOLERANCE_RATIO = 0.12
# Порог: variance X-центров нормализованная < этого → vertical
LAYOUT_VERTICAL_X_VARIANCE_MAX = 0.04


def _ocr_heights(ocr_inside: List[Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    for ob in ocr_inside:
        b = ob.get("bbox", [])
        if len(b) >= 4:
            out.append(b[3] - b[1])
    return out


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[len(s) // 2])


def _build_rows_inside_container(
    image_path: str,
    container: FormContainer,
    ocr_inside: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[FormRow], List[List[float]], List[int], float]:
    """
    Строки на основе OCR: текст сверху + поле снизу = одна логическая строка.
    Возвращает (rows, skipped_bboxes для header, textarea_row_indices, form_start_y).
    """
    import cv2
    import numpy as np

    ocr_inside = ocr_inside or []
    bbox = container.bbox
    if len(bbox) < 4:
        return [], [], [], bbox[1]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    form_start_y = y1

    # Высота OCR как proxy font_size
    ocr_heights = _ocr_heights(ocr_inside)
    median_font = _median(ocr_heights) if ocr_heights else 20.0

    # Header формы: OCR с font_size > median выше первой "input"-подобной зоны
    # Собираем Y-позиции OCR по вертикали
    ocr_sorted: List[Tuple[float, float, float, Dict]] = []  # (y_center, y_bottom, height, ob)
    for ob in ocr_inside:
        b = ob.get("bbox", [])
        if len(b) < 4:
            continue
        if b[0] < x1 or b[2] > x2 or b[1] < y1 or b[3] > y2:
            continue
        h = b[3] - b[1]
        yc = (b[1] + b[3]) / 2
        ocr_sorted.append((yc, b[3], h, ob))
    ocr_sorted.sort(key=lambda t: t[0])

    # Первая строка с "полем" — условно OCR с небольшим кол-вом символов (поле) или следующая после label
    # Header: все OCR-блоки с h > median_font выше первого блока с коротким текстом (потенциальный label/input)
    first_input_y = y2
    for _, yb, h, ob in ocr_sorted:
        txt = (ob.get("text") or "").strip()
        if len(txt) <= 2 or (len(txt) < 30 and h <= median_font * 1.5):
            first_input_y = min(first_input_y, yb)
            break

    skipped_bboxes: List[List[float]] = []
    header_bottom = y1
    for yc, yb, h, ob in ocr_sorted:
        if yb > first_input_y:
            break
        if h > median_font * 1.2:
            b = ob.get("bbox", [])
            if len(b) >= 4:
                skipped_bboxes.append(list(b))
            header_bottom = max(header_bottom, yb)
    if skipped_bboxes:
        form_start_y = header_bottom + 4.0

    # Логические строки: группируем OCR по вертикальной близости (одна строка = label + поле зона)
    if not ocr_sorted:
        # Fallback: edge-based как раньше, но с типами
        img = cv2.imread(str(image_path))
        if img is None:
            return [], skipped_bboxes, [], form_start_y
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            return [], skipped_bboxes, [], form_start_y
        h_crop, w_crop = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        row_edges = edges.sum(axis=1)
        kernel = min(5, max(3, h_crop // 30))
        if kernel % 2 == 0:
            kernel += 1
        try:
            row_edges_smooth = np.convolve(row_edges, np.ones(kernel) / kernel, mode="same")
        except Exception:
            row_edges_smooth = row_edges
        threshold = float(np.max(row_edges_smooth)) * 0.15 if hasattr(row_edges_smooth, "__len__") and len(row_edges_smooth) else 0
        n = len(row_edges_smooth) if hasattr(row_edges_smooth, "__len__") else 0
        in_row = False
        row_starts: List[int] = []
        row_ends: List[int] = []
        for i in range(1, n):
            val = float(row_edges_smooth[i]) if i < len(row_edges_smooth) else 0
            if not in_row and val > threshold:
                in_row = True
                row_starts.append(i)
            elif in_row and val <= threshold:
                in_row = False
                row_ends.append(i)
        if in_row:
            row_ends.append(n)
        rows = []
        for idx in range(min(len(row_starts), len(row_ends))):
            a, b = row_starts[idx], row_ends[idx]
            if b - a < MIN_ROW_HEIGHT:
                continue
            ry_min = y1 + a
            ry_max = y1 + b
            if ry_min < form_start_y:
                continue
            rows.append(FormRow(
                row_index=len(rows),
                y_min=ry_min,
                y_max=ry_max,
                x_min=x1,
                x_max=x2,
                column_count=1,
                row_type="FIELD",
                height_mode="adaptive",
            ))
        if rows:
            heights = [r.y_max - r.y_min for r in rows]
            median_h = _median(heights)
            textarea_indices: List[int] = []
            for r in rows:
                if (r.y_max - r.y_min) >= median_h * TEXTAREA_HEIGHT_MEDIAN_K:
                    r.row_type = "TEXTAREA"
                    textarea_indices.append(r.row_index)
            for r in rows:
                if r.row_type == "FIELD":
                    for ob in ocr_inside:
                        txt = (ob.get("text") or "").strip().lower()
                        if any(w in txt for w in ACTION_WORDS):
                            b = ob.get("bbox", [])
                            if len(b) >= 4 and r.y_min <= (b[1]+b[3])/2 <= r.y_max:
                                r.row_type = "ACTION"
                                break
            return rows, skipped_bboxes, textarea_indices, form_start_y
        return [], skipped_bboxes, [], form_start_y

    # Группируем OCR в логические строки по Y (близкие по вертикали = одна строка)
    row_groups: List[List[Tuple[float, float, float, Dict]]] = []
    tol = median_font * 2.0 if median_font > 0 else 24.0
    for t in ocr_sorted:
        yc, yb, h, ob = t
        if yc < form_start_y:
            continue
        placed = False
        for g in row_groups:
            g_y_center = sum(x[0] for x in g) / len(g)
            if abs(yc - g_y_center) <= tol:
                g.append(t)
                placed = True
                break
        if not placed:
            row_groups.append([t])

    rows = []
    textarea_indices: List[int] = []
    for idx, g in enumerate(row_groups):
        if not g:
            continue
        ys = [x[1] for x in g]
        ytops = [ob["bbox"][1] for _, _, _, ob in g if len(ob.get("bbox", [])) >= 4]
        ry_min = min(ytops) if ytops else min(ys) - 20
        ry_max = max(ys) + (median_font * 0.5 if median_font > 0 else 10)
        row_h = ry_max - ry_min
        if row_h < MIN_ROW_HEIGHT:
            continue
        # Не размножаем по первой высоте — каждая строка своя высота (adaptive)
        texts = [((ob.get("text") or "").strip().lower(), ob) for _, _, _, ob in g]
        is_action = any(any(w in t for w in ACTION_WORDS) for t, _ in texts)
        is_header = any(h > median_font * 1.2 for _, _, h, _ in g)
        row_type: RowType = "FIELD"
        if is_action:
            row_type = "ACTION"
        elif is_header and idx == 0:
            row_type = "HEADER"
        else:
            # TEXTAREA по высоте позже, после сбора всех row heights
            pass

        rows.append(FormRow(
            row_index=len(rows),
            y_min=ry_min,
            y_max=ry_max,
            x_min=x1,
            x_max=x2,
            column_count=1,
            row_type=row_type,
            height_mode="adaptive",
        ))

    # TEXTAREA: высота > median(row_height) * k
    if rows:
        heights = [r.y_max - r.y_min for r in rows]
        median_h = _median(heights)
        for r in rows:
            if r.row_type != "FIELD":
                continue
            if (r.y_max - r.y_min) >= median_h * TEXTAREA_HEIGHT_MEDIAN_K:
                r.row_type = "TEXTAREA"
                textarea_indices.append(r.row_index)
        # Кнопка внизу не считается input-row: ACTION уже выставлен по тексту
        for r in rows:
            if r.row_type == "FIELD":
                for ob in ocr_inside:
                    txt = (ob.get("text") or "").strip().lower()
                    if any(w in txt for w in ACTION_WORDS):
                        b = ob.get("bbox", [])
                        if len(b) >= 4 and r.y_min <= (b[1] + b[3]) / 2 <= r.y_max:
                            r.row_type = "ACTION"
                            break

    return rows, skipped_bboxes, textarea_indices, form_start_y


def _infer_layout_type(
    rows: List[FormRow],
    ocr_inside: List[Dict[str, Any]],
    container: FormContainer,
) -> str:
    """vertical если variance X-центров мала и мало field-rows с >1 кандидатом по X."""
    if not rows or len(container.bbox) < 4:
        return "vertical"
    x1, y1, x2, y2 = container.bbox[0], container.bbox[1], container.bbox[2], container.bbox[3]
    w = x2 - x1
    x_centers: List[float] = []
    field_row_multi: int = 0
    for ob in ocr_inside:
        b = ob.get("bbox", [])
        if len(b) < 4 or b[1] < y1 or b[3] > y2 or b[0] < x1 or b[2] > x2:
            continue
        xc = (b[0] + b[2]) / 2
        x_centers.append((xc - x1) / w if w > 0 else 0)
        for r in rows:
            if r.row_type != "FIELD" and r.row_type != "TEXTAREA":
                continue
            if r.y_min <= (b[1] + b[3]) / 2 <= r.y_max:
                break
        else:
            continue
        # count per-row OCR with different x
        pass
    if len(x_centers) < 2:
        return "vertical"
    variance = statistics.variance(x_centers)
    if variance <= LAYOUT_VERTICAL_X_VARIANCE_MAX:
        return "vertical"
    # Проверка: сколько field-rows имеют несколько разных X-позиций OCR
    for r in rows:
        if r.row_type not in ("FIELD", "TEXTAREA"):
            continue
        xs_in_row = [(ob["bbox"][0] + ob["bbox"][2]) / 2 for ob in ocr_inside
                     if len((ob.get("bbox") or [])) >= 4
                     and r.y_min <= (ob["bbox"][1] + ob["bbox"][3]) / 2 <= r.y_max]
        if len(set(round(x, 0) for x in xs_in_row)) > 1:
            field_row_multi += 1
    if field_row_multi == 0:
        return "vertical"
    return "grid"


def _build_columns_inside_container(
    image_path: str,
    container: FormContainer,
    rows: List[FormRow],
    ocr_inside: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[FormColumn], List[Tuple[float, float]]]:
    """Колонки по X-центрам. Вызывать только при layout_type != vertical."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None or not rows:
        return [], []
    bbox = container.bbox
    if len(bbox) < 4:
        return [], []
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return [], []
    w_crop = crop.shape[1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    x_centers: List[float] = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 200:
            continue
        rx, _, rw, _ = cv2.boundingRect(c)
        x_centers.append(x1 + rx + rw / 2.0)
    for ob in (ocr_inside or []):
        b = ob.get("bbox", [])
        if len(b) >= 4 and b[1] >= y1 and b[3] <= y2 and b[0] >= x1 and b[2] <= x2:
            x_centers.append((b[0] + b[2]) / 2)

    if not x_centers:
        columns = [FormColumn(col_index=0, x_min=float(x1), x_max=float(x2))]
        return columns, [(float(x1), float(x2))]

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

    columns = []
    boundaries = []
    if len(clusters) >= 2:
        for i, cl in enumerate(clusters):
            cx = statistics.median(cl)
            half = w_crop / max(len(clusters), 1) / 2
            cx1 = max(float(x1), cx - half)
            cx2 = min(float(x2), cx + half)
            columns.append(FormColumn(col_index=i, x_min=cx1, x_max=cx2))
            boundaries.append((cx1, cx2))
        boundaries.sort(key=lambda p: p[0])
    else:
        columns.append(FormColumn(col_index=0, x_min=float(x1), x_max=float(x2)))
        boundaries.append((float(x1), float(x2)))
    return columns, boundaries


def build_form_inner_layout(
    image_path: str,
    container: FormContainer,
    ocr_inside: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[FormSkeleton], Dict[str, Any]]:
    """
    FormInnerLayout: RowDetector (типы строк, OCR-якоря, adaptive height) + ColumnDetector только при layout_type != vertical.
    """
    ocr_inside = ocr_inside or []
    rows, skipped_bboxes, textarea_row_indices, form_start_y = _build_rows_inside_container(
        image_path, container, ocr_inside
    )

    if not rows:
        n_default = max(2, int((container.bbox[3] - form_start_y) // 50))
        step = (container.bbox[3] - form_start_y) / n_default
        for i in range(n_default):
            rows.append(FormRow(
                row_index=i,
                y_min=form_start_y + i * step,
                y_max=form_start_y + (i + 1) * step,
                x_min=container.bbox[0],
                x_max=container.bbox[2],
                column_count=1,
                row_type="FIELD",
                height_mode="fixed",
            ))

    layout_type = _infer_layout_type(rows, ocr_inside, container)

    if layout_type == "vertical":
        columns = [FormColumn(col_index=0, x_min=container.bbox[0], x_max=container.bbox[2])]
        column_boundaries = [(container.bbox[0], container.bbox[2])]
        for r in rows:
            r.column_count = 1
    else:
        columns, column_boundaries = _build_columns_inside_container(image_path, container, rows, ocr_inside)
        for r in rows:
            if r.row_type == "TEXTAREA":
                r.column_count = 1
            else:
                r.column_count = len(columns)

    skeleton = FormSkeleton(
        form_region=container,
        rows=rows,
        columns=columns if columns else None,
        column_boundaries=column_boundaries if column_boundaries else [(container.bbox[0], container.bbox[2])],
        layout_type=layout_type,
    )
    diag = {
        "n_rows": len(rows),
        "n_columns": len(columns),
        "layout_type": layout_type,
        "skipped_bboxes": skipped_bboxes,
        "textarea_row_indices": textarea_row_indices,
    }
    return skeleton, diag


def visualize_rows(
    image_path: str,
    container: FormContainer,
    rows: List[FormRow],
    output_path: str,
) -> None:
    """Сохраняет rows.png."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    for r in rows:
        y1, y2 = int(r.y_min), int(r.y_max)
        cv2.rectangle(out, (int(r.x_min), y1), (int(r.x_max), y2), (0, 255, 255), 1)
        cv2.putText(out, "R%d" % r.row_index, (int(r.x_min) + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_inner_layout: saved %s", output_path)


def visualize_rows_with_types(
    image_path: str,
    rows: List[FormRow],
    output_path: str,
) -> None:
    """Сохраняет rows_with_types.png — цвет по row_type."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    colors = {"HEADER": (255, 200, 0), "FIELD": (0, 255, 255), "TEXTAREA": (255, 150, 255), "ACTION": (0, 200, 255), "SPACER": (180, 180, 180)}
    for r in rows:
        y1, y2 = int(r.y_min), int(r.y_max)
        color = colors.get(r.row_type, (200, 200, 200))
        cv2.rectangle(out, (int(r.x_min), y1), (int(r.x_max), y2), color, 1)
        cv2.putText(out, "%s R%d" % (r.row_type, r.row_index), (int(r.x_min) + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_inner_layout: saved %s", output_path)


def visualize_skipped_rows(
    image_path: str,
    skipped_bboxes: List[List[float]],
    output_path: str,
) -> None:
    """Сохраняет skipped_rows.png — зоны header, вынесенные из RowDetector."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    for b in skipped_bboxes:
        if len(b) >= 4:
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(out, "skipped", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_inner_layout: saved %s", output_path)


def visualize_textarea_rows(
    image_path: str,
    rows: List[FormRow],
    textarea_row_indices: List[int],
    output_path: str,
) -> None:
    """Сохраняет textarea_rows.png — только TEXTAREA-строки."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    for r in rows:
        if r.row_index in textarea_row_indices or r.row_type == "TEXTAREA":
            y1, y2 = int(r.y_min), int(r.y_max)
            cv2.rectangle(out, (int(r.x_min), y1), (int(r.x_max), y2), (255, 100, 255), 2)
            cv2.putText(out, "TEXTAREA R%d" % r.row_index, (int(r.x_min) + 2, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(output_path, out)
    logger.debug("form_inner_layout: saved %s", output_path)
