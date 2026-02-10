"""
Уровень 1 — Семантические макро-регионы.

Строит Region объекты: header, footer, sidebar, content, form_area.
Использует цвет фона, ширину, позицию по Y, повторяемость. Не использует Detectron2.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.atoms_v2.experimental_v2.models import PageSegment, Region, RegionType

logger = logging.getLogger(__name__)

# Доли высоты страницы для header/footer
HEADER_MAX_RATIO = 0.15
FOOTER_MAX_RATIO = 0.20
# Узкая полоса по боку = sidebar
SIDEBAR_MAX_WIDTH_RATIO = 0.25
# Минимальная высота контентной зоны для form_area
FORM_AREA_MIN_HEIGHT_RATIO = 0.15


def build_semantic_regions(
    image_path: str,
    segments: List[PageSegment],
    page_diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Region], Dict[str, Any]]:
    """
    Строит семантические регионы из сегментов страницы.
    Возвращает список Region и диагностику.
    """
    import cv2
    h = page_diagnostics.get("image_h", 0) if page_diagnostics else 0
    w = page_diagnostics.get("image_w", 0) if page_diagnostics else 0
    if h <= 0 or w <= 0:
        img = cv2.imread(str(image_path))
        if img is None:
            return [], {"error": "image_not_read"}
        h, w = img.shape[:2]

    regions: List[Region] = []
    diag: Dict[str, Any] = {"image_h": h, "image_w": w}

    # Header: верхняя часть до HEADER_MAX_RATIO
    header_y2 = h * HEADER_MAX_RATIO
    regions.append(Region(
        region_type="header",
        bbox=[0.0, 0.0, float(w), header_y2],
        confidence=0.7,
        metadata={"source": "position_y"},
    ))

    # Footer: нижняя часть
    footer_y1 = h * (1.0 - FOOTER_MAX_RATIO)
    regions.append(Region(
        region_type="footer",
        bbox=[0.0, footer_y1, float(w), float(h)],
        confidence=0.7,
        metadata={"source": "position_y"},
    ))

    # Content: середина страницы (между header и footer)
    content_y1 = header_y2
    content_y2 = footer_y1
    if content_y2 > content_y1:
        regions.append(Region(
            region_type="content",
            bbox=[0.0, content_y1, float(w), content_y2],
            confidence=0.8,
            metadata={"source": "position_y"},
        ))

    # Form_area: внутри content — зона с повторяющимися "контентными" сегментами (ритм строк формы)
    # Упрощение: form_area = content целиком; ниже FormSkeletonBuilder сузит по скелету
    form_candidates = [s for s in segments if not s.is_likely_gap and s.y_min >= content_y1 and s.y_max <= content_y2]
    if form_candidates:
        fy1 = min(s.y_min for s in form_candidates)
        fy2 = max(s.y_max for s in form_candidates)
        if fy2 - fy1 >= h * FORM_AREA_MIN_HEIGHT_RATIO:
            regions.append(Region(
                region_type="form_area",
                bbox=[0.0, fy1, float(w), fy2],
                confidence=0.6,
                metadata={"source": "segment_rhythm", "n_content_strips": len(form_candidates)},
            ))
            diag["form_area_from_segments"] = True
    else:
        regions.append(Region(
            region_type="form_area",
            bbox=[0.0, content_y1, float(w), content_y2],
            confidence=0.5,
            metadata={"source": "fallback_content"},
        ))
        diag["form_area_from_segments"] = False

    # Sidebar: узкая вертикальная полоса по краю (если сегменты по краю отличаются по ширине)
    # Упрощение: не выделяем sidebar отдельно, если нет явной узкой колонки
    diag["regions_count"] = len(regions)
    return regions, diag


def visualize_semantic_regions(
    image_path: str,
    regions: List[Region],
    output_path: str,
) -> None:
    """Сохраняет визуализацию семантических регионов (уровень 1)."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = img.copy()
    colors = {
        "header": (180, 120, 255),
        "footer": (120, 180, 255),
        "sidebar": (255, 200, 100),
        "content": (100, 255, 180),
        "form_area": (255, 100, 100),
        "unknown": (128, 128, 128),
    }
    for r in regions:
        bbox = r.bbox
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = colors.get(r.region_type, colors["unknown"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, r.region_type, (x1, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(output_path, out)
    logger.debug("semantic_region_builder: saved %s", output_path)
