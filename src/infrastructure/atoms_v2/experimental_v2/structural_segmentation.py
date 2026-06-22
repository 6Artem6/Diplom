"""
S3 — Structural Segmentation (State Machine Architecture)

Группировка элементов в строки (rows) на основе вертикального выравнивания.

Использует ТОЛЬКО относительные размеры:
- gap = median_input_height * RELATIVE_GAP_FACTOR
- y_tolerance = median_input_height * RELATIVE_Y_TOLERANCE

Не модифицирует geometry элементов!
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .visual_geometry_extractor import VisualElement, GeometryContext, bbox_contains
from .ocr_extractor import OCRBlock, LanguageInfo, merge_ocr_blocks_in_row

logger = logging.getLogger(__name__)


# =============================================================================
# RELATIVE THRESHOLDS (no absolute px!)
# =============================================================================

class RowThresholds:
    """All thresholds relative to median_input_height."""
    
    # Y-overlap tolerance for same row
    Y_OVERLAP_TOLERANCE = 0.3  # 30% of element height
    
    # Gap between rows (relative to median_input_height)
    ROW_GAP_MIN = 0.3   # min 30% of median to start new row
    ROW_GAP_MAX = 2.0   # max 200% gap allowed within form
    
    # Minimum row height (relative to median_input_height)
    ROW_HEIGHT_MIN = 0.5  # min 50% of median
    
    # Maximum elements per row
    MAX_ELEMENTS_PER_ROW = 10


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class RowElement:
    """Element within a row (reference to VisualElement or OCRBlock)."""
    visual_element: Optional[VisualElement] = None
    ocr_block: Optional[OCRBlock] = None
    
    @property
    def bbox(self) -> List[float]:
        if self.visual_element:
            return self.visual_element.bbox
        if self.ocr_block:
            return self.ocr_block.bbox
        return [0, 0, 0, 0]
    
    @property
    def element_type(self) -> str:
        if self.visual_element:
            return self.visual_element.element_type
        if self.ocr_block:
            if self.ocr_block.is_label_hint:
                return "text_label"
            if self.ocr_block.is_value_hint:
                return "text_value"
            return "text"
        return "unknown"
    
    @property
    def y_center(self) -> float:
        bbox = self.bbox
        return (bbox[1] + bbox[3]) / 2
    
    @property
    def height(self) -> float:
        bbox = self.bbox
        return bbox[3] - bbox[1]


@dataclass
class FormRow:
    """Row of elements in the form."""
    elements: List[RowElement]
    y_min: float
    y_max: float
    row_index: int
    
    # Computed properties
    row_type: str = "mixed"  # header, field_row, button_row, checkbox_row
    
    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def x_min(self) -> float:
        if not self.elements:
            return 0
        return min(e.bbox[0] for e in self.elements)
    
    @property
    def x_max(self) -> float:
        if not self.elements:
            return 0
        return max(e.bbox[2] for e in self.elements)


@dataclass
class S3Result:
    """Result of S3 — Structural Segmentation."""
    rows: List[FormRow]
    orphan_elements: List[RowElement]  # элементы, не попавшие в строки
    diagnostics: Dict[str, Any]


# =============================================================================
# ROW CLUSTERING FUNCTIONS
# =============================================================================

def elements_on_same_row(
    elem1: RowElement,
    elem2: RowElement,
    tolerance_ratio: float = RowThresholds.Y_OVERLAP_TOLERANCE,
) -> bool:
    """
    Check if two elements are on the same row (vertical overlap).
    
    Uses relative tolerance based on element heights.
    """
    b1 = elem1.bbox
    b2 = elem2.bbox
    
    # Y overlap check
    y1_min, y1_max = b1[1], b1[3]
    y2_min, y2_max = b2[1], b2[3]
    
    # Compute overlap
    overlap_min = max(y1_min, y2_min)
    overlap_max = min(y1_max, y2_max)
    
    if overlap_max <= overlap_min:
        # No direct overlap, check if close enough
        smaller_height = min(elem1.height, elem2.height)
        gap = max(y1_min, y2_min) - min(y1_max, y2_max)
        return gap <= smaller_height * tolerance_ratio
    
    # Has overlap, check if significant
    overlap_height = overlap_max - overlap_min
    smaller_height = min(elem1.height, elem2.height)
    
    return overlap_height >= smaller_height * (1 - tolerance_ratio)


def cluster_into_rows(
    elements: List[RowElement],
    median_height: float,
) -> List[List[RowElement]]:
    """
    Кластеризация по строкам. Только одна линия по вертикали: center_y в пределах 0.5*median.
    По вертикали не объединяем — далёкий текст не должен попадать в одну строку.
    """
    if not elements:
        return []
    
    sorted_elems = sorted(elements, key=lambda e: e.y_center)
    rows: List[List[RowElement]] = []
    max_center_y_dist = 0.5 * max(1.0, median_height)  # одна строка, допуск ±25% по высоте

    for elem in sorted_elems:
        assigned = False
        for row in rows:
            for row_elem in row:
                if abs(elem.y_center - row_elem.y_center) > max_center_y_dist:
                    continue
                if elements_on_same_row(elem, row_elem):
                    if len(row) < RowThresholds.MAX_ELEMENTS_PER_ROW:
                        row.append(elem)
                        assigned = True
                        break
            if assigned:
                break
        if not assigned:
            rows.append([elem])
    return rows


def classify_row_type(
    row: FormRow,
    is_first: bool,
    is_last: bool,
) -> str:
    """
    Classify row type based on content.
    """
    if not row.elements:
        return "empty"
    
    types = [e.element_type for e in row.elements]
    
    # Header detection (first row, mostly text, large)
    if is_first and len(types) <= 2:
        text_count = sum(1 for t in types if t.startswith("text"))
        if text_count == len(types):
            return "header"
    
    # Button row (last row or row with buttons)
    button_count = sum(1 for t in types if t in ("button", "action"))
    if button_count > 0:
        if button_count == len(types):
            return "button_row"
        if is_last:
            return "button_row"
    
    # Checkbox/Radio row
    checkbox_count = sum(1 for t in types if t in ("checkbox", "radio"))
    if checkbox_count >= 2:
        return "checkbox_row"
    if checkbox_count == 1 and len(types) <= 3:
        return "checkbox_row"
    
    # Input row
    input_count = sum(1 for t in types if t in ("input", "textarea"))
    if input_count >= 1:
        return "field_row"
    
    # Label-only row
    text_count = sum(1 for t in types if t.startswith("text"))
    if text_count == len(types):
        return "label_row"
    
    return "mixed"


def merge_close_rows(
    rows: List[FormRow],
    median_height: float,
) -> List[FormRow]:
    """
    Merge rows that are too close together.
    
    Uses relative gap threshold.
    """
    if len(rows) <= 1:
        return rows
    
    # Sort by y_min
    sorted_rows = sorted(rows, key=lambda r: r.y_min)
    
    merged = [sorted_rows[0]]
    min_gap = median_height * RowThresholds.ROW_GAP_MIN
    
    for row in sorted_rows[1:]:
        prev = merged[-1]
        gap = row.y_min - prev.y_max
        
        if gap < min_gap:
            # Merge into previous row
            new_elements = prev.elements + row.elements
            new_y_min = min(prev.y_min, row.y_min)
            new_y_max = max(prev.y_max, row.y_max)
            
            merged[-1] = FormRow(
                elements=new_elements,
                y_min=new_y_min,
                y_max=new_y_max,
                row_index=prev.row_index,
            )
        else:
            # Keep as separate row
            row.row_index = len(merged)
            merged.append(row)
    
    return merged


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def segment_into_rows(
    visual_elements: List[VisualElement],
    ocr_blocks: List[OCRBlock],
    context: GeometryContext,
) -> S3Result:
    """
    S3 — Structural Segmentation.
    
    Groups elements into rows. Does NOT modify element geometry.
    
    Args:
        visual_elements: элементы из S1
        ocr_blocks: OCR-блоки из S2
        context: контекст с median sizes
    
    Returns:
        S3Result с rows и orphans
    """
    diagnostics: Dict[str, Any] = {
        "total_visual": len(visual_elements),
        "total_ocr": len(ocr_blocks),
        "rows_before_merge": 0,
        "rows_after_merge": 0,
        "orphans": 0,
        "row_types": {},
    }
    
    # Create RowElement wrappers
    all_elements: List[RowElement] = []
    
    # Add visual elements (exclude containers)
    for ve in visual_elements:
        if ve.is_container:
            continue
        all_elements.append(RowElement(visual_element=ve))
    
    # Добавляем ВСЕ OCR-блоки, чтобы они попали в строки и участвовали в объединении.
    # Иначе блоки, перекрывающие визуальные (заголовок, подписи), не попадали в кластер —
    # объединение не срабатывало, замена на объединённый блок к S4 не проходила.
    for ocr in ocr_blocks:
        all_elements.append(RowElement(ocr_block=ocr))
    
    logger.debug(f"S3: {len(all_elements)} elements to cluster")
    
    if not all_elements:
        return S3Result(rows=[], orphan_elements=[], diagnostics=diagnostics)
    
    # Cluster into rows
    row_clusters = cluster_into_rows(all_elements, context.median_input_height)
    diagnostics["rows_before_merge"] = len(row_clusters)
    
    # Convert to FormRow objects; внутри каждой строки объединяем OCR-блоки (merge после S3)
    rows = []
    median_h = max(1.0, context.median_input_height)
    for i, cluster in enumerate(row_clusters):
        if not cluster:
            continue
        
        visual_elements = [e for e in cluster if e.visual_element is not None]
        ocr_elements = [e for e in cluster if e.ocr_block is not None]
        if ocr_elements:
            ocr_blocks = [e.ocr_block for e in ocr_elements]
            merged_ocr = merge_ocr_blocks_in_row(ocr_blocks, median_h)
            # Внешний поглощает внутренний: удаляем только те визуальные, чей bbox целиком внутри merged OCR
            # (маленький label внутри зоны текста — удалить). Input/textarea снаружи не удаляем.
            keep_visual = [
                e for e in visual_elements
                if not any(bbox_contains(m.bbox, e.visual_element.bbox, 0.8) for m in merged_ocr)
            ]
            # Merged OCR не добавляем как отдельный элемент, если он внутри визуального (placeholder внутри input)
            merged_to_add = [
                ob for ob in merged_ocr
                if not any(
                    bbox_contains(ve.visual_element.bbox, ob.bbox, 0.8)
                    for ve in visual_elements
                )
            ]
            new_elements = keep_visual + [RowElement(ocr_block=ob) for ob in merged_to_add]
        else:
            new_elements = cluster
        
        y_min = min(e.bbox[1] for e in new_elements)
        y_max = max(e.bbox[3] for e in new_elements)
        
        row = FormRow(
            elements=new_elements,
            y_min=y_min,
            y_max=y_max,
            row_index=i,
        )
        rows.append(row)
    
    # По вертикали строки не объединяем — только горизонтальное объединение блоков (внутри строки, допуск ±25%).
    # merge_close_rows не вызываем, чтобы две отдельные строки (напр. два чекбокса) не склеивались в одну.
    diagnostics["rows_after_merge"] = len(rows)
    
    # Classify row types
    for i, row in enumerate(rows):
        is_first = (i == 0)
        is_last = (i == len(rows) - 1)
        row.row_type = classify_row_type(row, is_first, is_last)
        
        diagnostics["row_types"][row.row_type] = diagnostics["row_types"].get(row.row_type, 0) + 1
    
    # Sort elements within each row by X position
    for row in rows:
        row.elements.sort(key=lambda e: e.bbox[0])
    
    # Identify orphan elements (elements with unusual height)
    orphans: List[RowElement] = []
    # Currently no orphans — all elements assigned to rows
    diagnostics["orphans"] = len(orphans)
    
    logger.info(f"S3 completed: {len(rows)} rows, types={diagnostics['row_types']}")
    
    return S3Result(
        rows=rows,
        orphan_elements=orphans,
        diagnostics=diagnostics,
    )


def compute_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute overlap ratio (intersection / smaller area)."""
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0
    
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    smaller = min(area1, area2)
    return inter / max(1, smaller)
