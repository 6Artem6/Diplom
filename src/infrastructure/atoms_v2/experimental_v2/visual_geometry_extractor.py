"""
S1 — Visual Geometry Extractor (State Machine Architecture)

ЕДИНСТВЕННЫЙ ЭТАП где разрешено:
- находить bbox
- классифицировать visual type  
- делать NMS

После S1 visual_elements IMMUTABLE.

ИНВАРИАНТЫ:
- Никаких абсолютных размеров (только relative_to_container, relative_to_median, aspect_ratio)
- INPUT не может содержать другие bbox
- TEXTAREA не может содержать другие bbox
- CONTAINER не получает semantic role в S4
- OCR overlap check — текст без рамки не может быть INPUT
- NMS type-aware: CONTAINER не подавляет вложенные элементы
- Symmetry recovery выполняется ДО NMS

ПОРЯДОК ОБРАБОТКИ:
1. Найти все raw bbox
2. Symmetry recovery (до NMS!)
3. Определить parent-child отношения
4. Классифицировать (с учётом вложенности)
5. NMS по типам (type-aware)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


# =============================================================================
# ELEMENT TYPES
# =============================================================================

class ElementTypes:
    """Valid element types."""
    INPUT = "input"
    TEXTAREA = "textarea"
    ACTION = "action"       # button
    CHECKBOX = "checkbox"
    RADIO = "radio"
    CONTAINER = "container"  # содержит другие элементы, не участвует в slot assignment
    DECORATION = "decoration"  # визуальный элемент без семантики
    LABEL = "label"
    UNKNOWN = "unknown"


# =============================================================================
# DATA MODELS (Immutable after S1)
# =============================================================================

@dataclass
class VisualElement:
    """Immutable visual element detected in S1."""
    bbox: List[float]  # [x1, y1, x2, y2]
    element_type: str  # ElementTypes.*
    confidence: float
    source: str  # detection method
    
    # Optional fields
    is_checked: Optional[bool] = None  # для checkbox/radio
    has_border: bool = False
    is_container: bool = False  # содержит другие элементы
    contains_text: bool = False  # содержит текст (OCR)
    
    # Relative metrics (computed)
    relative_width: float = 0.0  # width / container_width
    relative_height: float = 0.0  # height / median_input_height
    aspect_ratio: float = 0.0
    
    # Parent-child relations (set during classification)
    parent_id: Optional[int] = None
    child_ids: List[int] = field(default_factory=list)


@dataclass
class GeometryContext:
    """Context for relative calculations."""
    container_bbox: List[float]
    container_width: float
    container_height: float
    median_input_height: float = 35.0  # будет вычислен
    median_input_width: float = 200.0  # будет вычислен
    median_input_area: float = 7000.0  # будет вычислен (width * height)
    median_checkbox_size: float = 16.0  # будет вычислен
    
    # OCR blocks for overlap check
    ocr_blocks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class S1Result:
    """Result of S1 — Visual Geometry Extraction."""
    visual_elements: List[VisualElement]
    context: GeometryContext
    diagnostics: Dict[str, Any]


# =============================================================================
# RELATIVE THRESHOLDS (NO ABSOLUTE PX VALUES!)
# =============================================================================

class RelativeThresholds:
    """All thresholds are relative to container or median. NO ABSOLUTE PX!"""
    
    # ===========================================
    # CHECKBOX/RADIO — must be SMALL
    # ===========================================
    CHECKBOX_SIZE_MIN_RATIO = 0.012  # min 1.2% of container diagonal
    CHECKBOX_SIZE_MAX_RATIO = 0.07   # max 7% of container diagonal
    CHECKBOX_ASPECT_MIN = 0.8        # aspect must be 0.8-1.2
    CHECKBOX_ASPECT_MAX = 1.2
    CHECKBOX_MAX_AREA_RATIO = 0.5    # NEW: max 50% of median_input_area
    CHECKBOX_OCR_OVERLAP_MAX = 0.25  # reject if >25% overlap with OCR
    CHECKBOX_OCR_AREA_MAX = 0.4      # NEW: reject if OCR area > 40% of bbox
    CHECKBOX_REQUIRE_BORDER = True
    CHECKBOX_REQUIRE_CONTRAST = True
    CHECKBOX_BORDER_THIN_RATIO = 0.6 # border <= 60% of median_input_border
    
    # Symmetry recovery
    CHECKBOX_SIZE_TOLERANCE = 0.2    # ±20% size difference
    CHECKBOX_SEARCH_DISTANCE = 3.0   # 3x checkbox width
    CHECKBOX_VERTICAL_TOLERANCE = 0.3  # 30% of height
    
    # ===========================================
    # INPUT — field-like element
    # ===========================================
    INPUT_HEIGHT_MIN_RATIO = 0.5   # min 50% of median (relaxed)
    INPUT_HEIGHT_MAX_RATIO = 1.5   # max 150% of median
    INPUT_ASPECT_MIN = 2.0         # min width/height ratio (RELAXED from 2.5)
    INPUT_MUST_HAVE_BORDER = True  # border required
    INPUT_MAX_AREA_RATIO = 2.5     # max 2.5 * median_input_area
    INPUT_MIN_PADDING_RATIO = 0.05 # min 5% internal padding (used elsewhere)
    
    # ===========================================
    # TEXTAREA — tall field (STRICTER)
    # ===========================================
    TEXTAREA_HEIGHT_MIN_RATIO = 2.5  # min 250% of median (STRICTER)
    TEXTAREA_MUST_HAVE_BORDER = True
    TEXTAREA_MAX_ASPECT = 3.0  # max aspect (narrower than input)
    
    # ===========================================
    # ACTION/BUTTON
    # ===========================================
    ACTION_WIDTH_MIN_RATIO = 0.1    # min 10% of container width
    ACTION_WIDTH_MAX_RATIO = 0.7    # max 70% of container width
    ACTION_HEIGHT_MIN_RATIO = 0.5   # min 50% of median_input_height
    ACTION_HEIGHT_MAX_RATIO = 2.0   # max 200% of median_input_height
    ACTION_BOTTOM_ZONE_RATIO = 0.3  # нижние 30% контейнера — зона кнопок
    
    # ===========================================
    # LABEL — text without border
    # ===========================================
    LABEL_HEIGHT_MAX_RATIO = 0.7    # max 70% of median_input_height
    LABEL_ASPECT_MIN = 2.0          # wider than tall
    
    # ===========================================
    # CONTAINER — determined by CONTEXT, not geometry!
    # ===========================================
    CONTAINER_AREA_THRESHOLD = 3.0    # area >= 3 * median_input_area
    CONTAINER_MIN_PADDING_RATIO = 0.10  # internal padding >= 10%
    CONTAINER_MUST_HAVE_FIELD_CHILDREN = True  # must contain field-like elements
    CONTAINER_CHILD_AREA_MIN_RATIO = 0.15  # child must occupy >= 15% of parent
    
    # ===========================================
    # OCR overlap
    # ===========================================
    OCR_OVERLAP_REJECT_RATIO = 0.4  # reject if >40% overlap with OCR
    
    # ===========================================
    # NMS
    # ===========================================
    NMS_IOU_THRESHOLD = 0.5
    NMS_IOU_THRESHOLD_SMALL = 0.3  # stricter for checkbox/radio/button
    
    # ===========================================
    # Label binding (for S4)
    # ===========================================
    LABEL_MAX_DISTANCE_RATIO = 1.2  # max 1.2 * median_input_width


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_container_diagonal(ctx: GeometryContext) -> float:
    """Compute container diagonal for relative sizing."""
    return (ctx.container_width ** 2 + ctx.container_height ** 2) ** 0.5


def compute_bbox_area(bbox: List[float]) -> float:
    """Compute bbox area."""
    if len(bbox) < 4:
        return 0.0
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def compute_overlap_ratio(bbox1: List[float], bbox2: List[float]) -> float:
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
    area1 = compute_bbox_area(bbox1)
    area2 = compute_bbox_area(bbox2)
    
    smaller = min(area1, area2)
    return inter / max(1, smaller)


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute IoU of two bboxes."""
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0
    
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = compute_bbox_area(bbox1)
    area2 = compute_bbox_area(bbox2)
    
    return inter / max(1, area1 + area2 - inter)


def bbox_contains(outer: List[float], inner: List[float], threshold: float = 0.8) -> bool:
    """Check if outer bbox contains inner bbox."""
    if len(outer) < 4 or len(inner) < 4:
        return False
    
    # Check if inner is mostly inside outer
    overlap = compute_overlap_ratio(outer, inner)
    inner_area = compute_bbox_area(inner)
    outer_area = compute_bbox_area(outer)
    
    if inner_area >= outer_area:
        return False  # inner larger than outer
    
    return overlap >= threshold


def has_visible_border(image, bbox: List[float], threshold: float = 0.04) -> bool:
    """Check if bbox has visible border (edge density on perimeter)."""
    import cv2
    import numpy as np
    
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = image.shape[:2]
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return False
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # Use lower Canny thresholds to detect lighter borders
    edges = cv2.Canny(gray, 30, 100)
    
    # Check edge density on perimeter
    roi_h, roi_w = edges.shape[:2]
    strip_size = max(2, min(5, roi_w // 10, roi_h // 10))
    
    perimeter_edges = 0
    perimeter_pixels = 0
    
    # Top, bottom, left, right strips
    for strip in [edges[:strip_size, :], edges[-strip_size:, :],
                  edges[:, :strip_size], edges[:, -strip_size:]]:
        perimeter_edges += np.sum(strip > 0)
        perimeter_pixels += strip.size
    
    edge_density = perimeter_edges / max(1, perimeter_pixels)
    
    # #region agent log
    import json, time
    log_path = "/app/debug/debug.log"
    border_log = {
        "timestamp": int(time.time() * 1000),
        "location": "has_visible_border",
        "hypothesisId": "H7-border",
        "message": "border_check",
        "data": {
            "bbox": [float(x) for x in bbox],
            "edge_density": round(float(edge_density), 4),
            "threshold": threshold,
            "result": bool(edge_density >= threshold),
            "strip_size": int(strip_size),
            "perimeter_edges": int(perimeter_edges),
            "perimeter_pixels": int(perimeter_pixels),
        }
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(border_log) + "\n")
    # #endregion
    
    return edge_density >= threshold


def has_background_fill(image, bbox: List[float], threshold: float = 20) -> bool:
    """Check if bbox has distinct background fill (different from surrounding)."""
    import cv2
    import numpy as np
    
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = image.shape[:2]
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    inner_mean = gray.mean()
    
    # Compare with surrounding area
    pad = max(5, min(20, (x2 - x1) // 4))
    outer_vals = []
    
    if y1 - pad >= 0:
        outer_vals.append(image[max(0, y1 - pad):y1, x1:x2].mean())
    if y2 + pad <= h:
        outer_vals.append(image[y2:min(h, y2 + pad), x1:x2].mean())
    
    if outer_vals:
        outer_mean = np.mean(outer_vals)
        return abs(inner_mean - outer_mean) >= threshold
    
    return False


def compute_checkbox_ocr_overlap(
    bbox: List[float],
    ocr_blocks: List[Dict[str, Any]],
) -> float:
    """
    Compute maximum OCR overlap ratio for checkbox detection.
    
    Returns: max overlap ratio (0.0-1.0)
    """
    max_overlap = 0.0
    
    for ocr in ocr_blocks:
        ocr_bbox = ocr.get("bbox", [])
        if len(ocr_bbox) < 4:
            continue
        
        overlap = compute_overlap_ratio(bbox, ocr_bbox)
        max_overlap = max(max_overlap, overlap)
    
    return max_overlap


def check_ocr_overlap(
    bbox: List[float],
    ocr_blocks: List[Dict[str, Any]],
    image,
) -> Tuple[bool, bool]:
    """
    Check OCR overlap for bbox.
    
    Returns: (has_significant_overlap, contains_text)
    """
    T = RelativeThresholds
    contains_text = False
    significant_overlap = False
    
    for ocr in ocr_blocks:
        ocr_bbox = ocr.get("bbox", [])
        if len(ocr_bbox) < 4:
            continue
        
        overlap = compute_overlap_ratio(bbox, ocr_bbox)
        
        if overlap > 0.1:  # any overlap
            contains_text = True
        
        if overlap > T.OCR_OVERLAP_REJECT_RATIO:
            # Check if has visible border or background fill
            if not has_visible_border(image, bbox) and not has_background_fill(image, bbox):
                significant_overlap = True
    
    return significant_overlap, contains_text


# =============================================================================
# PARENT-CHILD DETECTION
# =============================================================================

@dataclass
class RawBBox:
    """Raw bbox before classification."""
    bbox: List[float]
    source: str
    has_border: bool
    contains_text: bool
    children: List[int] = field(default_factory=list)  # indices of children
    parent: Optional[int] = None


def find_parent_child_relations(raw_bboxes: List[RawBBox]) -> None:
    """
    Find parent-child relations between bboxes.
    
    Updates children and parent fields in-place.
    """
    T = RelativeThresholds
    n = len(raw_bboxes)
    
    # #region agent log
    import json, time
    log_path = "/app/debug/debug.log"
    relations_found = 0
    # #endregion
    
    for i in range(n):
        parent_bbox = raw_bboxes[i].bbox
        parent_area = compute_bbox_area(parent_bbox)
        
        for j in range(n):
            if i == j:
                continue
            
            child_bbox = raw_bboxes[j].bbox
            child_area = compute_bbox_area(child_bbox)
            
            # Check if j is inside i
            if bbox_contains(parent_bbox, child_bbox, threshold=0.7):
                # Check area ratio
                area_ratio = child_area / max(1, parent_area)
                
                if area_ratio >= T.CONTAINER_CHILD_AREA_MIN_RATIO:
                    raw_bboxes[i].children.append(j)
                    if raw_bboxes[j].parent is None:
                        raw_bboxes[j].parent = i
                    relations_found += 1
    
    # #region agent log
    parent_child_log = {
        "timestamp": int(time.time() * 1000),
        "location": "visual_geometry_extractor.py:find_parent_child_relations",
        "hypothesisId": "H4-parent-child",
        "message": "parent_child_relations",
        "data": {
            "total_bboxes": n,
            "relations_found": relations_found,
            "bboxes_with_children": sum(1 for r in raw_bboxes if r.children),
            "bboxes_with_parent": sum(1 for r in raw_bboxes if r.parent is not None),
        }
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(parent_child_log) + "\n")
    # #endregion


# =============================================================================
# CLASSIFICATION
# =============================================================================

def check_internal_contrast(image, bbox: List[float], threshold: float = 30.0) -> bool:
    """Check if bbox has internal contrast (not just uniform fill)."""
    import cv2
    import numpy as np
    
    img_h, img_w = image.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(img_w, int(bbox[2]))
    y2 = min(img_h, int(bbox[3]))
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    roi = image[y1:y2, x1:x2]
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Measure standard deviation — low means uniform, high means contrast
    std_dev = float(np.std(roi))
    return std_dev > threshold


def check_internal_padding(bbox: List[float], ocr_blocks: List[Dict], min_ratio: float = 0.05) -> bool:
    """
    Check if bbox has internal padding (content doesn't fill edge to edge).
    Returns True if there's at least min_ratio padding on sides.
    """
    bw = bbox[2] - bbox[0]
    if bw <= 0:
        return False
    
    # Find OCR text inside this bbox
    texts_inside = []
    for ocr in ocr_blocks:
        ob = ocr.get('bbox', [0, 0, 0, 0])
        # Check if OCR block is mostly inside bbox
        inter_x1 = max(bbox[0], ob[0])
        inter_y1 = max(bbox[1], ob[1])
        inter_x2 = min(bbox[2], ob[2])
        inter_y2 = min(bbox[3], ob[3])
        
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            ocr_area = max(1, (ob[2] - ob[0]) * (ob[3] - ob[1]))
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            if inter_area / ocr_area > 0.5:
                texts_inside.append(ob)
    
    if not texts_inside:
        # No text inside — assume has padding (empty input)
        return True
    
    # Calculate left and right padding
    min_text_x = min(t[0] for t in texts_inside)
    max_text_x = max(t[2] for t in texts_inside)
    
    left_padding = min_text_x - bbox[0]
    right_padding = bbox[2] - max_text_x
    
    return (left_padding / bw >= min_ratio) or (right_padding / bw >= min_ratio)


def is_field_like(raw: 'RawBBox', ctx: 'GeometryContext', image=None) -> bool:
    """
    Check if bbox looks like a typical input field.
    
    Field-like means:
    - has_border (required for visual distinction)
    - height within 0.5x–1.5x median_input_height
    - area <= 2.5 * median_input_area
    - aspect >= 2.0 (wider than tall) — RELAXED from 2.5
    
    Note: padding check removed as it was too strict.
    """
    bbox = raw.bbox
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    if w <= 0 or h <= 0:
        return False
    
    T = RelativeThresholds
    aspect = w / h
    area = w * h
    rel_height = h / ctx.median_input_height
    rel_area = area / ctx.median_input_area if ctx.median_input_area > 0 else 0
    
    # 1. Must have border (required to distinguish from empty space)
    if not raw.has_border:
        return False
    
    # 2. Height within range (0.5x–1.5x median)
    if not (T.INPUT_HEIGHT_MIN_RATIO <= rel_height <= T.INPUT_HEIGHT_MAX_RATIO):
        return False
    
    # 3. Area not too large
    if rel_area > T.INPUT_MAX_AREA_RATIO:
        return False
    
    # 4. Aspect ratio >= 2.0 (RELAXED from 2.5)
    if aspect < 2.0:
        return False
    
    # Padding check removed — was too strict
    return True


def is_label_like(raw: 'RawBBox', ctx: 'GeometryContext') -> bool:
    """
    Check if bbox looks like a label (text without border/fill).
    
    STRICT LABEL CRITERIA (fallback only):
    - no border (already checked by caller via can_be_label)
    - OCR text exists
    - height <= 0.7 * median_input_height (small)
    - aspect >= 2.0 (wider than tall)
    
    NOTE: has_border and has_fill are checked by caller (can_be_label flag).
    """
    bbox = raw.bbox
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    if w <= 0 or h <= 0:
        return False
    
    T = RelativeThresholds
    aspect = w / h
    rel_height = h / ctx.median_input_height
    
    # 1. No border (redundant but safe)
    if raw.has_border:
        return False
    
    # 2. Has text (required for label)
    if not raw.contains_text:
        return False
    
    # 3. Short height (labels are small)
    if rel_height > T.LABEL_HEIGHT_MAX_RATIO:
        return False
    
    # 4. Wider than tall (aspect >= 2)
    if aspect < T.LABEL_ASPECT_MIN:
        return False
    
    return True


def count_field_like_children(
    raw: 'RawBBox',
    all_raw: List['RawBBox'],
    ctx: 'GeometryContext',
) -> int:
    """Count how many field-like children this raw bbox has."""
    count = 0
    for child_idx in raw.children:
        if 0 <= child_idx < len(all_raw):
            child = all_raw[child_idx]
            if is_field_like(child, ctx):
                count += 1
    return count


def compute_ocr_area_ratio(bbox: List[float], ocr_blocks: List[Dict[str, Any]]) -> float:
    """Compute what fraction of bbox is covered by OCR text."""
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if bbox_area <= 0:
        return 0.0
    
    ocr_area_inside = 0.0
    for ocr in ocr_blocks:
        ob = ocr.get('bbox', [0, 0, 0, 0])
        if len(ob) < 4:
            continue
        
        # Compute intersection
        inter_x1 = max(bbox[0], ob[0])
        inter_y1 = max(bbox[1], ob[1])
        inter_x2 = min(bbox[2], ob[2])
        inter_y2 = min(bbox[3], ob[3])
        
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            ocr_area_inside += (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    return ocr_area_inside / bbox_area


def is_button_like(raw: 'RawBBox', ctx: 'GeometryContext', image) -> bool:
    """
    Check if bbox looks like a button (ACTION).
    
    Button-like means:
    - has_border OR has_fill
    - height within 0.8–1.6 * median_input_height
    - aspect >= 1.5 (wider than tall)
    - NOT too wide (aspect < 6 to avoid input confusion)
    """
    bbox = raw.bbox
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    if w <= 0 or h <= 0:
        return False
    
    T = RelativeThresholds
    aspect = w / h
    rel_height = h / ctx.median_input_height
    
    # 1. Has border OR fill
    has_fill = has_background_fill(image, bbox) if image is not None else False
    if not raw.has_border and not has_fill:
        return False
    
    # 2. Height within button range (0.8–1.6 * median)
    if not (0.8 <= rel_height <= 1.6):
        return False
    
    # 3. Aspect ratio: wider than tall, but not too wide
    if aspect < 1.5 or aspect >= 6.0:
        return False
    
    return True


def classify_raw_bbox(
    raw: RawBBox,
    idx: int,
    all_raw: List[RawBBox],
    ctx: GeometryContext,
    image,
) -> Tuple[str, float]:
    """
    Classify raw bbox into element type.
    
    CRITICAL ARCHITECTURE:
    - GEOMETRY determines type
    - CONTAINER checked EARLY to avoid large blocks becoming TEXTAREA
    - has_border OR has_fill → NEVER LABEL
    - ACTION does NOT depend on color
    
    Classification priority:
    1. CHECKBOX/RADIO — small, square (0.8-1.2 aspect)
    2. CONTAINER — large (area >= 2.5x median), has children
    3. INPUT — wide (aspect >= 3), border, height 0.6-1.5x
    4. ACTION/BUTTON — aspect 1.5-4, border/fill, height 0.6-1.5x
    5. TEXTAREA — tall (>= 2.5x), border, aspect < 3
    6. DECORATION
    7. LABEL — no border, no fill, text
    8. UNKNOWN
    """
    T = RelativeThresholds
    bbox = raw.bbox
    
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    if w <= 0 or h <= 0:
        return ElementTypes.UNKNOWN, 0.0
    
    aspect = w / h
    area = w * h
    diagonal = compute_container_diagonal(ctx)
    
    rel_width = w / ctx.container_width
    rel_height = h / ctx.median_input_height
    rel_size = (w + h) / 2 / diagonal
    rel_area = area / ctx.median_input_area if ctx.median_input_area > 0 else 0
    
    # Pre-compute fill for reuse
    has_fill = has_background_fill(image, bbox)
    
    # #region agent log
    import json, time
    log_path = "/app/debug/debug.log"
    log_data = {
        "timestamp": int(time.time() * 1000),
        "location": "visual_geometry_extractor.py:classify_raw_bbox",
        "hypothesisId": "H1-classification",
        "message": "classify_raw_bbox_input",
        "data": {
            "idx": idx,
            "bbox": [float(x) for x in bbox],
            "w": float(w),
            "h": float(h),
            "aspect": round(float(aspect), 2),
            "rel_width": round(float(rel_width), 3),
            "rel_height": round(float(rel_height), 3),
            "rel_area": round(float(rel_area), 3),
            "rel_size": round(float(rel_size), 4),
            "has_border": bool(raw.has_border),
            "has_fill": bool(has_fill),
            "has_children": bool(raw.children),
            "contains_text": bool(raw.contains_text),
            "median_input_height": float(ctx.median_input_height),
            "median_input_area": float(ctx.median_input_area),
        }
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_data) + "\n")
    # #endregion
    
    # ===========================================
    # RULE: ONLY border determines if element can be LABEL
    # has_fill is NOT reliable — page background gives false positives
    # If no border → likely OCR text → LABEL candidate
    # ===========================================
    can_be_label = not raw.has_border
    
    # ===========================================
    # 1. CHECKBOX/RADIO — small square elements
    # Aspect 0.8-1.2, small relative size
    # STRICT: must have visible border (not just contrast!)
    # ===========================================
    if (T.CHECKBOX_ASPECT_MIN <= aspect <= T.CHECKBOX_ASPECT_MAX and
        T.CHECKBOX_SIZE_MIN_RATIO <= rel_size <= T.CHECKBOX_SIZE_MAX_RATIO):
        # STRICT: require actual border, not just contrast
        # This prevents letters from being classified as checkboxes
        if raw.has_border:
            return ElementTypes.CHECKBOX, 0.85
    
    # ===========================================
    # 2. CONTAINER — EARLY CHECK!
    # Large area (>= 2.5x median) AND has children
    # Prevents large blocks from becoming TEXTAREA
    # ===========================================
    if rel_area >= 2.5 and raw.children:
        return ElementTypes.CONTAINER, 0.70
    
    # Also check very large elements without explicit children
    if rel_area >= 4.0 and rel_height >= 2.0:
        # Very large — likely container or decoration, not field
        if raw.has_border:
            return ElementTypes.DECORATION, 0.50
    
    # ===========================================
    # 2c. LABEL — small text element (not a real field)
    # Rule: OCR without real border = just text
    # Small elements from color_segmentation are likely text (letters create edges)
    # Criteria: small area + small width (not a wide input field)
    # ===========================================
    is_small_element = rel_area < 1.5  # small area relative to median
    is_narrow = rel_width < 0.15  # narrow (< 15% of container width)
    is_from_color_seg = raw.source == "color_segmentation"
    
    if is_small_element and is_narrow and is_from_color_seg and raw.contains_text:
        # Small narrow text — likely label or navigation text, not input field
        # Should be LABEL (will be filtered if not bound to INPUT)
        return ElementTypes.LABEL, 0.60
    
    # ===========================================
    # 3. INPUT — wide field with border
    # aspect >= 2.5, border, typical height (0.5-1.8x)
    # SKIP: small narrow elements (likely text)
    # ===========================================
    if (aspect >= 2.5 and
        raw.has_border and
        0.5 <= rel_height <= 1.8 and
        not (is_small_element and is_narrow)):  # exclude small text
        return ElementTypes.INPUT, 0.80
    
    # ===========================================
    # 3b. INPUT — wide field WITHOUT border (geometry-based)
    # For input fields where border detection failed (light borders)
    # STRICT: aspect >= 5 (very wide), from edge_detection, typical height
    # This catches input text/placeholder that was detected instead of border
    # ===========================================
    if (aspect >= 5.0 and
        not raw.has_border and
        raw.source == "edge_detection" and
        0.5 <= rel_height <= 2.5 and
        rel_width >= 0.2):  # at least 20% of container width
        return ElementTypes.INPUT, 0.65  # lower confidence
    
    # ===========================================
    # 4. ACTION/BUTTON — has border OR has fill with proper geometry
    # aspect 1.3-8, height 0.5-2.0x
    # NOTE: allow has_fill for buttons that are filled (no border)
    # SKIP: small narrow elements (likely text)
    # ===========================================
    if (raw.has_border and
        1.3 <= aspect <= 8.0 and
        0.5 <= rel_height <= 2.0 and
        not (is_small_element and is_narrow)):  # exclude small text
        return ElementTypes.ACTION, 0.75
    
    # ===========================================
    # 4b. ACTION/BUTTON — filled button without border
    # For filled/colored buttons where border detection failed
    # STRICT: has_fill, aspect 2-5, height 1.0-2.5x, contains_text
    # ===========================================
    if (has_fill and
        not raw.has_border and
        raw.contains_text and
        2.0 <= aspect <= 5.0 and
        1.0 <= rel_height <= 2.5):
        return ElementTypes.ACTION, 0.70
    
    # ===========================================
    # 5. TEXTAREA — tall element
    # height >= 2.0x median, border
    # Removed aspect restriction — tall with border = textarea
    # ===========================================
    if (rel_height >= 2.0 and raw.has_border):
        return ElementTypes.TEXTAREA, 0.70
    
    # ===========================================
    # 6. DECORATION — very large (>= 70% container width), MUST have border
    # Only truly large decorative elements
    # ===========================================
    if raw.has_border and rel_width >= 0.7 and rel_height >= 1.5:
        return ElementTypes.DECORATION, 0.40
    
    # ===========================================
    # 7. LABEL — FALLBACK ONLY
    # no border, no fill, has text
    # ===========================================
    if can_be_label and raw.contains_text:
        return ElementTypes.LABEL, 0.50
    
    # ===========================================
    # 8. UNKNOWN fallback
    # ===========================================
    # #region agent log
    result_type = ElementTypes.UNKNOWN
    log_result = {
        "timestamp": int(time.time() * 1000),
        "location": "visual_geometry_extractor.py:classify_result",
        "hypothesisId": "H2-result",
        "message": "classify_result",
        "data": {
            "idx": idx,
            "result": result_type,
            "reason": "fallback",
            "aspect": round(float(aspect), 2),
            "rel_height": round(float(rel_height), 3),
            "has_border": bool(raw.has_border),
            "has_fill": bool(has_fill),
            "can_be_label": bool(can_be_label),
            "contains_text": bool(raw.contains_text),
        }
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_result) + "\n")
    # #endregion
    return ElementTypes.UNKNOWN, 0.20


# =============================================================================
# RAW BBOX DETECTION
# =============================================================================

def detect_raw_bboxes(
    image,
    ctx: GeometryContext,
) -> List[RawBBox]:
    """
    Detect all raw bboxes without classification.
    """
    import cv2
    import numpy as np
    
    bbox = ctx.container_bbox
    if len(bbox) < 4:
        return []
    
    img_h, img_w = image.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(img_w, int(bbox[2]))
    y2 = min(img_h, int(bbox[3]))
    
    if x2 <= x1 or y2 <= y1:
        return []
    
    crop = image[y1:y2, x1:x2]
    
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    results: List[RawBBox] = []
    seen_bboxes: List[List[float]] = []
    
    def is_duplicate(new_bbox, threshold=0.5):
        for existing in seen_bboxes:
            if compute_iou(new_bbox, existing) > threshold:
                return True
        return False
    
    # 1. Edge detection (inputs, textareas, sections)
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
    
    contours_edges, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours_edges:
        cx, cy, cw, ch = cv2.boundingRect(c)
        
        rel_width = cw / ctx.container_width
        rel_height = ch / ctx.median_input_height
        
        if rel_width < 0.05 or rel_height < 0.3:
            continue
        
        # Rectangularity check
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) < 4:
            continue
        
        element_bbox = [float(x1 + cx), float(y1 + cy), float(x1 + cx + cw), float(y1 + cy + ch)]
        
        if is_duplicate(element_bbox):
            continue
        
        # OCR overlap check
        should_reject, contains_text = check_ocr_overlap(element_bbox, ctx.ocr_blocks, image)
        if should_reject:
            continue
        
        has_border = has_visible_border(image, element_bbox)
        
        results.append(RawBBox(
            bbox=element_bbox,
            source="edge_detection",
            has_border=has_border,
            contains_text=contains_text,
        ))
        seen_bboxes.append(element_bbox)
    
    # 2. Color segmentation (buttons, icons)
    if len(crop.shape) == 3:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        _, color_mask = cv2.threshold(saturation, 25, 255, cv2.THRESH_BINARY)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        color_closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_small)
        contours_color, _ = cv2.findContours(color_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours_color:
            cx, cy, cw, ch = cv2.boundingRect(c)
            
            rel_width = cw / ctx.container_width
            rel_height = ch / ctx.median_input_height
            
            if rel_width < 0.05 or rel_height < 0.3:
                continue
            
            element_bbox = [float(x1 + cx), float(y1 + cy), float(x1 + cx + cw), float(y1 + cy + ch)]
            
            if is_duplicate(element_bbox, threshold=0.4):
                continue
            
            should_reject, contains_text = check_ocr_overlap(element_bbox, ctx.ocr_blocks, image)
            if should_reject:
                continue
            
            has_border = has_visible_border(image, element_bbox) or has_background_fill(image, element_bbox)
            
            results.append(RawBBox(
                bbox=element_bbox,
                source="color_segmentation",
                has_border=has_border,
                contains_text=contains_text,
            ))
            seen_bboxes.append(element_bbox)
    
    # 3. Checkbox-specific detection (small squares)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_binary, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    diagonal = compute_container_diagonal(ctx)
    T = RelativeThresholds
    
    for c in contours_binary:
        cx, cy, cw, ch = cv2.boundingRect(c)
        
        rel_size = ((cw + ch) / 2) / diagonal
        if not (T.CHECKBOX_SIZE_MIN_RATIO <= rel_size <= T.CHECKBOX_SIZE_MAX_RATIO):
            continue
        
        aspect = cw / max(1, ch)
        if not (T.CHECKBOX_ASPECT_MIN <= aspect <= T.CHECKBOX_ASPECT_MAX):
            continue
        
        # Fill ratio check
        area = cv2.contourArea(c)
        rect_area = cw * ch
        fill_ratio = area / max(1, rect_area)
        if fill_ratio > 0.75:
            continue  # Likely a letter
        
        element_bbox = [float(x1 + cx), float(y1 + cy), float(x1 + cx + cw), float(y1 + cy + ch)]
        
        if is_duplicate(element_bbox, threshold=0.3):
            continue
        
        results.append(RawBBox(
            bbox=element_bbox,
            source="checkbox_detection",
            has_border=True,
            contains_text=False,
        ))
        seen_bboxes.append(element_bbox)
    
    return results


# =============================================================================
# SYMMETRY RECOVERY (BEFORE NMS!)
# =============================================================================

def recover_checkbox_symmetry(
    raw_bboxes: List[RawBBox],
    image,
    ctx: GeometryContext,
) -> List[RawBBox]:
    """
    Checkbox symmetry recovery — BEFORE NMS!
    
    Algorithm:
    1. Find all square bboxes (checkbox candidates)
    2. Group by vertical position (same row)
    3. If single checkbox in row, search for paired checkbox nearby
    """
    import cv2
    import numpy as np
    
    T = RelativeThresholds
    diagonal = compute_container_diagonal(ctx)
    
    # Find checkbox candidates
    checkbox_candidates = []
    for i, raw in enumerate(raw_bboxes):
        bbox = raw.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        if w <= 0 or h <= 0:
            continue
        
        rel_size = ((w + h) / 2) / diagonal
        aspect = w / h
        
        if (T.CHECKBOX_SIZE_MIN_RATIO <= rel_size <= T.CHECKBOX_SIZE_MAX_RATIO and
            T.CHECKBOX_ASPECT_MIN <= aspect <= T.CHECKBOX_ASPECT_MAX):
            checkbox_candidates.append((i, raw))
    
    if not checkbox_candidates:
        return raw_bboxes
    
    # Compute median checkbox size
    sizes = [(raw.bbox[2] - raw.bbox[0] + raw.bbox[3] - raw.bbox[1]) / 2 
             for _, raw in checkbox_candidates]
    median_cb_size = sorted(sizes)[len(sizes) // 2] if sizes else 16
    ctx.median_checkbox_size = median_cb_size
    
    # Group by Y position
    y_tolerance = median_cb_size * T.CHECKBOX_VERTICAL_TOLERANCE
    
    clusters: List[List[Tuple[int, RawBBox]]] = []
    used: Set[int] = set()
    
    for idx, raw in checkbox_candidates:
        if idx in used:
            continue
        
        cluster = [(idx, raw)]
        used.add(idx)
        
        cy = (raw.bbox[1] + raw.bbox[3]) / 2
        
        for other_idx, other_raw in checkbox_candidates:
            if other_idx in used:
                continue
            other_cy = (other_raw.bbox[1] + other_raw.bbox[3]) / 2
            if abs(cy - other_cy) <= y_tolerance:
                cluster.append((other_idx, other_raw))
                used.add(other_idx)
        
        clusters.append(cluster)
    
    # For single-element clusters, search for paired checkbox
    recovered: List[RawBBox] = []
    
    for cluster in clusters:
        if len(cluster) != 1:
            continue
        
        _, cb_raw = cluster[0]
        cb_bbox = cb_raw.bbox
        cb_w = cb_bbox[2] - cb_bbox[0]
        cb_h = cb_bbox[3] - cb_bbox[1]
        cb_cy = (cb_bbox[1] + cb_bbox[3]) / 2
        
        # Search distance
        search_distance = cb_w * T.CHECKBOX_SEARCH_DISTANCE
        
        # Search to the right
        search_x1 = cb_bbox[2] + cb_w * 0.5
        search_x2 = min(ctx.container_bbox[2], cb_bbox[2] + search_distance)
        search_y1 = cb_cy - cb_h
        search_y2 = cb_cy + cb_h
        
        if search_x2 <= search_x1:
            continue
        
        img_h, img_w = image.shape[:2]
        sx1 = max(0, int(search_x1))
        sy1 = max(0, int(search_y1))
        sx2 = min(img_w, int(search_x2))
        sy2 = min(img_h, int(search_y2))
        
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        
        search_roi = image[sy1:sy2, sx1:sx2]
        if search_roi.size == 0:
            continue
        
        if len(search_roi.shape) == 3:
            gray = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = search_roi
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            cx, cy_c, cw, ch = cv2.boundingRect(c)
            
            # Size similarity
            size_diff_w = abs(cw - cb_w) / cb_w
            size_diff_h = abs(ch - cb_h) / cb_h
            
            if size_diff_w > T.CHECKBOX_SIZE_TOLERANCE or size_diff_h > T.CHECKBOX_SIZE_TOLERANCE:
                continue
            
            aspect = cw / max(1, ch)
            if not (T.CHECKBOX_ASPECT_MIN <= aspect <= T.CHECKBOX_ASPECT_MAX):
                continue
            
            candidate_bbox = [float(sx1 + cx), float(sy1 + cy_c), 
                            float(sx1 + cx + cw), float(sy1 + cy_c + ch)]
            
            # Check not already detected
            is_dup = False
            for raw in raw_bboxes:
                if compute_iou(candidate_bbox, raw.bbox) > 0.3:
                    is_dup = True
                    break
            for rec in recovered:
                if compute_iou(candidate_bbox, rec.bbox) > 0.3:
                    is_dup = True
                    break
            
            if not is_dup:
                recovered.append(RawBBox(
                    bbox=candidate_bbox,
                    source="symmetry_recovery",
                    has_border=True,
                    contains_text=False,
                ))
                break  # one recovery per cluster
    
    return raw_bboxes + recovered


# =============================================================================
# TYPE-AWARE NMS
# =============================================================================

def apply_type_aware_nms(elements: List[VisualElement]) -> List[VisualElement]:
    """
    Apply type-aware NMS — ONE TIME ONLY in S1.
    
    Rules:
    - CONTAINER does not suppress CHECKBOX, RADIO, INPUT, ACTION
    - Smaller elements (checkbox/radio) have priority
    - Apply NMS separately per type group
    """
    if not elements:
        return []
    
    T = RelativeThresholds
    
    # Group by type priority
    small_elements = [e for e in elements if e.element_type in (ElementTypes.CHECKBOX, ElementTypes.RADIO)]
    action_elements = [e for e in elements if e.element_type == ElementTypes.ACTION]
    input_elements = [e for e in elements if e.element_type in (ElementTypes.INPUT, ElementTypes.TEXTAREA)]
    container_elements = [e for e in elements if e.element_type == ElementTypes.CONTAINER]
    other_elements = [e for e in elements if e.element_type not in 
                     (ElementTypes.CHECKBOX, ElementTypes.RADIO, ElementTypes.ACTION,
                      ElementTypes.INPUT, ElementTypes.TEXTAREA, ElementTypes.CONTAINER)]
    
    def nms_within_group(group: List[VisualElement], iou_threshold: float) -> List[VisualElement]:
        """Apply NMS within a group."""
        if not group:
            return []
        
        # Sort by confidence descending
        sorted_group = sorted(group, key=lambda e: -e.confidence)
        keep = []
        
        for elem in sorted_group:
            should_keep = True
            for kept in keep:
                if compute_iou(elem.bbox, kept.bbox) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(elem)
        
        return keep
    
    # Apply NMS per group
    kept_small = nms_within_group(small_elements, T.NMS_IOU_THRESHOLD_SMALL)
    kept_action = nms_within_group(action_elements, T.NMS_IOU_THRESHOLD_SMALL)
    kept_input = nms_within_group(input_elements, T.NMS_IOU_THRESHOLD)
    kept_container = nms_within_group(container_elements, T.NMS_IOU_THRESHOLD)
    kept_other = nms_within_group(other_elements, T.NMS_IOU_THRESHOLD)
    
    # Combine all kept elements
    # Small elements have highest priority — they suppress larger elements that overlap
    result: List[VisualElement] = []
    
    # Add small elements first (they don't get suppressed)
    result.extend(kept_small)
    
    # Add action elements (check against small)
    for elem in kept_action:
        suppressed = False
        for small in kept_small:
            if compute_iou(elem.bbox, small.bbox) > T.NMS_IOU_THRESHOLD:
                suppressed = True
                break
        if not suppressed:
            result.append(elem)
    
    # Add input elements (check against small and action)
    for elem in kept_input:
        suppressed = False
        for existing in result:
            if existing.element_type in (ElementTypes.CHECKBOX, ElementTypes.RADIO, ElementTypes.ACTION):
                if compute_iou(elem.bbox, existing.bbox) > T.NMS_IOU_THRESHOLD:
                    suppressed = True
                    break
        if not suppressed:
            result.append(elem)
    
    # Add containers (they don't suppress other types!)
    for elem in kept_container:
        # Container only suppressed by other containers with higher confidence
        suppressed = False
        for existing in result:
            if existing.element_type == ElementTypes.CONTAINER:
                if compute_iou(elem.bbox, existing.bbox) > T.NMS_IOU_THRESHOLD:
                    suppressed = True
                    break
        if not suppressed:
            result.append(elem)
    
    # Add other elements
    for elem in kept_other:
        suppressed = False
        for existing in result:
            if compute_iou(elem.bbox, existing.bbox) > T.NMS_IOU_THRESHOLD:
                suppressed = True
                break
        if not suppressed:
            result.append(elem)
    
    return result


# =============================================================================
# MEDIAN ESTIMATION
# =============================================================================

def estimate_medians(elements: List[VisualElement], ctx: GeometryContext) -> None:
    """
    Estimate median input height, width, and area from detected elements.
    
    CRITICAL: Only use field-like INPUT elements!
    Exclude: CONTAINER, DECORATION, ACTION
    
    This prevents pollution of median values by large non-field elements.
    """
    input_heights = []
    input_widths = []
    input_areas = []
    
    # Excluded types that should not affect median calculation
    excluded_types = (
        ElementTypes.CONTAINER,
        ElementTypes.DECORATION,
        ElementTypes.ACTION,
        "container",
        "decoration", 
        "action",
    )
    
    for elem in elements:
        # Only INPUT elements (not ACTION, CONTAINER, DECORATION)
        if elem.element_type in excluded_types:
            continue
        
        # Only INPUT type
        if elem.element_type not in (ElementTypes.INPUT, "input"):
            continue
        
        h = elem.bbox[3] - elem.bbox[1]
        w = elem.bbox[2] - elem.bbox[0]
        
        if w <= 0 or h <= 0:
            continue
        
        aspect = w / h
        
        # Additional field-like checks for median candidates
        # - Reasonable height (2-20% of container)
        # - Reasonable width (10-90% of container)
        # - Aspect ratio >= 2.5 (wider than tall)
        # - Has border (already checked by is_field_like during classification)
        
        height_ratio = h / ctx.container_height
        width_ratio = w / ctx.container_width
        
        if not (0.02 < height_ratio < 0.2):
            continue
        if not (0.1 < width_ratio < 0.9):
            continue
        if aspect < 2.5:
            continue  # Not field-like (too square)
        
        input_heights.append(h)
        input_widths.append(w)
        
        area = w * h
        if area > 0:
            input_areas.append(area)
    
    if input_heights:
        sorted_h = sorted(input_heights)
        ctx.median_input_height = sorted_h[len(sorted_h) // 2]
    else:
        ctx.median_input_height = ctx.container_height * 0.06  # default ~6%
    
    if input_widths:
        sorted_w = sorted(input_widths)
        ctx.median_input_width = sorted_w[len(sorted_w) // 2]
    else:
        ctx.median_input_width = ctx.container_width * 0.5  # default 50%
    
    if input_areas:
        sorted_a = sorted(input_areas)
        ctx.median_input_area = sorted_a[len(sorted_a) // 2]
    else:
        ctx.median_input_area = ctx.median_input_width * ctx.median_input_height


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def extract_visual_geometry(
    image_path: str,
    container_bbox: List[float],
    ocr_blocks: Optional[List[Dict[str, Any]]] = None,
) -> S1Result:
    """
    S1 — Visual Geometry Extraction.
    
    ЕДИНСТВЕННЫЙ ЭТАП где разрешено находить bbox и классифицировать.
    После этого visual_elements IMMUTABLE.
    
    ПОРЯДОК:
    1. Найти все raw bbox
    2. Symmetry recovery (ДО NMS!)
    3. Определить parent-child отношения
    4. Классифицировать (с учётом вложенности)
    5. NMS по типам (type-aware)
    """
    import cv2
    
    diagnostics: Dict[str, Any] = {
        "detected_raw": 0,
        "after_symmetry_recovery": 0,
        "after_classification": 0,
        "after_nms": 0,
        "by_type": {},
        "containers_found": 0,
    }
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return S1Result(
            visual_elements=[],
            context=GeometryContext(container_bbox=container_bbox, container_width=1, container_height=1),
            diagnostics={"error": "could not read image"},
        )
    
    # Create context
    container_w = container_bbox[2] - container_bbox[0] if len(container_bbox) >= 4 else 1
    container_h = container_bbox[3] - container_bbox[1] if len(container_bbox) >= 4 else 1
    
    ctx = GeometryContext(
        container_bbox=container_bbox,
        container_width=container_w,
        container_height=container_h,
        median_input_height=35.0,  # temporary, will be estimated
        median_input_width=200.0,  # temporary
        median_input_area=7000.0,  # temporary
        ocr_blocks=ocr_blocks or [],
    )
    
    # =========================================================================
    # STEP 1: Detect all raw bboxes
    # =========================================================================
    raw_bboxes = detect_raw_bboxes(image, ctx)
    diagnostics["detected_raw"] = len(raw_bboxes)
    logger.debug(f"S1: detected {len(raw_bboxes)} raw bboxes")
    
    # =========================================================================
    # STEP 1.5: ESTIMATE MEDIAN FROM RAW BBOXES (BEFORE classification!)
    # Look for horizontal rectangles with border — likely inputs
    # =========================================================================
    candidate_heights = []
    candidate_widths = []
    candidate_areas = []
    
    for raw in raw_bboxes:
        bbox = raw.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        if w <= 0 or h <= 0:
            continue
        
        aspect = w / h
        
        # Candidate for median: horizontal (aspect >= 2.0), has border, reasonable size
        # Relaxed thresholds to catch smaller UI elements
        if (aspect >= 2.0 and 
            raw.has_border and
            0.008 < h / container_h < 0.2 and  # relaxed: 0.8% to 20% of container
            0.03 < w / container_w < 0.95):    # relaxed: 3% to 95% of container
            candidate_heights.append(h)
            candidate_widths.append(w)
            candidate_areas.append(w * h)
    
    if candidate_heights:
        candidate_heights.sort()
        candidate_widths.sort()
        candidate_areas.sort()
        ctx.median_input_height = candidate_heights[len(candidate_heights) // 2]
        ctx.median_input_width = candidate_widths[len(candidate_widths) // 2]
        ctx.median_input_area = candidate_areas[len(candidate_areas) // 2]
        logger.debug(f"S1: estimated median from {len(candidate_heights)} candidates: "
                    f"height={ctx.median_input_height:.1f}, width={ctx.median_input_width:.1f}")
    else:
        # Fallback: use container-relative estimates
        ctx.median_input_height = container_h * 0.025  # ~2.5% of container
        ctx.median_input_width = container_w * 0.4
        ctx.median_input_area = ctx.median_input_height * ctx.median_input_width
        logger.debug(f"S1: no candidates, using fallback median: "
                    f"height={ctx.median_input_height:.1f}")
    
    # #region agent log
    import json, time
    log_path = "/app/debug/debug.log"
    
    # Log all raw bboxes for analysis
    all_raw_info = []
    for raw in raw_bboxes:
        bbox = raw.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > 0 and h > 0:
            all_raw_info.append({
                "bbox": [float(x) for x in bbox],
                "w": float(w),
                "h": float(h),
                "aspect": round(float(w/h), 2),
                "h_ratio": round(float(h / container_h), 3),
                "w_ratio": round(float(w / container_w), 3),
                "has_border": bool(raw.has_border),
                "source": raw.source,
            })
    
    raw_bboxes_log = {
        "timestamp": int(time.time() * 1000),
        "location": "visual_geometry_extractor.py:raw_bboxes",
        "hypothesisId": "H5-raw-bboxes",
        "message": "all_raw_bboxes",
        "data": {
            "total": len(raw_bboxes),
            "bboxes": all_raw_info[:20],  # first 20
        }
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(raw_bboxes_log) + "\n")
    
    median_log = {
        "timestamp": int(time.time() * 1000),
        "location": "visual_geometry_extractor.py:estimate_median",
        "hypothesisId": "H3-median",
        "message": "median_estimation",
        "data": {
            "num_candidates": len(candidate_heights),
            "median_input_height": float(ctx.median_input_height),
            "median_input_width": float(ctx.median_input_width),
            "median_input_area": float(ctx.median_input_area),
            "container_height": float(container_h),
            "container_width": float(container_w),
            "candidate_heights": [float(h) for h in candidate_heights[:10]] if candidate_heights else [],
        }
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(median_log) + "\n")
    # #endregion
    
    # =========================================================================
    # STEP 2: Symmetry recovery (BEFORE classification and NMS!)
    # =========================================================================
    raw_bboxes = recover_checkbox_symmetry(raw_bboxes, image, ctx)
    diagnostics["after_symmetry_recovery"] = len(raw_bboxes)
    logger.debug(f"S1: after symmetry recovery {len(raw_bboxes)} bboxes")
    
    # =========================================================================
    # STEP 3: Find parent-child relations
    # =========================================================================
    find_parent_child_relations(raw_bboxes)
    
    # =========================================================================
    # STEP 4: Classify all bboxes
    # =========================================================================
    classified: List[VisualElement] = []
    
    for idx, raw in enumerate(raw_bboxes):
        elem_type, confidence = classify_raw_bbox(raw, idx, raw_bboxes, ctx, image)
        
        bbox = raw.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        element = VisualElement(
            bbox=bbox,
            element_type=elem_type,
            confidence=confidence,
            source=raw.source,
            has_border=raw.has_border,
            contains_text=raw.contains_text,
            is_container=(elem_type == ElementTypes.CONTAINER),
            relative_width=w / ctx.container_width,
            relative_height=h / ctx.median_input_height,
            aspect_ratio=w / max(1, h),
            parent_id=raw.parent,
            child_ids=raw.children,
        )
        
        # Determine if checkbox is checked
        if elem_type in (ElementTypes.CHECKBOX, ElementTypes.RADIO):
            element.is_checked = detect_checkbox_state(image, bbox)
        
        classified.append(element)
    
    # #region agent log — log ALL classification results
    import json, time
    log_path = "/app/debug/debug.log"
    all_results = []
    for i, elem in enumerate(classified):
        all_results.append({
            "idx": i,
            "type": elem.element_type,
            "bbox": [float(x) for x in elem.bbox],
            "has_border": bool(elem.has_border),
            "aspect": round(float(elem.aspect_ratio), 2),
            "rel_height": round(float(elem.relative_height), 3),
        })
    
    results_log = {
        "timestamp": int(time.time() * 1000),
        "location": "visual_geometry_extractor.py:classification_results",
        "hypothesisId": "H6-all-results",
        "message": "all_classification_results",
        "data": {"results": all_results}
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(results_log) + "\n")
    # #endregion
    
    diagnostics["after_classification"] = len(classified)
    
    # =========================================================================
    # STEP 5: Estimate medians with classified elements
    # =========================================================================
    estimate_medians(classified, ctx)
    
    # Update relative heights with better median
    for elem in classified:
        elem.relative_height = (elem.bbox[3] - elem.bbox[1]) / ctx.median_input_height
    
    # =========================================================================
    # STEP 6: Type-aware NMS
    # =========================================================================
    final_elements = apply_type_aware_nms(classified)
    diagnostics["after_nms"] = len(final_elements)
    logger.debug(f"S1: after NMS {len(final_elements)} elements")
    
    # Count by type
    for elem in final_elements:
        t = elem.element_type
        diagnostics["by_type"][t] = diagnostics["by_type"].get(t, 0) + 1
        if t == ElementTypes.CONTAINER:
            diagnostics["containers_found"] += 1
    
    logger.info(f"S1 completed: {len(final_elements)} elements, types={diagnostics['by_type']}")
    
    return S1Result(
        visual_elements=final_elements,
        context=ctx,
        diagnostics=diagnostics,
    )


def detect_checkbox_state(image, bbox: List[float]) -> bool:
    """Detect if checkbox is checked."""
    import cv2
    import numpy as np
    
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = image.shape[:2]
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return False
    
    roi = image[y1:y2, x1:x2]
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # Compare inner vs outer region
    roi_h, roi_w = gray.shape[:2]
    margin = max(2, min(roi_h, roi_w) // 4)
    
    if roi_h > 2 * margin and roi_w > 2 * margin:
        inner = gray[margin:-margin, margin:-margin]
        inner_mean = inner.mean()
    else:
        inner_mean = gray.mean()
    
    outer_mean = gray.mean()
    
    # If inner is significantly different from outer, it's checked
    return abs(inner_mean - outer_mean) > 25
