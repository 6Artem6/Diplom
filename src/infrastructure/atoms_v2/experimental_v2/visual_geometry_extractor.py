"""
S1 — Visual Geometry Extractor (State Machine Architecture)

ЕДИНСТВЕННЫЙ ЭТАП где разрешено:
- находить bbox
- классифицировать visual type  
- делать NMS

После S1 visual_elements IMMUTABLE.

Инварианты:
- Никаких абсолютных размеров (только relative_to_container, relative_to_median, aspect_ratio)
- OCR overlap check — не детектить bbox поверх OCR-блоков
- Textarea containment — textarea не может содержать другие bbox
- Checkbox symmetry recovery — поиск парных checkbox
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS (Immutable after S1)
# =============================================================================

@dataclass
class VisualElement:
    """Immutable visual element detected in S1."""
    bbox: List[float]  # [x1, y1, x2, y2]
    element_type: str  # input, textarea, button, checkbox, radio, container, label
    confidence: float
    source: str  # detection method
    
    # Optional fields
    is_checked: Optional[bool] = None  # для checkbox/radio
    has_border: bool = False
    is_container: bool = False  # содержит другие элементы
    
    # Relative metrics (computed)
    relative_width: float = 0.0  # width / container_width
    relative_height: float = 0.0  # height / median_input_height
    aspect_ratio: float = 0.0


@dataclass
class GeometryContext:
    """Context for relative calculations."""
    container_bbox: List[float]
    container_width: float
    container_height: float
    median_input_height: float = 35.0  # будет вычислен
    median_input_width: float = 200.0  # будет вычислен
    
    # OCR blocks for overlap check
    ocr_blocks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class S1Result:
    """Result of S1 — Visual Geometry Extraction."""
    visual_elements: List[VisualElement]
    context: GeometryContext
    diagnostics: Dict[str, Any]


# =============================================================================
# RELATIVE THRESHOLDS (no absolute px values!)
# =============================================================================

class RelativeThresholds:
    """All thresholds are relative to container or median."""
    
    # Checkbox/Radio (relative to container diagonal)
    CHECKBOX_SIZE_MIN_RATIO = 0.015  # min 1.5% of container diagonal
    CHECKBOX_SIZE_MAX_RATIO = 0.06   # max 6% of container diagonal
    CHECKBOX_ASPECT_TOLERANCE = 0.35  # aspect must be ~1.0 ± 0.35
    
    # Input (relative to median_input_height)
    INPUT_HEIGHT_MIN_RATIO = 0.7   # min 70% of median
    INPUT_HEIGHT_MAX_RATIO = 1.5   # max 150% of median
    INPUT_ASPECT_MIN = 2.5         # min width/height ratio
    
    # Textarea (relative to median_input_height)
    TEXTAREA_HEIGHT_MIN_RATIO = 1.8  # min 180% of median
    TEXTAREA_ASPECT_MAX = 4.0        # max width/height ratio
    
    # Button (relative to container)
    BUTTON_WIDTH_MIN_RATIO = 0.1   # min 10% of container width
    BUTTON_WIDTH_MAX_RATIO = 0.6   # max 60% of container width
    BUTTON_HEIGHT_MIN_RATIO = 0.7  # min 70% of median_input_height
    BUTTON_HEIGHT_MAX_RATIO = 2.0  # max 200% of median_input_height
    
    # Container/Section
    CONTAINER_MIN_AREA_RATIO = 0.1  # min 10% of container area
    
    # OCR overlap threshold
    OCR_OVERLAP_REJECT_RATIO = 0.7  # reject if >70% overlap with OCR
    
    # NMS
    NMS_IOU_THRESHOLD = 0.5
    NMS_IOU_THRESHOLD_SMALL = 0.3  # stricter for checkbox/radio/button


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def compute_container_diagonal(ctx: GeometryContext) -> float:
    """Compute container diagonal for relative sizing."""
    return (ctx.container_width ** 2 + ctx.container_height ** 2) ** 0.5


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
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
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
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    return inter / max(1, area1 + area2 - inter)


def has_visible_border(image, bbox: List[float], threshold: float = 0.15) -> bool:
    """Check if bbox has visible border (edge density on perimeter)."""
    import cv2
    import numpy as np
    
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = image.shape[:2]
    
    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    # Extract ROI
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    
    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Check edge density on perimeter (top, bottom, left, right strips)
    strip_size = max(2, min(5, (x2 - x1) // 10, (y2 - y1) // 10))
    
    perimeter_edges = 0
    perimeter_pixels = 0
    
    # Top strip
    top = edges[:strip_size, :]
    perimeter_edges += np.sum(top > 0)
    perimeter_pixels += top.size
    
    # Bottom strip
    bottom = edges[-strip_size:, :]
    perimeter_edges += np.sum(bottom > 0)
    perimeter_pixels += bottom.size
    
    # Left strip
    left = edges[strip_size:-strip_size, :strip_size] if edges.shape[0] > 2 * strip_size else edges[:, :strip_size]
    perimeter_edges += np.sum(left > 0)
    perimeter_pixels += left.size
    
    # Right strip
    right = edges[strip_size:-strip_size, -strip_size:] if edges.shape[0] > 2 * strip_size else edges[:, -strip_size:]
    perimeter_edges += np.sum(right > 0)
    perimeter_pixels += right.size
    
    edge_density = perimeter_edges / max(1, perimeter_pixels)
    return edge_density >= threshold


def should_reject_ocr_overlap(
    bbox: List[float],
    ocr_blocks: List[Dict[str, Any]],
    image,
    threshold: float = RelativeThresholds.OCR_OVERLAP_REJECT_RATIO,
) -> bool:
    """
    Check if bbox should be rejected due to OCR overlap.
    
    Reject if:
    - >70% overlaps with OCR block
    - AND no visible border contrast inside
    """
    for ocr in ocr_blocks:
        ocr_bbox = ocr.get("bbox", [])
        if len(ocr_bbox) < 4:
            continue
        
        overlap = compute_overlap_ratio(bbox, ocr_bbox)
        if overlap > threshold:
            # Check if has visible border
            if not has_visible_border(image, bbox):
                return True
    
    return False


def classify_element_relative(
    bbox: List[float],
    ctx: GeometryContext,
    image,
    is_colored: bool = False,
) -> Tuple[str, float]:
    """
    Classify element using ONLY relative metrics.
    
    Returns: (element_type, confidence)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    if w <= 0 or h <= 0:
        return "unknown", 0.0
    
    aspect = w / h
    diagonal = compute_container_diagonal(ctx)
    
    # Relative metrics
    rel_width = w / ctx.container_width
    rel_height = h / ctx.median_input_height
    rel_size = (w + h) / 2 / diagonal
    
    T = RelativeThresholds
    
    # 1. CHECKBOX/RADIO — small square elements
    if (T.CHECKBOX_SIZE_MIN_RATIO <= rel_size <= T.CHECKBOX_SIZE_MAX_RATIO and
        abs(aspect - 1.0) <= T.CHECKBOX_ASPECT_TOLERANCE):
        return "checkbox", 0.85
    
    # 2. BUTTON — colored, medium size
    if is_colored:
        if (T.BUTTON_WIDTH_MIN_RATIO <= rel_width <= T.BUTTON_WIDTH_MAX_RATIO and
            T.BUTTON_HEIGHT_MIN_RATIO <= rel_height <= T.BUTTON_HEIGHT_MAX_RATIO and
            1.5 <= aspect <= 6.0):
            return "button", 0.8
    
    # 3. TEXTAREA — tall elements
    if (rel_height >= T.TEXTAREA_HEIGHT_MIN_RATIO and
        aspect < T.TEXTAREA_ASPECT_MAX and
        rel_width >= 0.25):
        return "textarea", 0.75
    
    # 4. INPUT — horizontal, typical height
    if (T.INPUT_HEIGHT_MIN_RATIO <= rel_height <= T.INPUT_HEIGHT_MAX_RATIO and
        aspect >= T.INPUT_ASPECT_MIN):
        return "input", 0.75
    
    # 5. BUTTON fallback — with border, medium aspect
    if has_visible_border(image, bbox):
        if (1.5 <= aspect <= 5.0 and
            T.BUTTON_HEIGHT_MIN_RATIO <= rel_height <= T.BUTTON_HEIGHT_MAX_RATIO and
            rel_width <= 0.5):
            return "button", 0.5
    
    # 6. CONTAINER/SECTION — large areas
    area_ratio = (w * h) / (ctx.container_width * ctx.container_height)
    if area_ratio >= T.CONTAINER_MIN_AREA_RATIO and aspect < 3.0:
        return "container", 0.4
    
    # 7. LABEL — wide, short
    if aspect > 4.0 and rel_height < 0.8:
        return "label", 0.3
    
    return "element", 0.25


def detect_checkbox_radio(
    image,
    ctx: GeometryContext,
) -> List[VisualElement]:
    """
    Priority detection of checkbox/radio elements.
    Uses relative sizing.
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
    
    # Binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    diagonal = compute_container_diagonal(ctx)
    T = RelativeThresholds
    
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        
        # Relative size check
        rel_size = ((cw + ch) / 2) / diagonal
        if not (T.CHECKBOX_SIZE_MIN_RATIO <= rel_size <= T.CHECKBOX_SIZE_MAX_RATIO):
            continue
        
        # Aspect check
        aspect = cw / max(1, ch)
        if abs(aspect - 1.0) > T.CHECKBOX_ASPECT_TOLERANCE:
            continue
        
        # Fill ratio check (checkbox/radio not fully filled)
        area = cv2.contourArea(c)
        rect_area = cw * ch
        fill_ratio = area / max(1, rect_area)
        
        if fill_ratio > 0.7:
            continue  # Likely a letter
        
        # Determine type: checkbox (rectangular) or radio (circular)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / max(1, perimeter ** 2) if perimeter > 0 else 0
        
        elem_type = "radio" if circularity > 0.7 else "checkbox"
        
        # Check if checked (has content inside)
        roi = gray[cy:cy+ch, cx:cx+cw]
        inner_mean = roi[2:-2, 2:-2].mean() if roi.shape[0] > 4 and roi.shape[1] > 4 else roi.mean()
        outer_mean = roi.mean()
        is_checked = abs(inner_mean - outer_mean) > 30
        
        element_bbox = [float(x1 + cx), float(y1 + cy), float(x1 + cx + cw), float(y1 + cy + ch)]
        
        results.append(VisualElement(
            bbox=element_bbox,
            element_type=elem_type,
            confidence=0.8,
            source="checkbox_detection",
            is_checked=is_checked,
            has_border=True,
            relative_width=cw / ctx.container_width,
            relative_height=ch / ctx.median_input_height,
            aspect_ratio=aspect,
        ))
    
    return results


def detect_all_elements(
    image,
    ctx: GeometryContext,
) -> List[VisualElement]:
    """
    Detect all visual elements using multiple methods.
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
    crop_h, crop_w = crop.shape[:2]
    
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    results = []
    seen_bboxes = []
    
    def is_duplicate(new_bbox, threshold=0.5):
        for existing in seen_bboxes:
            if compute_iou(new_bbox, existing) > threshold:
                return True
        return False
    
    # 1. Color segmentation (buttons, icons)
    if len(crop.shape) == 3:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        _, color_mask = cv2.threshold(saturation, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        color_closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        contours_color, _ = cv2.findContours(color_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours_color:
            cx, cy, cw, ch = cv2.boundingRect(c)
            
            # Minimum size (relative)
            rel_width = cw / ctx.container_width
            rel_height = ch / ctx.median_input_height
            
            if rel_width < 0.05 or rel_height < 0.3:
                continue
            
            element_bbox = [float(x1 + cx), float(y1 + cy), float(x1 + cx + cw), float(y1 + cy + ch)]
            
            if is_duplicate(element_bbox):
                continue
            
            # OCR overlap check
            if should_reject_ocr_overlap(element_bbox, ctx.ocr_blocks, image):
                continue
            
            elem_type, conf = classify_element_relative(element_bbox, ctx, image, is_colored=True)
            
            results.append(VisualElement(
                bbox=element_bbox,
                element_type=elem_type,
                confidence=conf,
                source="color_segmentation",
                has_border=has_visible_border(image, element_bbox),
                relative_width=rel_width,
                relative_height=rel_height,
                aspect_ratio=cw / max(1, ch),
            ))
            seen_bboxes.append(element_bbox)
    
    # 2. Edge detection (inputs, textareas, sections)
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours_edges, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours_edges:
        cx, cy, cw, ch = cv2.boundingRect(c)
        
        # Minimum size (relative)
        rel_width = cw / ctx.container_width
        rel_height = ch / ctx.median_input_height
        
        if rel_width < 0.1 or rel_height < 0.5:
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
        if should_reject_ocr_overlap(element_bbox, ctx.ocr_blocks, image):
            continue
        
        elem_type, conf = classify_element_relative(element_bbox, ctx, image, is_colored=False)
        
        results.append(VisualElement(
            bbox=element_bbox,
            element_type=elem_type,
            confidence=conf,
            source="edge_detection",
            has_border=True,
            relative_width=rel_width,
            relative_height=rel_height,
            aspect_ratio=cw / max(1, ch),
        ))
        seen_bboxes.append(element_bbox)
    
    return results


def apply_nms(elements: List[VisualElement]) -> List[VisualElement]:
    """
    Apply NMS — ONE TIME ONLY in S1.
    
    Priority: smaller elements first (checkbox/radio/button), then by confidence.
    """
    if not elements:
        return []
    
    T = RelativeThresholds
    
    # Sort by: (element size, -confidence)
    def sort_key(e):
        area = (e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1])
        # Smaller elements first
        return (area, -e.confidence)
    
    sorted_elements = sorted(elements, key=sort_key)
    
    keep = []
    for elem in sorted_elements:
        should_keep = True
        
        for kept in keep:
            iou = compute_iou(elem.bbox, kept.bbox)
            
            # Stricter threshold for small elements
            threshold = T.NMS_IOU_THRESHOLD_SMALL if elem.element_type in ("checkbox", "radio", "button") else T.NMS_IOU_THRESHOLD
            
            if iou > threshold:
                should_keep = False
                break
        
        if should_keep:
            keep.append(elem)
    
    return keep


def check_textarea_containment(elements: List[VisualElement]) -> List[VisualElement]:
    """
    Check textarea containment invariant.
    
    Textarea cannot contain other bboxes — if it does, reclassify as container.
    """
    result = []
    
    for elem in elements:
        if elem.element_type != "textarea":
            result.append(elem)
            continue
        
        # Check if contains other elements
        contains_other = False
        for other in elements:
            if other is elem:
                continue
            if other.element_type in ("container", "section"):
                continue
            
            # Check containment
            overlap = compute_overlap_ratio(elem.bbox, other.bbox)
            if overlap > 0.8:  # other is mostly inside elem
                contains_other = True
                break
        
        if contains_other:
            # Reclassify as container
            elem_copy = VisualElement(
                bbox=elem.bbox,
                element_type="container",
                confidence=elem.confidence * 0.8,
                source=elem.source,
                has_border=elem.has_border,
                is_container=True,
                relative_width=elem.relative_width,
                relative_height=elem.relative_height,
                aspect_ratio=elem.aspect_ratio,
            )
            result.append(elem_copy)
        else:
            result.append(elem)
    
    return result


def recover_checkbox_symmetry(
    elements: List[VisualElement],
    image,
    ctx: GeometryContext,
) -> List[VisualElement]:
    """
    Checkbox symmetry recovery.
    
    If only one checkbox found in a row, search for paired checkbox nearby.
    """
    import cv2
    import numpy as np
    
    checkboxes = [e for e in elements if e.element_type in ("checkbox", "radio")]
    
    if not checkboxes:
        return elements
    
    # Cluster checkboxes by Y position (same row)
    y_tolerance = ctx.median_input_height * 0.3
    
    clusters = []
    used = set()
    
    for i, cb in enumerate(checkboxes):
        if i in used:
            continue
        cluster = [cb]
        used.add(i)
        
        cb_cy = (cb.bbox[1] + cb.bbox[3]) / 2
        
        for j, other in enumerate(checkboxes):
            if j in used:
                continue
            other_cy = (other.bbox[1] + other.bbox[3]) / 2
            if abs(cb_cy - other_cy) <= y_tolerance:
                cluster.append(other)
                used.add(j)
        
        clusters.append(cluster)
    
    # For single-element clusters, try to find paired checkbox
    recovered = []
    
    for cluster in clusters:
        if len(cluster) != 1:
            continue
        
        cb = cluster[0]
        cb_w = cb.bbox[2] - cb.bbox[0]
        cb_h = cb.bbox[3] - cb.bbox[1]
        cb_cy = (cb.bbox[1] + cb.bbox[3]) / 2
        
        # Search area: same row, within reasonable distance
        search_distance = ctx.median_input_width * 2
        
        # Search to the right of existing checkbox
        search_x1 = cb.bbox[2] + cb_w  # start after checkbox
        search_x2 = min(ctx.container_bbox[2], cb.bbox[2] + search_distance)
        search_y1 = cb_cy - cb_h
        search_y2 = cb_cy + cb_h
        
        if search_x2 <= search_x1:
            continue
        
        # Extract search region
        sx1 = max(0, int(search_x1))
        sy1 = max(0, int(search_y1))
        sx2 = min(image.shape[1], int(search_x2))
        sy2 = min(image.shape[0], int(search_y2))
        
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        
        search_roi = image[sy1:sy2, sx1:sx2]
        if search_roi.size == 0:
            continue
        
        if len(search_roi.shape) == 3:
            gray = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = search_roi
        
        # Look for square-ish regions of similar size
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            
            # Size similarity (±10%)
            if abs(cw - cb_w) > cb_w * 0.1:
                continue
            if abs(ch - cb_h) > cb_h * 0.1:
                continue
            
            # Aspect ratio ~1
            aspect = cw / max(1, ch)
            if abs(aspect - 1.0) > 0.35:
                continue
            
            # Found candidate
            candidate_bbox = [float(sx1 + cx), float(sy1 + cy), float(sx1 + cx + cw), float(sy1 + cy + ch)]
            
            # Check it's not already detected
            is_duplicate = False
            for elem in elements:
                if compute_iou(candidate_bbox, elem.bbox) > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                recovered.append(VisualElement(
                    bbox=candidate_bbox,
                    element_type=cb.element_type,  # same type as original
                    confidence=0.6,
                    source="symmetry_recovery",
                    is_checked=False,  # assume unchecked
                    has_border=True,
                    relative_width=cw / ctx.container_width,
                    relative_height=ch / ctx.median_input_height,
                    aspect_ratio=aspect,
                ))
                break  # one recovery per cluster
    
    return elements + recovered


def estimate_median_input_height(elements: List[VisualElement]) -> float:
    """Estimate median input height from detected elements."""
    input_heights = []
    
    for elem in elements:
        if elem.element_type in ("input", "button"):
            h = elem.bbox[3] - elem.bbox[1]
            if 15 < h < 100:  # reasonable range
                input_heights.append(h)
    
    if input_heights:
        sorted_h = sorted(input_heights)
        return sorted_h[len(sorted_h) // 2]
    
    return 35.0  # default


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
    
    Args:
        image_path: путь к изображению
        container_bbox: bbox контейнера формы [x1, y1, x2, y2]
        ocr_blocks: OCR-блоки для overlap check (опционально)
    
    Returns:
        S1Result с visual_elements и context
    """
    import cv2
    
    diagnostics: Dict[str, Any] = {
        "detected_raw": 0,
        "after_ocr_filter": 0,
        "after_nms": 0,
        "after_containment": 0,
        "recovered_symmetry": 0,
        "by_type": {},
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
        ocr_blocks=ocr_blocks or [],
    )
    
    # 1. Detect checkbox/radio (priority)
    checkboxes = detect_checkbox_radio(image, ctx)
    logger.debug(f"S1: detected {len(checkboxes)} checkbox/radio")
    
    # 2. Detect all other elements
    all_elements = detect_all_elements(image, ctx)
    logger.debug(f"S1: detected {len(all_elements)} other elements")
    
    # 3. Combine (checkbox/radio have priority)
    combined = checkboxes + [e for e in all_elements if e.element_type not in ("checkbox", "radio")]
    diagnostics["detected_raw"] = len(combined)
    
    # 4. Estimate median input height and update context
    ctx.median_input_height = estimate_median_input_height(combined)
    ctx.median_input_width = ctx.container_width * 0.6  # rough estimate
    
    # Re-compute relative heights with updated median
    for elem in combined:
        elem.relative_height = (elem.bbox[3] - elem.bbox[1]) / ctx.median_input_height
    
    diagnostics["after_ocr_filter"] = len(combined)  # OCR filter already applied in detection
    
    # 5. Apply NMS — ONE TIME
    after_nms = apply_nms(combined)
    diagnostics["after_nms"] = len(after_nms)
    logger.debug(f"S1: after NMS {len(after_nms)} elements")
    
    # 6. Textarea containment check
    after_containment = check_textarea_containment(after_nms)
    diagnostics["after_containment"] = len(after_containment)
    
    # 7. Checkbox symmetry recovery
    final_elements = recover_checkbox_symmetry(after_containment, image, ctx)
    diagnostics["recovered_symmetry"] = len(final_elements) - len(after_containment)
    
    # Count by type
    for elem in final_elements:
        t = elem.element_type
        diagnostics["by_type"][t] = diagnostics["by_type"].get(t, 0) + 1
    
    logger.info(f"S1 completed: {len(final_elements)} elements, types={diagnostics['by_type']}")
    
    return S1Result(
        visual_elements=final_elements,
        context=ctx,
        diagnostics=diagnostics,
    )
