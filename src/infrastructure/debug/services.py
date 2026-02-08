"""
Debug pipeline: layout → text detection → OCR → UI blocks.

Used by HTTP /debug/* endpoints. No global flags; all params passed explicitly.
Logs counts and drop reasons. Saves debug images to configurable output dir.

EXPERIMENTAL: CV/heuristic path does not guarantee correct UI detection. See docs/PIPELINE_STATUS.md.
DL-only path (use_dl_only=True): layout from LayoutParser only, full-page Paddle text, assign by overlap.
"""

from __future__ import annotations

import io
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# When set (e.g. Mac/arm64 under QEMU), PaddleOCR is not loaded; OCR returns [] and is logged as skipped.
DISABLE_PADDLEOCR = os.environ.get("DISABLE_PADDLEOCR", "0") == "1"
# Optional: when DISABLE_PADDLEOCR=1, call this URL (standalone PaddleOCR service) for OCR. Empty = skip.
PADDLE_OCR_SERVICE_URL = os.environ.get("PADDLE_OCR_SERVICE_URL", "").strip()

DEBUG_OUTPUT_DIR = os.environ.get("DEBUG_OUTPUT_DIR", "debug/output")
MAX_BLOCK_SCREEN_RATIO = 0.8  # block ≥ 80% screen → invalid

# Button vs container: region must not be huge vs text (else container, not button)
MAX_BUTTON_AREA_MULT = 10  # region_area ≤ text_area * this
# Badge vs button by semantics: badge = h_region/h_text ≤ 1.4, short text; not "text ≈ container"
BADGE_H_REGION_TEXT_RATIO = 1.4
BADGE_MAX_CHARS = 15
# Button valid by text: region_w ≥ text_w*1.1, region_h ≥ text_h*1.4 (with absolute fallback)
BUTTON_TEXT_W_RATIO = 1.1
BUTTON_TEXT_H_RATIO = 1.4


def _ensure_debug_dir() -> Path:
    p = Path(DEBUG_OUTPUT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_layout(image_path: str) -> List[Dict[str, Any]]:
    """
    Layout detection (regions only). No OCR.
    Returns list of {"x", "y", "w", "h", "type"}.
    Uses OpenCV region prepass (legacy). Prefer run_ui_regions for layout-first.
    """
    try:
        from src.infrastructure.layout.region_prepass import cv_detect_regions
    except ImportError:
        logger.warning("debug: layout dependency not available")
        return []
    regions = cv_detect_regions(image_path)
    n_total = len(regions)
    n_text = sum(1 for r in regions if r.region_type == "text_region")
    n_ui = sum(1 for r in regions if r.region_type == "ui_region")
    n_bg = sum(1 for r in regions if r.region_type == "background")
    logger.info(
        "debug/layout: image_path=%s regions=%d (text=%d ui=%d bg=%d)",
        image_path, n_total, n_text, n_ui, n_bg,
    )
    return [
        {"x": r.x, "y": r.y, "w": r.w, "h": r.h, "type": r.region_type}
        for r in regions
    ]


def run_ui_regions(image_path: str) -> List[Dict[str, Any]]:
    """
    Layout-first UI region detection. DL (LayoutParser + Detectron2) first, fallback to CV.
    No OCR. Returns list of {"x", "y", "w", "h", "type", "parent_region_id", "_detector_source"}.
    type in: button, badge, card, text_region, navbar, section, input_like, pill.
    _detector_source: "dl" | "cv" — откуда пришёл регион (для отладки).
    """
    regions: List[Dict[str, Any]] = []
    source = "dl"
    try:
        from src.infrastructure.layout.dl_region_detection import dl_detect_ui_regions
        regions = dl_detect_ui_regions(image_path)
    except ImportError:
        logger.debug("debug: dl_region_detection not available, using CV fallback")
        source = "cv"
    except Exception as e:
        logger.warning("debug: DL region detection failed: %s, using CV fallback", e)
        source = "cv"

    if not regions:
        try:
            from src.infrastructure.layout.ui_region_detection import cv_detect_ui_regions
            regions = cv_detect_ui_regions(image_path)
            source = "cv"
        except ImportError:
            logger.warning("debug: ui_region_detection not available")
    for r in regions:
        r["_detector_source"] = source
    logger.info("debug/ui-regions: image_path=%s regions=%d source=%s", image_path, len(regions), source)
    return regions


def run_ui_regions_dl_only(image_path: str) -> List[Dict[str, Any]]:
    """
    Layout from DL only (LayoutParser PubLayNet). No CV second pass — no button/badge from contours.
    Types: card, section, text_region only. Use with full-page Paddle text + assign by overlap.
    Falls back to cv_detect_ui_regions if DL fails or layoutparser not available.
    """
    regions: List[Dict[str, Any]] = []
    try:
        from src.infrastructure.layout.dl_region_detection import dl_detect_layout_only
        regions = dl_detect_layout_only(image_path)
    except ImportError:
        logger.debug("debug: dl_detect_layout_only not available, using CV fallback")
    except Exception as e:
        logger.warning("debug: DL layout-only failed: %s, using CV fallback", e)

    if not regions:
        try:
            from src.infrastructure.layout.ui_region_detection import cv_detect_ui_regions
            regions = cv_detect_ui_regions(image_path)
        except ImportError:
            logger.warning("debug: ui_region_detection not available")
    logger.info("debug/ui-regions-dl-only: image_path=%s regions=%d", image_path, len(regions))
    return regions


ROI_TEXT_SCALE = 2.0   # scale ROI for text_region, card, etc.
ROI_TEXT_SCALE_BUTTON = 3.0  # higher scale for button/badge so thin text is found
ROI_MIN_SIDE_PX = 32   # pad tiny ROIs so Paddle gets valid input


def _run_text_detect_via_remote(img: Any, image_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Call standalone PaddleOCR service (Linux amd64). Returns list of {x, y, w, h}. On failure returns []."""
    if not PADDLE_OCR_SERVICE_URL:
        return []
    url = f"{PADDLE_OCR_SERVICE_URL.rstrip('/')}/ocr"
    try:
        from PIL import Image as PILImage
        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                body = f.read()
            filename = Path(image_path).name
        else:
            if img is None or getattr(img, "size", 0) == 0:
                return []
            pil_img = PILImage.fromarray(img.astype("uint8") if hasattr(img, "astype") else img)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            body = buf.getvalue()
            filename = "image.png"
        boundary = "----FormBoundary"
        payload = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + body + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return [{"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]} for b in data]
    except Exception as e:
        logger.warning("PaddleOCR remote service failed (%s); returning empty boxes", e)
        return []


def _create_paddle_ocr() -> Any:
    """Create PaddleOCR 2.x instance (paddleocr 2.7.x, paddlepaddle 2.6.x). Never imports when DISABLE_PADDLEOCR=1."""
    if DISABLE_PADDLEOCR:
        return None
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=False, show_log=False, use_gpu=False)


def _run_text_detect_on_image(img: Any, image_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run PaddleOCR 2.x text detection on numpy image (BGR or RGB). Returns list of {x, y, w, h}. No import when DISABLE_PADDLEOCR=1."""
    if DISABLE_PADDLEOCR:
        if PADDLE_OCR_SERVICE_URL:
            boxes = _run_text_detect_via_remote(img, image_path=image_path)
            if boxes:
                logger.info("PaddleOCR via remote service (%s): %d boxes", PADDLE_OCR_SERVICE_URL, len(boxes))
            return boxes
        logger.info("PaddleOCR skipped (DISABLE_PADDLEOCR=1); returning empty text boxes")
        return []
    boxes: List[Dict[str, Any]] = []
    try:
        if img is None or getattr(img, "size", 0) == 0:
            return []
        ocr_engine = _create_paddle_ocr()
        if ocr_engine is None:
            return []
        result = ocr_engine.ocr(img, cls=False)
        if result and result[0]:
            for line in result[0]:
                if not line or len(line) < 2:
                    continue
                box_points, _ = line[0], line[1]
                x_coords = [p[0] for p in box_points]
                y_coords = [p[1] for p in box_points]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    except ImportError as e:
        logger.warning("PaddleOCR not available (ImportError): %s — text detection will return empty", e)
    except Exception as e:
        logger.warning("PaddleOCR text detection failed: %s — returning empty boxes", e)
    return boxes


def run_text_detect(image_path: str) -> List[Dict[str, Any]]:
    """
    Text detection on full image (bbox only). No layout, no OCR.
    For pipeline prefer run_text_detect_per_regions so text is detected INSIDE each region (ROI + scale).
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return []
    path = Path(image_path)
    if not path.exists():
        return []
    try:
        img = np.array(Image.open(path).convert("RGB"))
    except Exception:
        return []
    boxes = _run_text_detect_on_image(img, image_path=image_path)
    logger.info("debug/text-detect: image_path=%s boxes=%d", image_path, len(boxes))
    return boxes


def _preprocess_roi_input_like(roi: Any) -> Any:
    """Grayscale + adaptive threshold for placeholder/label (low contrast). Returns numpy array for Paddle."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return roi
    if roi is None or roi.size == 0:
        return roi
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)


def run_text_detect_roi(
    image_path: str,
    region: Dict[str, Any],
    scale: float = ROI_TEXT_SCALE,
    input_like_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Text detection INSIDE region: crop ROI, scale ×scale, run detector, remap bbox to global.
    input_like_mode: for placeholder/label — scale=3, grayscale + adaptive threshold (do not drop by contrast).
    Very small ROIs are padded to ROI_MIN_SIDE_PX so Paddle gets valid input.
    Returns list of {x, y, w, h} in global image coordinates.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        return []
    path = Path(image_path)
    if not path.exists():
        return []
    try:
        full = np.array(Image.open(path).convert("RGB"))
    except Exception:
        return []
    rx = int(region.get("x", 0))
    ry = int(region.get("y", 0))
    rw = int(region.get("w", 0))
    rh = int(region.get("h", 0))
    if rw <= 0 or rh <= 0:
        return []
    h_full, w_full = full.shape[:2]
    x2 = min(w_full, rx + rw)
    y2 = min(h_full, ry + rh)
    roi = full[ry:y2, rx:x2].copy()
    if roi.size == 0:
        return []
    roi_h, roi_w = roi.shape[:2]
    if input_like_mode:
        scale = 3.0
    min_side = max(ROI_MIN_SIDE_PX, int(min(roi_w, roi_h) * scale))
    if roi_w * scale < min_side or roi_h * scale < min_side:
        target_w = max(min_side, int(roi_w * scale))
        target_h = max(min_side, int(roi_h * scale))
        scaled = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        effective_scale_x = target_w / roi_w
        effective_scale_y = target_h / roi_h
    else:
        scaled = cv2.resize(roi, (int(roi_w * scale), int(roi_h * scale)), interpolation=cv2.INTER_LINEAR)
        effective_scale_x = effective_scale_y = scale
    if input_like_mode:
        scaled = _preprocess_roi_input_like(scaled)
    boxes_local = _run_text_detect_on_image(scaled)
    out: List[Dict[str, Any]] = []
    for b in boxes_local:
        out.append({
            "x": rx + int(b["x"] / effective_scale_x),
            "y": ry + int(b["y"] / effective_scale_y),
            "w": max(1, int(b["w"] / effective_scale_x)),
            "h": max(1, int(b["h"] / effective_scale_y)),
        })
    return out


def run_text_detect_per_regions(
    image_path: str,
    regions: List[Dict[str, Any]],
    scale: float = ROI_TEXT_SCALE,
    scale_button_badge: float = ROI_TEXT_SCALE_BUTTON,
    only_for_types: Optional[Tuple[str, ...]] = None,
) -> List[Dict[str, Any]]:
    """
    Run Paddle text detection INSIDE each region (crop ROI, scale, detect, remap). Each box gets region_id.
    If only_for_types is set, Paddle runs only for those types. Use ("text_region", "button", "badge", "pill", "input_like")
    so text_region gets lines/paragraphs and button/badge/pill/input_like get text_boxes for OCR (ROI ×3 for controls).
    """
    all_boxes: List[Dict[str, Any]] = []
    for ri, reg in enumerate(regions):
        reg_type = reg.get("type", "")
        if only_for_types is not None and reg_type not in only_for_types:
            continue
        s = scale_button_badge if reg_type in ("button", "badge") else scale
        input_like_mode = reg_type == "input_like"
        boxes = run_text_detect_roi(image_path, reg, scale=s, input_like_mode=input_like_mode)
        for b in boxes:
            all_boxes.append({**b, "region_id": ri})
    logger.info(
        "debug/text-detect-per-regions: image_path=%s regions=%d only_for=%s total_boxes=%d",
        image_path, len(regions), only_for_types, len(all_boxes),
    )
    return all_boxes


OCR_FIRST_REGION_TYPES = ("button", "badge", "pill", "input_like")
ROI_OCR_SCALE = 2.5


def classify_text_box_ui_type(
    text_bbox: Dict[str, Any],
    region: Optional[Dict[str, Any]],
    image_area: float,
    label: str,
) -> str:
    """
    Classify a text unit (single box or line) as button/badge/text by geometry + card≠button rule.
    DL does not classify buttons — this post-step does. Used when parent region is card/section.
    Rule: if region_area > text_area * 10 → always 'text' (container, not button).
    """
    if region:
        region_area = float(region.get("w", 0) * region.get("h", 0))
        tw, th = text_bbox.get("w", 0), text_bbox.get("h", 0)
        text_area = tw * th
        if text_area > 0 and region_area > text_area * MAX_BUTTON_AREA_MULT:
            return "text"
    tw = text_bbox.get("w", 0)
    th = text_bbox.get("h", 0)
    if th <= 0:
        return "text"
    if tw < th * 0.3:
        return "text"
    label = (label or "").strip()
    if len(label) <= BADGE_MAX_CHARS and tw <= 200 and th <= 40:
        return "badge"
    if len(label) <= 30 and tw <= 400 and th <= 60:
        return "button"
    return "text"


def _preprocess_roi_for_ocr(roi: Any) -> Any:
    """Grayscale, adaptive threshold or Otsu, optional invert. Returns PIL Image for Tesseract."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    if roi is None or roi.size == 0:
        return None
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    pil = Image.fromarray(thresh)
    return pil


def _preprocess_roi_for_ocr_inverted(roi: Any) -> Any:
    """Same but invert (light text on dark background)."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    if roi is None or roi.size == 0:
        return None
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    pil = Image.fromarray(thresh)
    return pil


def run_ocr_roi(
    image_path: str,
    region: Dict[str, Any],
    scale: float = ROI_OCR_SCALE,
) -> Dict[str, Any]:
    """
    OCR-first for button/badge/pill/input_like: no Paddle. Crop ROI, scale, preprocess (grayscale + Otsu),
    run Tesseract on full ROI. Returns {"text": str, "confidence": float}. If text ≥ 2 chars → label.
    """
    try:
        import cv2
        import numpy as np
        import pytesseract
        from PIL import Image
    except ImportError as e:
        logger.debug("run_ocr_roi: %s", e)
        return {"text": "", "confidence": 0.0}
    path = Path(image_path)
    if not path.exists():
        return {"text": "", "confidence": 0.0}
    try:
        full = np.array(Image.open(path).convert("RGB"))
    except Exception:
        return {"text": "", "confidence": 0.0}
    rx, ry = int(region.get("x", 0)), int(region.get("y", 0))
    rw, rh = int(region.get("w", 0)), int(region.get("h", 0))
    if rw <= 0 or rh <= 0:
        return {"text": "", "confidence": 0.0}
    h_full, w_full = full.shape[:2]
    x2 = min(w_full, rx + rw)
    y2 = min(h_full, ry + rh)
    roi = full[ry:y2, rx:x2]
    if roi.size == 0:
        return {"text": "", "confidence": 0.0}
    roi_h, roi_w = roi.shape[:2]
    scaled = cv2.resize(roi, (int(roi_w * scale), int(roi_h * scale)), interpolation=cv2.INTER_LINEAR)
    pil_norm = _preprocess_roi_for_ocr(scaled)
    if pil_norm is None:
        return {"text": "", "confidence": 0.0}
    try:
        text_norm = pytesseract.image_to_string(pil_norm, config="--psm 7").strip()
        data_norm = pytesseract.image_to_data(pil_norm, config="--psm 7", output_type=pytesseract.Output.DICT)
        confs = [c for c in data_norm.get("conf", []) if c != -1]
        conf_norm = float(sum(confs) / len(confs)) / 100.0 if confs else 0.0
    except Exception:
        text_norm, conf_norm = "", 0.0
    if len(text_norm) >= 2:
        return {"text": text_norm, "confidence": conf_norm}
    pil_inv = _preprocess_roi_for_ocr_inverted(scaled)
    if pil_inv is None:
        return {"text": text_norm, "confidence": conf_norm}
    try:
        text_inv = pytesseract.image_to_string(pil_inv, config="--psm 7").strip()
        data_inv = pytesseract.image_to_data(pil_inv, config="--psm 7", output_type=pytesseract.Output.DICT)
        confs_inv = [c for c in data_inv.get("conf", []) if c != -1]
        conf_inv = float(sum(confs_inv) / len(confs_inv)) / 100.0 if confs_inv else 0.0
    except Exception:
        text_inv, conf_inv = "", 0.0
    if len(text_inv) >= 2 and (conf_inv > conf_norm or len(text_norm) < 2):
        return {"text": text_inv, "confidence": conf_inv}
    return {"text": text_norm, "confidence": conf_norm}


def _preprocess_roi_adaptive_ocr(roi: Any) -> Any:
    """Adaptive threshold for small/pale text. Returns PIL Image or None."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    if roi is None or getattr(roi, "size", 0) == 0:
        return None
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    # Block size odd, C for constant subtracted from mean
    block = max(3, min(gray.shape[0], gray.shape[1]) // 4) | 1
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 2
    )
    return Image.fromarray(adaptive)


def run_ocr_boxes(image_path: str, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    OCR on given bbox list. Returns list of {"text", "confidence"} per box.
    """
    try:
        import numpy as np
        import pytesseract
        from PIL import Image
    except ImportError as e:
        logger.warning("debug/ocr: dependency not available: %s", e)
        return [{"text": "", "confidence": 0.0} for _ in boxes]
    path = Path(image_path)
    if not path.exists():
        logger.warning("debug/ocr: image not found %s", image_path)
        return [{"text": "", "confidence": 0.0} for _ in boxes]
    try:
        img = np.array(Image.open(path).convert("RGB"))
    except Exception as e:
        logger.warning("debug/ocr: failed to load %s: %s", image_path, e)
        return [{"text": "", "confidence": 0.0} for _ in boxes]

    results: List[Dict[str, Any]] = []
    for i, box in enumerate(boxes):
        x = int(box.get("x", 0))
        y = int(box.get("y", 0))
        w = int(box.get("w", 0))
        h = int(box.get("h", 0))
        if w <= 0 or h <= 0:
            results.append({"text": "", "confidence": 0.0})
            logger.debug("debug/ocr: box[%d] skipped (invalid w/h)", i)
            continue
        h_img, w_img = img.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            results.append({"text": "", "confidence": 0.0})
            continue
        crop = img[y1:y2, x1:x2]
        pil_crop = Image.fromarray(crop)
        try:
            text = pytesseract.image_to_string(pil_crop, config="--psm 7").strip()
            data = pytesseract.image_to_data(pil_crop, config="--psm 7", output_type=pytesseract.Output.DICT)
            confs = [c for c in data.get("conf", []) if c != -1]
            conf = float(sum(confs) / len(confs)) / 100.0 if confs else 0.0
        except Exception as e:
            logger.debug("debug/ocr: box[%d] tesseract failed: %s", i, e)
            text, conf = "", 0.0
        results.append({"text": text, "confidence": conf})
    logger.info("debug/ocr: image_path=%s boxes=%d results=%d", image_path, len(boxes), len(results))
    return results


def run_ocr_boxes_with_adaptive(
    image_path: str,
    boxes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    OCR with fallback for small/pale text: Otsu first, then adaptive threshold, then inverted.
    Returns list of {"text", "confidence"} per box. Uses Tesseract; no Paddle.
    """
    try:
        import cv2
        import numpy as np
        import pytesseract
        from PIL import Image
    except ImportError as e:
        logger.warning("debug/ocr-adaptive: dependency not available: %s", e)
        return [{"text": "", "confidence": 0.0} for _ in boxes]
    path = Path(image_path)
    if not path.exists():
        return [{"text": "", "confidence": 0.0} for _ in boxes]
    try:
        img = np.array(Image.open(path).convert("RGB"))
    except Exception:
        return [{"text": "", "confidence": 0.0} for _ in boxes]

    results: List[Dict[str, Any]] = []
    for i, box in enumerate(boxes):
        x = int(box.get("x", 0))
        y = int(box.get("y", 0))
        w = int(box.get("w", 0))
        h = int(box.get("h", 0))
        if w <= 0 or h <= 0:
            results.append({"text": "", "confidence": 0.0})
            continue
        h_img, w_img = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            results.append({"text": "", "confidence": 0.0})
            continue
        crop = img[y1:y2, x1:x2]

        def _run_ocr(pil_img: Any) -> Tuple[str, float]:
            try:
                t = pytesseract.image_to_string(pil_img, config="--psm 7").strip()
                data = pytesseract.image_to_data(pil_img, config="--psm 7", output_type=pytesseract.Output.DICT)
                confs = [c for c in data.get("conf", []) if c != -1]
                c = float(sum(confs) / len(confs)) / 100.0 if confs else 0.0
                return t, c
            except Exception:
                return "", 0.0

        pil_norm = _preprocess_roi_for_ocr(crop)
        text_norm, conf_norm = ("", 0.0)
        if pil_norm is not None:
            text_norm, conf_norm = _run_ocr(pil_norm)
        if len(text_norm) >= 2:
            results.append({"text": text_norm, "confidence": conf_norm})
            continue
        pil_adapt = _preprocess_roi_adaptive_ocr(crop)
        if pil_adapt is not None:
            text_adapt, conf_adapt = _run_ocr(pil_adapt)
            if len(text_adapt) >= 2 and (conf_adapt > conf_norm or len(text_norm) < 2):
                results.append({"text": text_adapt, "confidence": conf_adapt})
                continue
        pil_inv = _preprocess_roi_for_ocr_inverted(crop)
        if pil_inv is not None:
            text_inv, conf_inv = _run_ocr(pil_inv)
            if len(text_inv) >= 2 and (conf_inv > conf_norm or len(text_norm) < 2):
                results.append({"text": text_inv, "confidence": conf_inv})
                continue
        results.append({"text": text_norm, "confidence": conf_norm})
    return results


def run_full_pipeline(image_path: str, use_dl_only: bool = False) -> Dict[str, Any]:
    """
    Layout-first: UI regions → text detection → assign → one-line-one-region → lines/paragraphs → gui_blocks.

    use_dl_only=True: DL layout only (no CV second pass), Paddle text on full image, assign by overlap.
    Types from model: card, section, text_region only. No button/badge from contours.

    use_dl_only=False: run_ui_regions (DL+CV or CV), Paddle text per ROI per type.

    Debug transparency: returns raw_paddle_text_boxes, text_boxes, drop_reasons, text_box_drop_reasons.
    """
    from src.infrastructure.layout.text_grouping import (
        assign_text_boxes_to_regions,
        ensure_one_line_one_region,
        group_text_boxes_into_lines,
        group_lines_into_paragraphs,
    )

    # Debug transparency: track whether OCR ran and why text might be empty
    ocr_called = True
    ocr_engine = "paddle"
    bpg_fallback_called = False
    reason_why_text_boxes_empty: Optional[str] = None

    regions = run_ui_regions_dl_only(image_path) if use_dl_only else run_ui_regions(image_path)
    if use_dl_only:
        raw_text_boxes = run_text_detect(image_path)
        boxes_with_region = assign_text_boxes_to_regions(
            raw_text_boxes, regions, min_overlap_ratio=0.3, prefer_smallest_region=True
        )
        # Fallback: DL regions (card/section) with no text must get ROI text detection
        n_initial = len(boxes_with_region)
        for rid, reg in enumerate(regions):
            if reg.get("type") not in ("card", "section"):
                continue
            if any(b.get("region_id") == rid for b in boxes_with_region):
                continue
            fallback_boxes = run_text_detect_roi(image_path, reg, scale=ROI_TEXT_SCALE)
            for fb in fallback_boxes:
                fb["region_id"] = rid
                fb["box_index"] = n_initial + len(boxes_with_region)
                boxes_with_region.append(fb)
        raw_text_boxes = boxes_with_region[:n_initial] if n_initial else []
    else:
        raw_text_boxes = run_text_detect_per_regions(
            image_path, regions,
            only_for_types=("text_region", "button", "badge", "pill", "input_like"),
        )
        boxes_with_region = assign_text_boxes_to_regions(
            raw_text_boxes, regions, min_overlap_ratio=0.3, prefer_smallest_region=True
        )

    # BPG/fallback: if primary text detection returned 0 boxes, try full-page (exceptional — not normal for UI screenshots)
    if len(boxes_with_region) == 0:
        logger.warning("OCR returned 0 text boxes for %s — running full-page text detection fallback", image_path)
        bpg_fallback_called = True
        full_page = run_text_detect(image_path)
        if full_page:
            boxes_with_region = assign_text_boxes_to_regions(
                full_page, regions, min_overlap_ratio=0.3, prefer_smallest_region=True
            )
            raw_text_boxes = full_page
        if len(boxes_with_region) == 0:
            reason_why_text_boxes_empty = "paddle_returned_empty_or_unavailable"

    assert raw_text_boxes is not None
    logger.info("debug/full-pipeline: raw_paddle_text_boxes len=%d", len(raw_text_boxes))

    boxes_with_index = [{**b, "box_index": i} for i, b in enumerate(boxes_with_region)]
    boxes_with_index = ensure_one_line_one_region(boxes_with_index, regions)
    text_boxes = boxes_with_index
    ocr_results = run_ocr_boxes(image_path, text_boxes) if text_boxes else []

    try:
        from PIL import Image
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    except Exception:
        img_w, img_h = 800, 600
    image_area = img_w * img_h

    all_lines: List[List[Dict[str, Any]]] = []
    all_paragraphs: List[List[List[Dict[str, Any]]]] = []
    gui_blocks: List[Dict[str, Any]] = []
    drop_reasons: List[str] = []
    # If a raw box is ever explicitly dropped, append {"box_index": i, "reason": "..."}. No drop by confidence/contrast.
    text_box_drop_reasons: List[Dict[str, Any]] = []
    emitted_box_indices: Set[int] = set()  # invariant: every box either in a gui_block or in text_box_drop_reasons
    button_labels: Dict[int, str] = {}
    empty_parent_ids: Set[int] = set()

    def is_empty_parent(rid: int) -> bool:
        if rid >= len(regions):
            return True
        r = regions[rid]
        if r.get("parent_region_id") is not None:
            return False
        if r.get("type") not in ("navbar", "section"):
            return False
        has_boxes = any(b.get("region_id") == rid for b in boxes_with_index)
        if has_boxes:
            return False
        has_children = any(regions[j].get("parent_region_id") == rid for j in range(len(regions)))
        return not has_children

    for rid, reg in enumerate(regions):
        reg_type = reg.get("type", "text_region")
        boxes_in_region = [b for b in boxes_with_index if b.get("region_id") == rid]

        if is_empty_parent(rid):
            drop_reasons.append(f"empty parent region {rid} ({reg_type})")
            continue

        if not use_dl_only and reg_type in OCR_FIRST_REGION_TYPES:
            if boxes_in_region:
                texts = [ocr_results[b["box_index"]].get("text", "") for b in boxes_in_region if 0 <= b.get("box_index", -1) < len(ocr_results)]
                label = " ".join(t for t in texts if t).strip()
            else:
                ocr = run_ocr_roi(image_path, reg, scale=ROI_OCR_SCALE)
                label = ocr.get("text", "").strip()
            if len(label) >= 2:
                button_labels[rid] = label
            # Invariant: no button/badge without text — if no label, we will downgrade to section in post-pass
            container_bbox = {"x": reg["x"], "y": reg["y"], "w": reg["w"], "h": reg["h"]}
            region_area = reg["w"] * reg["h"]
            block_area = region_area
            if image_area > 0 and block_area >= MAX_BLOCK_SCREEN_RATIO * image_area:
                drop_reasons.append(f"block area {block_area} >= 80% screen")
                # Text priority: still emit text so no text_box is lost
                if boxes_in_region:
                    xs = [b["x"] for b in boxes_in_region]
                    ys = [b["y"] for b in boxes_in_region]
                    x2s = [b["x"] + b["w"] for b in boxes_in_region]
                    y2s = [b["y"] + b["h"] for b in boxes_in_region]
                    text_bbox = {"x": min(xs), "y": min(ys), "w": max(x2s) - min(xs), "h": max(y2s) - min(ys)}
                    lbl = " ".join(ocr_results[b["box_index"]].get("text", "") for b in boxes_in_region if 0 <= b.get("box_index", -1) < len(ocr_results))
                    gui_blocks.append({
                        "region_index": rid,
                        "region_type": "section",
                        "container_bbox": container_bbox,
                        "text_bbox": text_bbox,
                        "text": lbl.strip(),
                        "text_boxes_count": len(boxes_in_region),
                    })
                    for b in boxes_in_region:
                        emitted_box_indices.add(b.get("box_index", -1))
            else:
                if boxes_in_region:
                    xs = [b["x"] for b in boxes_in_region]
                    ys = [b["y"] for b in boxes_in_region]
                    x2s = [b["x"] + b["w"] for b in boxes_in_region]
                    y2s = [b["y"] + b["h"] for b in boxes_in_region]
                    text_bbox = {"x": min(xs), "y": min(ys), "w": max(x2s) - min(xs), "h": max(y2s) - min(ys)}
                    text_area = text_bbox["w"] * text_bbox["h"]
                    # Huge region → container, not button (text priority: still emit text)
                    if text_area > 0 and region_area > text_area * MAX_BUTTON_AREA_MULT:
                        drop_reasons.append(f"region {rid} area {region_area} > text_area*{MAX_BUTTON_AREA_MULT} (container not button)")
                        gui_blocks.append({
                            "region_index": rid,
                            "region_type": "section",
                            "container_bbox": container_bbox,
                            "text_bbox": text_bbox,
                            "text": label,
                            "text_boxes_count": len(boxes_in_region),
                        })
                        for b in boxes_in_region:
                            emitted_box_indices.add(b.get("box_index", -1))
                    else:
                        # Badge vs button by semantics: h_region/h_text ≤ 1.4 + short text → badge
                        reg_h, text_h = reg["h"], max(1, text_bbox["h"])
                        effective_type = reg_type
                        if reg_h / text_h <= BADGE_H_REGION_TEXT_RATIO and len(label) <= BADGE_MAX_CHARS:
                            effective_type = "badge"
                        else:
                            effective_type = "button"
                        gui_blocks.append({
                            "region_index": rid,
                            "region_type": effective_type,
                            "container_bbox": container_bbox,
                            "text_bbox": text_bbox,
                            "text": label,
                            "text_boxes_count": len(boxes_in_region),
                        })
                        for b in boxes_in_region:
                            emitted_box_indices.add(b.get("box_index", -1))
                else:
                    # Phantom button/badge (no text): emit as section, not button (post-pass will enforce)
                    text_bbox = dict(container_bbox)
                    effective_type = "section" if (reg_type in ("button", "badge") and not (label or "").strip()) else reg_type
                    gui_blocks.append({
                        "region_index": rid,
                        "region_type": effective_type,
                        "container_bbox": container_bbox,
                        "text_bbox": text_bbox,
                        "text": label,
                        "text_boxes_count": 0,
                    })
                    for b in boxes_in_region:
                        emitted_box_indices.add(b.get("box_index", -1))
            continue

        if reg_type == "text_region" or (use_dl_only and reg_type in ("card", "section")):
            lines = group_text_boxes_into_lines(boxes_with_index, rid)
            all_lines.extend(lines)
            paragraphs = group_lines_into_paragraphs(lines)
            all_paragraphs.extend(paragraphs)
            boxes_in_paras: Set[int] = set()
            for para in paragraphs:
                boxes_in_para = [b for ln in para for b in ln]
                for b in boxes_in_para:
                    idx = b.get("box_index", -1)
                    if idx >= 0:
                        boxes_in_paras.add(idx)
                xs = [b["x"] for b in boxes_in_para]
                ys = [b["y"] for b in boxes_in_para]
                x2s = [b["x"] + b["w"] for b in boxes_in_para]
                y2s = [b["y"] + b["h"] for b in boxes_in_para]
                text_bbox = {"x": min(xs), "y": min(ys), "w": max(x2s) - min(xs), "h": max(y2s) - min(ys)}
                block_area = text_bbox["w"] * text_bbox["h"]
                if image_area > 0 and block_area >= MAX_BLOCK_SCREEN_RATIO * image_area:
                    drop_reasons.append(f"block area {block_area} >= 80% screen")
                    continue
                texts = [ocr_results[b["box_index"]].get("text", "") for b in boxes_in_para if 0 <= b.get("box_index", -1) < len(ocr_results)]
                label_para = " ".join(t for t in texts if t)
                effective_type = reg_type
                if use_dl_only and reg_type in ("card", "section"):
                    effective_type = classify_text_box_ui_type(text_bbox, reg, image_area, label_para)
                gui_blocks.append({
                    "region_index": rid,
                    "region_type": effective_type,
                    "container_bbox": reg,
                    "text_bbox": text_bbox,
                    "text": label_para,
                    "text_boxes_count": len(boxes_in_para),
                })
                for b in boxes_in_para:
                    emitted_box_indices.add(b.get("box_index", -1))
            # Orphan boxes in this text_region (not in any paragraph): emit so text is never lost
            for b in boxes_in_region:
                if b.get("box_index", -1) in boxes_in_paras:
                    continue
                bt = ocr_results[b["box_index"]].get("text", "") if 0 <= b.get("box_index", -1) < len(ocr_results) else ""
                effective_type = reg_type
                if use_dl_only and reg_type in ("card", "section"):
                    effective_type = classify_text_box_ui_type(
                        {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]}, reg, image_area, bt
                    )
                gui_blocks.append({
                    "region_index": rid,
                    "region_type": effective_type,
                    "container_bbox": reg,
                    "text_bbox": {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]},
                    "text": bt,
                    "text_boxes_count": 1,
                })
                emitted_box_indices.add(b.get("box_index", -1))
            continue

        container_bbox = {"x": reg["x"], "y": reg["y"], "w": reg["w"], "h": reg["h"]}
        block_area = reg["w"] * reg["h"]
        if image_area > 0 and block_area >= MAX_BLOCK_SCREEN_RATIO * image_area:
            drop_reasons.append(f"block area {block_area} >= 80% screen")
        if boxes_in_region:
            # Text priority: emit every text box even in non-text regions (e.g. navbar)
            for b in boxes_in_region:
                idx = b.get("box_index", -1)
                bt = ocr_results[idx].get("text", "") if 0 <= idx < len(ocr_results) else ""
                tb = {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]}
                effective_type = classify_text_box_ui_type(tb, reg, image_area, bt)
                gui_blocks.append({
                    "region_index": rid,
                    "region_type": effective_type,
                    "container_bbox": container_bbox,
                    "text_bbox": tb,
                    "text": bt,
                    "text_boxes_count": 1,
                })
                emitted_box_indices.add(idx)
        elif image_area <= 0 or block_area < MAX_BLOCK_SCREEN_RATIO * image_area:
            # No button/badge without text: downgrade to section when DL gave button/badge but no boxes
            effective_type = "section" if reg_type in ("button", "badge") else reg_type
            gui_blocks.append({
                "region_index": rid,
                "region_type": effective_type,
                "container_bbox": container_bbox,
                "text_bbox": dict(container_bbox),
                "text": "",
                "text_boxes_count": 0,
            })

    # Text priority: every box that made it to debug must appear in result. Emit standalone for region_id=-1.
    for b in boxes_with_index:
        if b.get("region_id", -1) != -1:
            continue
        idx = b.get("box_index", -1)
        if idx < 0 or idx >= len(ocr_results):
            continue
        bt = ocr_results[idx].get("text", "")
        gui_blocks.append({
            "region_index": -1,
            "region_type": "text_region",
            "container_bbox": {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]},
            "text_bbox": {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]},
            "text": bt,
            "text_boxes_count": 1,
        })
        emitted_box_indices.add(idx)
        logger.debug("debug: emitted standalone text block for unassigned box_index=%d text=%r", idx, bt[:30])

    # Invariant: every text_box either in a gui_block or in text_box_drop_reasons (no silent drop)
    for b in boxes_with_index:
        idx = b.get("box_index", -1)
        if idx < 0 or idx in emitted_box_indices:
            continue
        text_box_drop_reasons.append({
            "box_index": idx,
            "reason": "not_emitted",
        })
        logger.debug("debug: text_box box_index=%d not emitted (recorded in text_box_drop_reasons)", idx)

    # Invariant: no button/badge without text — downgrade any remaining to section
    for blk in gui_blocks:
        if blk.get("region_type") in ("button", "badge") and blk.get("text_boxes_count", 0) == 0 and not (blk.get("text") or "").strip():
            blk["region_type"] = "section"

    if not text_boxes and reason_why_text_boxes_empty is None:
        reason_why_text_boxes_empty = "no_text_detected"

    message = "Layout-first; raw/final boxes and drop reasons for transparency."
    if not text_boxes:
        message = "No text detected; layout-only. Check OCR/engine (ocr_engine, reason_why_text_boxes_empty). Layout-first requires text."

    logger.info(
        "debug/full-pipeline: image_path=%s regions=%d raw_boxes=%d text_boxes=%d lines=%d paragraphs=%d gui_blocks=%d button_labels=%d dropped=%d",
        image_path, len(regions), len(raw_text_boxes), len(text_boxes), len(all_lines), sum(len(p) for p in all_paragraphs), len(gui_blocks), len(button_labels), len(drop_reasons),
    )
    return {
        "regions": regions,
        "raw_paddle_text_boxes": raw_text_boxes,
        "text_boxes": text_boxes,
        "text_box_drop_reasons": text_box_drop_reasons,
        "lines": all_lines,
        "paragraphs": all_paragraphs,
        "gui_blocks": gui_blocks,
        "button_labels": button_labels,
        "empty_parent_ids": empty_parent_ids,
        "dropped_count": len(drop_reasons),
        "drop_reasons": drop_reasons,
        "ocr_called": ocr_called,
        "ocr_engine": ocr_engine,
        "bpg_called": bpg_fallback_called,
        "reason_why_text_boxes_empty": reason_why_text_boxes_empty,
        "message": message,
    }


def _normalize_gui_type(region_type: str) -> str:
    """Map layout region type to output: button, card, text, input."""
    t = (region_type or "").strip().lower()
    if t in ("button", "badge", "pill"):
        return "button"
    if t == "input_like":
        return "input"
    if t == "card":
        return "card"
    return "text"


LAYOUT_BBOX_MARGIN_PX = 5
BUTTON_LIKE_MAX_CHARS = 30
BUTTON_LIKE_MAX_HEIGHT_PX = 48
# Layout container NEVER is final block: if block area >= this ratio of container area, split into lines.
CONTAINER_BBOX_OVERLAP_FORBID = 0.75  # block >= 75% container → emit lines, not one block
# Pre-filter: (0,0, >80%w, <15%h) = page_frame/chrome
PAGE_FRAME_WIDTH_RATIO = 0.8
PAGE_FRAME_HEIGHT_RATIO_MAX = 0.15
# Переключение: REMOVE_PAGE_FRAME_REGIONS=1 — удалять верхний блок из списка (не OCR, не emit).
# Иначе — только помечать page_frame (OCR=yes, emit=no).
def _remove_page_frame_regions_enabled() -> bool:
    return os.environ.get("REMOVE_PAGE_FRAME_REGIONS", "").strip().lower() in ("1", "true", "yes")
# Table: same row if y_center within this (px)
TABLE_ROW_Y_TOLERANCE_PX = 20


def _layout_container_type(reg_type: str) -> str:
    """Map layout region to container: header, card, table, button, panel. No text bbox at layout stage."""
    t = (reg_type or "").strip().lower()
    if t in ("navbar", "header"):
        return "header"
    if t == "card":
        return "card"
    if t == "table":
        return "table"
    if t in ("button", "badge", "pill"):
        return "button"
    if t == "input_like":
        return "input"
    return "panel"


def _is_page_frame_geometry(reg: Dict[str, Any], img_w: int, img_h: int) -> bool:
    """(0,0, >80%w, <15%h) = внешний блок (page_frame/chrome). Не участвует в пайплайне."""
    if img_w <= 0 or img_h <= 0:
        return False
    x, y, w, h = reg.get("x", 0), reg.get("y", 0), reg.get("w", 0), reg.get("h", 0)
    return (
        x == 0
        and y == 0
        and w >= PAGE_FRAME_WIDTH_RATIO * img_w
        and h <= PAGE_FRAME_HEIGHT_RATIO_MAX * img_h
    )


def _is_button_like_bbox(w: int, h: int, text: str) -> bool:
    """Short text + small bbox → button so not merged with adjacent sentence."""
    if h > BUTTON_LIKE_MAX_HEIGHT_PX:
        return False
    return len((text or "").strip()) <= BUTTON_LIKE_MAX_CHARS and w <= 400


def _iou_box_region(box: Dict[str, Any], reg: Dict[str, Any]) -> float:
    """IoU(box, region) = intersection_area / box_area. 0 if no overlap."""
    bx, by, bw, bh = box.get("x", 0), box.get("y", 0), box.get("w", 0), box.get("h", 0)
    rx, ry, rw, rh = reg.get("x", 0), reg.get("y", 0), reg.get("w", 0), reg.get("h", 0)
    ix1 = max(bx, rx)
    iy1 = max(by, ry)
    ix2 = min(bx + bw, rx + rw)
    iy2 = min(by + bh, ry + rh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    box_area = max(1, bw * bh)
    return inter / box_area


def _center_in_region(box: Dict[str, Any], reg: Dict[str, Any]) -> bool:
    """True если центр box лежит внутри reg."""
    bx, by, bw, bh = box.get("x", 0), box.get("y", 0), box.get("w", 0), box.get("h", 0)
    cx, cy = bx + bw // 2, by + bh // 2
    rx, ry, rw, rh = reg.get("x", 0), reg.get("y", 0), reg.get("w", 0), reg.get("h", 0)
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def _assign_ocr_to_regions_by_iou(
    text_boxes: List[Dict[str, Any]],
    ocr_results: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """
    Привязка OCR к контейнерам: IoU, при отсутствии перекрытия — fallback по центру bbox.
    """
    all_boxes_with_region: List[Dict[str, Any]] = []
    region_ocr: Dict[int, List[Dict[str, Any]]] = {ri: [] for ri in range(len(regions))}
    for i, box in enumerate(text_boxes):
        ocr = ocr_results[i] if i < len(ocr_results) else {}
        best_ri = -1
        best_iou = 0.0
        best_area = 0
        for ri, reg in enumerate(regions):
            iou = _iou_box_region(box, reg)
            if iou > 0:
                area = reg.get("w", 0) * reg.get("h", 0)
                if iou > best_iou or (iou == best_iou and (best_ri < 0 or area < best_area)):
                    best_iou = iou
                    best_ri = ri
                    best_area = area
        if best_ri < 0:
            for ri, reg in enumerate(regions):
                if _center_in_region(box, reg):
                    area = reg.get("w", 0) * reg.get("h", 0)
                    if best_ri < 0 or area < best_area:
                        best_ri = ri
                        best_area = area
        if best_ri >= 0:
            all_boxes_with_region.append({
                "x": box.get("x", 0), "y": box.get("y", 0),
                "w": box.get("w", 0), "h": box.get("h", 0),
                "region_id": best_ri,
            })
            region_ocr[best_ri].append(ocr)
    return all_boxes_with_region, region_ocr


def _table_boxes_to_rows(
    boxes: List[Dict[str, Any]],
    y_tolerance_px: int = TABLE_ROW_Y_TOLERANCE_PX,
) -> List[List[Dict[str, Any]]]:
    """
    Group boxes into rows by y_center. Distances are between OCR bbox only; container is not used.
    Returns list of rows; each row = list of boxes sorted by x.
    """
    if not boxes:
        return []
    with_center = [(b, b.get("y", 0) + 0.5 * b.get("h", 0)) for b in boxes]
    sorted_by_y = sorted(with_center, key=lambda t: t[1])
    rows: List[List[Dict[str, Any]]] = []
    current_row: List[Dict[str, Any]] = [sorted_by_y[0][0]]
    ref_y = sorted_by_y[0][1]
    for b, yc in sorted_by_y[1:]:
        if abs(yc - ref_y) <= y_tolerance_px:
            current_row.append(b)
        else:
            current_row.sort(key=lambda bx: bx.get("x", 0))
            rows.append(current_row)
            current_row = [b]
            ref_y = yc
    if current_row:
        current_row.sort(key=lambda bx: bx.get("x", 0))
        rows.append(current_row)
    return rows


def run_improved_full_pipeline(image_path: str) -> Dict[str, Any]:
    """
    Vision-first: границы контейнеров только из vision (Detectron2 web UI или fallback DL/CV).
    OCR не создаёт контейнеры; текст привязывается к ближайшему UI-элементу по IoU.
    Порядок: 1) Vision → регионы (card, button, input, navbar), 2) Full-page OCR, 3) Assign OCR по IoU,
    4) GUI blocks: bbox из vision, текст из OCR. Текст никогда не является контейнером.
    """
    from src.infrastructure.layout.text_grouping import (
        group_lines_into_paragraphs_strict,
        group_text_boxes_into_lines_strict,
    )

    layout_log: List[str] = []
    ocr_log: List[str] = []
    merge_log: List[str] = []

    # --- Step 1: Vision-only — Detectron2 Mask R-CNN (web UI). PubLayNet fallback убран. ---
    regions = []
    try:
        from src.infrastructure.layout.vision_ui_detection import run_vision_ui_regions
        regions = run_vision_ui_regions(image_path)
        if regions:
            layout_log.append("layout_source=vision_ui")
    except ImportError:
        logger.debug("improved-pipeline: vision_ui not available")
    except Exception as e:
        logger.warning("improved-pipeline: vision_ui failed %s", e)
    # Нет fallback на PubLayNet/DL: только vision. Нет регионов — пустой результат.
    img_w, img_h = 0, 0
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            img_w, img_h = im.width, im.height
    except Exception:
        pass
    for ri, r in enumerate(regions):
        r["layout_id"] = f"L{ri}"
        r["container_type"] = _layout_container_type(r.get("type", "text_region"))
        ct = r.get("container_type", "panel")
        if ct not in ("button", "input", "header") and _is_page_frame_geometry(r, img_w, img_h):
            r["container_type"] = "page_frame"
            layout_log.append("page_frame detected: OCR=yes, emit=no")
            logger.info("improved-pipeline page_frame detected: layout_id=%s OCR=yes, emit=no", r["layout_id"])
    type_counts = {}
    for r in regions:
        t = r.get("container_type", "panel")
        type_counts[t] = type_counts.get(t, 0) + 1
    layout_log.append(f"containers={len(regions)} ids=L0..L{len(regions)-1} types={type_counts}")
    logger.info("improved-pipeline Layout: %s", layout_log[-1])

    if not regions:
        return {
            "gui_blocks": [],
            "layout_log": layout_log,
            "ocr_log": ocr_log,
            "merge_log": merge_log,
            "ocr_skipped": DISABLE_PADDLEOCR,
            "debug_image_path": None,
            "message": "No layout regions detected.",
        }

    # --- Step 2: OCR только для текста. Full-page OCR → привязка к контейнерам по IoU. OCR не создаёт контейнеры. ---
    text_boxes = run_text_detect(image_path)
    ocr_results = run_ocr_boxes(image_path, text_boxes) if text_boxes else []
    all_boxes_with_region, region_ocr = _assign_ocr_to_regions_by_iou(text_boxes, ocr_results, regions)
    for ri in range(len(regions)):
        n = len(region_ocr.get(ri, []))
        ocr_log.append(f"layout_id=L{ri} type={regions[ri].get('type', '')} assigned_ocr_boxes={n}")
    logger.info("improved-pipeline OCR: full-page, assign by IoU to %d regions", len(regions))

    # --- Step 3: Text BBox — STRICT grouping, no merge across containers; emit CHILDREN only ---
    # Containers = soft spatial filter only (limit OCR region); not used as anchor for block bbox.
    gui_blocks: List[Dict[str, Any]] = []
    image_area = 1.0
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            image_area = float(im.width * im.height)
    except Exception:
        pass

    for ri, reg in enumerate(regions):
        layout_id = reg.get("layout_id", f"L{ri}")
        container_type = reg.get("container_type", "panel")
        reg_type = reg.get("type", "text_region")
        rx, ry, rw, rh = reg.get("x", 0), reg.get("y", 0), reg.get("w", 0), reg.get("h", 0)
        reg_boxes = [b for b in all_boxes_with_region if b.get("region_id") == ri]
        ocr_list = region_ocr.get(ri, [])

        # page_frame: не эмитим один блок по контейнеру; fallback — 1 OCR bbox = 1 GUI block (чтобы верхняя зона не была пустой).
        if container_type == "page_frame":
            if reg_boxes and ocr_list:
                merge_log.append(f"layout_id={layout_id} page_frame fallback: 1 OCR bbox = 1 GUI block ({len(reg_boxes)} blocks)")
                logger.info("improved-pipeline page_frame fallback: layout_id=%s → 1 bbox = 1 block (%d blocks)", layout_id, len(reg_boxes))
                for bi, b in enumerate(reg_boxes):
                    t = ocr_list[bi].get("text", "") if bi < len(ocr_list) else ""
                    c = ocr_list[bi].get("confidence", 0.0) if bi < len(ocr_list) else 0.0
                    gui_blocks.append({"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"], "type": "text", "text": t, "confidence": round(c, 4), "parent": layout_id})
            continue

        # Buttons/inputs: always emit by layout bbox, even without OCR
        if container_type in ("button", "input"):
            text = " ".join(ocr_list[i].get("text", "") for i in range(len(ocr_list))).strip() if ocr_list else ""
            conf = (float(sum(ocr_list[i].get("confidence", 0) for i in range(len(ocr_list))) / len(ocr_list))) if ocr_list else 0.0
            gui_blocks.append({"x": rx, "y": ry, "w": rw, "h": rh, "type": container_type, "text": text, "confidence": round(conf, 4), "parent": layout_id})
            continue

        if container_type == "table":
            # Table: structure first (rows/cells), then text. If table doesn't split → no blocks, only warning.
            rows = _table_boxes_to_rows(reg_boxes)
            logger.info("improved-pipeline container %s bbox=(%d,%d,%d,%d) → text_bbox count=%d rows=%d", layout_id, rx, ry, rw, rh, len(reg_boxes), len(rows))
            if not rows:
                logger.warning("improved-pipeline: layout container %s (table) yielded no rows; no gui_blocks emitted", layout_id)
                continue
            # "Table didn't split": single cell covering table → do not create any text/gui block
            if len(rows) == 1 and len(rows[0]) == 1:
                logger.warning("improved-pipeline: layout container %s (table) did not split (single row, single cell); no gui_blocks emitted", layout_id)
                continue
            for row in rows:
                for b in row:
                    bi = next((j for j, br in enumerate(reg_boxes) if br.get("x") == b.get("x") and br.get("y") == b.get("y")), -1)
                    t = ocr_list[bi].get("text", "") if 0 <= bi < len(ocr_list) else ""
                    c = ocr_list[bi].get("confidence", 0.0) if 0 <= bi < len(ocr_list) else 0.0
                    gui_blocks.append({"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"], "type": "text", "text": t, "confidence": round(c, 4), "parent": layout_id})
            continue

        if reg_boxes and ocr_list:
            if container_type in ("header", "card", "panel"):
                # Nested: emit CHILDREN only. Эмит по СТРОКАМ, без объединения в параграфы — карточки по границам, не по словам.
                region_boxes = [dict(b, box_index=i) for i, b in enumerate(b for b in all_boxes_with_region if b.get("region_id") == ri)]
                lines = group_text_boxes_into_lines_strict(region_boxes, ri)
                # Одна строка = один GUI-блок (не объединяем строки в параграфы — меньше склеивания по вертикали).
                paragraphs = [[ln] for ln in lines]
                container_area = rw * rh
                merge_log.append(f"layout_id={layout_id} lines={len(lines)} emit_per_line (no paragraph merge)")
                logger.info(
                    "improved-pipeline merge before: layout_id=%s bbox=(%d,%d,%d,%d) → text_bbox count=%d",
                    layout_id, rx, ry, rw, rh, len(region_boxes),
                )
                children_count = 0
                for para in paragraphs:
                    flat = [b for ln in para for b in ln]
                    xs = [b["x"] for b in flat]
                    ys = [b["y"] for b in flat]
                    x2s = [b["x"] + b["w"] for b in flat]
                    y2s = [b["y"] + b["h"] for b in flat]
                    x, y = min(xs), min(ys)
                    w = max(x2s) - x
                    h = max(y2s) - y
                    block_area = w * h
                    if image_area > 0 and block_area >= MAX_BLOCK_SCREEN_RATIO * image_area:
                        logger.debug("improved-pipeline: skip oversized block (area >= 80%% screen)")
                        continue
                    # Forbidden: use container bbox as final block. If block ≈ container, split into LINES.
                    if container_area > 0 and block_area >= CONTAINER_BBOX_OVERLAP_FORBID * container_area:
                        overlap_ratio = block_area / container_area if container_area else 0
                        logger.info(
                            "MERGE_REJECT reason=container_overlap gap_px=N/A layout_id=%s block_area_ratio=%.2f boxes=(paragraph_with_%d_lines)",
                            layout_id, overlap_ratio, len(para),
                        )
                        merge_log.append(f"MERGE_REJECT reason=container_overlap layout_id={layout_id} block_area_ratio={overlap_ratio:.2f}")
                        for ln in para:
                            ln_flat = ln
                            lx = min(b["x"] for b in ln_flat)
                            ly = min(b["y"] for b in ln_flat)
                            lw = max(b["x"] + b["w"] for b in ln_flat) - lx
                            lh = max(b["y"] + b["h"] for b in ln_flat) - ly
                            texts_ln = []
                            confs_ln = []
                            for b in ln_flat:
                                j = next((j for j, br in enumerate(region_boxes) if br.get("x") == b.get("x") and br.get("y") == b.get("y")), -1)
                                if 0 <= j < len(ocr_list):
                                    texts_ln.append(ocr_list[j].get("text", ""))
                                    confs_ln.append(ocr_list[j].get("confidence", 0.0))
                            text_ln = " ".join(t for t in texts_ln if t).strip()
                            conf_ln = float(sum(confs_ln) / len(confs_ln)) if confs_ln else 0.0
                            block_type_ln = "button" if _is_button_like_bbox(lw, lh, text_ln) else "text"
                            gui_blocks.append({"x": lx, "y": ly, "w": lw, "h": lh, "type": block_type_ln, "text": text_ln, "confidence": round(conf_ln, 4), "parent": layout_id})
                            children_count += 1
                            logger.debug("improved-pipeline merge after (line): bbox=(%d,%d,%d,%d) type=%s parent=%s", lx, ly, lw, lh, block_type_ln, layout_id)
                        continue
                    # Block must not equal container bbox (forbidden: layout container as final block).
                    if container_area > 0 and block_area >= 0.9 * container_area:
                        logger.debug("improved-pipeline: skip block that equals container bbox (forbidden)")
                        for ln in para:
                            ln_flat = ln
                            lx = min(b["x"] for b in ln_flat)
                            ly = min(b["y"] for b in ln_flat)
                            lw = max(b["x"] + b["w"] for b in ln_flat) - lx
                            lh = max(b["y"] + b["h"] for b in ln_flat) - ly
                            texts_ln = []
                            confs_ln = []
                            for b in ln_flat:
                                j = next((j for j, br in enumerate(region_boxes) if br.get("x") == b.get("x") and br.get("y") == b.get("y")), -1)
                                if 0 <= j < len(ocr_list):
                                    texts_ln.append(ocr_list[j].get("text", ""))
                                    confs_ln.append(ocr_list[j].get("confidence", 0.0))
                            text_ln = " ".join(t for t in texts_ln if t).strip()
                            conf_ln = float(sum(confs_ln) / len(confs_ln)) if confs_ln else 0.0
                            block_type_ln = "button" if _is_button_like_bbox(lw, lh, text_ln) else "text"
                            gui_blocks.append({"x": lx, "y": ly, "w": lw, "h": lh, "type": block_type_ln, "text": text_ln, "confidence": round(conf_ln, 4), "parent": layout_id})
                            children_count += 1
                        continue
                    texts = []
                    confs = []
                    for b in flat:
                        j = next((j for j, br in enumerate(region_boxes) if br.get("x") == b.get("x") and br.get("y") == b.get("y")), -1)
                        if 0 <= j < len(ocr_list):
                            texts.append(ocr_list[j].get("text", ""))
                            confs.append(ocr_list[j].get("confidence", 0.0))
                    text = " ".join(t for t in texts if t).strip()
                    conf = float(sum(confs) / len(confs)) if confs else 0.0
                    block_type = "button" if _is_button_like_bbox(w, h, text) else "text"
                    gui_blocks.append({"x": x, "y": y, "w": w, "h": h, "type": block_type, "text": text, "confidence": round(conf, 4), "parent": layout_id})
                    children_count += 1
                    logger.debug("improved-pipeline merge after: bbox=(%d,%d,%d,%d) type=%s parent=%s", x, y, w, h, block_type, layout_id)
                logger.info("improved-pipeline merge after: layout_id=%s → emitted %d children", layout_id, children_count)
                merge_log.append(f"layout_id={layout_id} → emitted {children_count} children")
                # Fallback: if strict merge yielded 0 blocks, emit one GUI block per OCR box (single OCR line → one block)
                if children_count == 0 and reg_boxes and ocr_list:
                    merge_log.append(f"layout_id={layout_id} fallback: 1 OCR box = 1 GUI block (strict merge yielded 0)")
                    logger.info("improved-pipeline fallback: layout_id=%s → 1 box = 1 block (%d blocks)", layout_id, len(reg_boxes))
                    for bi, b in enumerate(reg_boxes):
                        t = ocr_list[bi].get("text", "") if bi < len(ocr_list) else ""
                        c = ocr_list[bi].get("confidence", 0.0) if bi < len(ocr_list) else 0.0
                        block_type = "button" if _is_button_like_bbox(b.get("w", 0), b.get("h", 0), t) else "text"
                        gui_blocks.append({"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"], "type": block_type, "text": t, "confidence": round(c, 4), "parent": layout_id})
                elif children_count == 0:
                    logger.warning("improved-pipeline: layout container %s (%s) yielded no child elements (no text boxes)", layout_id, container_type)
            # header/card/panel without reg_boxes: no fallback (no OCR boxes)
        elif ocr_list and reg_type in OCR_FIRST_REGION_TYPES and container_type not in ("button", "input"):
            # e.g. panel with roi_ocr only
            text = " ".join(ocr_list[i].get("text", "") for i in range(len(ocr_list))).strip()
            conf = (float(sum(ocr_list[i].get("confidence", 0) for i in range(len(ocr_list))) / len(ocr_list))) if ocr_list else 0.0
            gui_blocks.append({"x": rx, "y": ry, "w": rw, "h": rh, "type": "text", "text": text, "confidence": round(conf, 4), "parent": layout_id})
        elif container_type in ("header", "card", "panel"):
            logger.warning("improved-pipeline: layout container %s (%s) yielded no child elements (no text boxes)", layout_id, container_type)

    merge_log.append(f"gui_blocks={len(gui_blocks)} (children only; containers not final blocks)")
    logger.info("improved-pipeline Merge: %s", merge_log[-1])

    # Инвариант: OCR есть, но GUI-блоков 0 — запрещено молча возвращать empty.
    total_ocr_boxes = len(all_boxes_with_region)
    if total_ocr_boxes > 0 and len(gui_blocks) == 0:
        err_msg = "Pipeline invariant violated: OCR present but no GUI blocks emitted"
        logger.error("improved-pipeline %s (ocr_boxes=%d)", err_msg, total_ocr_boxes)
        merge_log.append(f"ERROR: {err_msg}")

    # Debug image: regions (dashed) + gui_blocks (solid with type)
    debug_image_path: Optional[str] = None
    try:
        stem = Path(image_path).stem
        debug_image_path = save_debug_image_improved_pipeline(
            image_path, regions, gui_blocks, f"improved_{stem}.png"
        )
    except Exception as e:
        logger.warning("debug: failed to save improved-pipeline image: %s", e)

    msg = (
        "Layout → OCR (per block, adaptive) → GUI blocks. "
        "ocr_skipped=true means PaddleOCR in-process is disabled (e.g. Mac); "
        "text can still come from Tesseract when bbox from remote Paddle or run_ocr_roi."
    )

    return {
        "gui_blocks": gui_blocks,
        "layout_log": layout_log,
        "ocr_log": ocr_log,
        "merge_log": merge_log,
        "ocr_skipped": DISABLE_PADDLEOCR,
        "debug_image_path": debug_image_path,
        "message": msg,
    }


def save_debug_image_improved_pipeline(
    image_path: str,
    regions: List[Dict[str, Any]],
    gui_blocks: List[Dict[str, Any]],
    output_filename: str,
) -> Optional[str]:
    """Draw regions (dashed) and gui_blocks (solid + type label). Saves to DEBUG_OUTPUT_DIR. Returns path or None."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    out_dir = _ensure_debug_dir()
    out_path = out_dir / output_filename
    try:
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except Exception:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
            except Exception:
                font = ImageFont.load_default()
        for r in regions:
            if r.get("container_type") == "page_frame":
                continue  # не рисуем page_frame — внешний блок ни к чему не привязан
            x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
            _dashed_rect(draw, x, y, x + w, y + h, COLOR_PARENT_REGIONS)
        type_colors = {"button": (0, 200, 0), "card": (200, 100, 0), "text": (255, 0, 0), "input": (0, 150, 255)}
        for b in gui_blocks:
            x, y, w, h = b.get("x", 0), b.get("y", 0), b.get("w", 0), b.get("h", 0)
            t = b.get("type", "text")
            color = type_colors.get(t, (128, 128, 128))
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            label = (b.get("text", "") or t)[:30]
            if label:
                draw.text((x, max(0, y - 14)), label, fill=color, font=font)
        img.save(out_path, "PNG")
        logger.info("debug: saved improved-pipeline image to %s", out_path)
        return str(out_path)
    except Exception as e:
        logger.warning("debug: failed to save improved-pipeline image %s: %s", out_path, e)
        return None


def save_debug_image_regions(
    image_path: str,
    regions: List[Dict[str, Any]],
    output_filename: str,
    outline_color_override: Optional[Tuple[int, int, int]] = None,
) -> Optional[str]:
    """Draw region bbox on image and save to DEBUG_OUTPUT_DIR. Returns path or None.
    If outline_color_override is set (e.g. (0,0,255) for UI regions), all regions use that color."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    out_dir = _ensure_debug_dir()
    out_path = out_dir / output_filename
    try:
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        colors = {"text_region": (100, 200, 100), "ui_region": (200, 150, 50), "background": (120, 120, 120)}
        for r in regions:
            x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
            if outline_color_override is not None:
                color = outline_color_override
            else:
                t = r.get("type", "text_region")
                color = colors.get(t, (128, 128, 128))
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        img.save(out_path, "PNG")
        logger.info("debug: saved layout image to %s", out_path)
        return str(out_path)
    except Exception as e:
        logger.warning("debug: failed to save layout image %s: %s", out_path, e)
        return None


def save_debug_image_boxes(
    image_path: str,
    boxes: List[Dict[str, Any]],
    output_filename: str,
    color: Tuple[int, int, int] = (255, 100, 100),
) -> Optional[str]:
    """Draw text boxes on image and save to DEBUG_OUTPUT_DIR. Returns path or None."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    out_dir = _ensure_debug_dir()
    out_path = out_dir / output_filename
    try:
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for b in boxes:
            x, y, w, h = b.get("x", 0), b.get("y", 0), b.get("w", 0), b.get("h", 0)
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        img.save(out_path, "PNG")
        logger.info("debug: saved text-detect image to %s", out_path)
        return str(out_path)
    except Exception as e:
        logger.warning("debug: failed to save text-detect image %s: %s", out_path, e)
        return None


# Debug: parent=blue, child=purple, text_boxes=red, button text_boxes=green
COLOR_PARENT_REGIONS = (0, 0, 255)
COLOR_CHILD_REGIONS = (128, 0, 128)
COLOR_TEXT_BOXES = (255, 0, 0)
COLOR_BUTTON_TEXT_BOXES = (0, 255, 0)
COLOR_LINES = (0, 200, 100)
COLOR_PARAGRAPHS = (255, 255, 0)


def _dashed_rect(draw: Any, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], width: int = 2) -> None:
    dash, gap = 6, 4
    step = dash + gap
    for x in range(x1, x2, step):
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
    for x in range(x1, x2, step):
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
    for y in range(y1, y2, step):
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
    for y in range(y1, y2, step):
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)


def save_debug_image_ui_regions_hierarchy(
    image_path: str,
    regions: List[Dict[str, Any]],
    output_filename: str,
) -> Optional[str]:
    """Draw parent regions blue, child regions purple. Saves to DEBUG_OUTPUT_DIR."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    out_dir = _ensure_debug_dir()
    out_path = out_dir / output_filename
    try:
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for r in regions:
            x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
            if r.get("parent_region_id") is None:
                _dashed_rect(draw, x, y, x + w, y + h, COLOR_PARENT_REGIONS)
            else:
                _dashed_rect(draw, x, y, x + w, y + h, COLOR_CHILD_REGIONS)
        img.save(out_path, "PNG")
        logger.info("debug: saved ui-regions hierarchy image to %s", out_path)
        return str(out_path)
    except Exception as e:
        logger.warning("debug: failed to save ui-regions image %s: %s", out_path, e)
        return None


def save_debug_image_full_pipeline(
    image_path: str,
    regions: List[Dict[str, Any]],
    text_boxes: List[Dict[str, Any]],
    lines: List[List[Dict[str, Any]]],
    paragraphs: List[List[List[Dict[str, Any]]]],
    output_filename: str,
    button_labels: Optional[Dict[int, str]] = None,
    skip_region_ids: Optional[Set[int]] = None,
) -> Optional[str]:
    """
    Draw: parent=blue, child=purple (skip empty parents if skip_region_ids); button labels;
    green=text_boxes inside button/badge/pill/input_like, red=text_region; lines/paragraphs=yellow.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    out_dir = _ensure_debug_dir()
    out_path = out_dir / output_filename
    button_labels = button_labels or {}
    skip_region_ids = skip_region_ids or set()
    try:
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "C:\\Windows\\Fonts\\arial.ttf" if Path("C:\\Windows\\Fonts\\arial.ttf").exists()
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
            )
        except Exception:
            font = ImageFont.load_default()
        for rid, r in enumerate(regions):
            if rid in skip_region_ids:
                continue
            x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
            if r.get("parent_region_id") is None:
                _dashed_rect(draw, x, y, x + w, y + h, COLOR_PARENT_REGIONS)
            else:
                _dashed_rect(draw, x, y, x + w, y + h, COLOR_CHILD_REGIONS)
            if rid in button_labels and button_labels[rid]:
                tx, ty = x, y + h + 2
                draw.text((tx, ty), button_labels[rid][:40], fill=COLOR_CHILD_REGIONS, font=font)
        for b in text_boxes:
            x, y, w, h = b.get("x", 0), b.get("y", 0), b.get("w", 0), b.get("h", 0)
            rid = b.get("region_id", -1)
            reg_type = regions[rid].get("type", "") if 0 <= rid < len(regions) else ""
            color = COLOR_BUTTON_TEXT_BOXES if reg_type in ("button", "badge", "pill", "input_like") else COLOR_TEXT_BOXES
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        for line in lines:
            if not line:
                continue
            xs = [bx["x"] for bx in line]
            ys = [bx["y"] for bx in line]
            x2s = [bx["x"] + bx["w"] for bx in line]
            y2s = [bx["y"] + bx["h"] for bx in line]
            draw.rectangle([min(xs), min(ys), max(x2s), max(y2s)], outline=COLOR_LINES, width=1)
        for para in paragraphs:
            if not para:
                continue
            boxes_para = [b for ln in para for b in ln]
            xs = [b["x"] for b in boxes_para]
            ys = [b["y"] for b in boxes_para]
            x2s = [b["x"] + b["w"] for b in boxes_para]
            y2s = [b["y"] + b["h"] for b in boxes_para]
            draw.rectangle([min(xs), min(ys), max(x2s), max(y2s)], outline=COLOR_PARAGRAPHS, width=2)
        img.save(out_path, "PNG")
        logger.info("debug: saved full-pipeline image to %s", out_path)
        return str(out_path)
    except Exception as e:
        logger.warning("debug: failed to save full-pipeline image %s: %s", out_path, e)
        return None


def save_debug_image_atoms_v2(
    image_path: str,
    regions: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
    output_filename: str,
    raw_ocr_boxes: Optional[List[Dict[str, Any]]] = None,
    text_ui_links: Optional[List[Dict[str, Any]]] = None,
    lines: Optional[List[List[Dict[str, Any]]]] = None,
    independent_text_blocks: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Draw regions, atoms, raw_ocr_boxes, text_ui_links (линии OCR→atom: label=синий, content=фиолетовый), lines, independent_text_blocks.
    text_ui_links: Merge Layer v2 [{ocr_box_id, atom_id, link_type, coverage_ratio}]. standalone не рисуется."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    out_dir = _ensure_debug_dir()
    out_path = out_dir / output_filename
    try:
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
        except Exception:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
            except Exception:
                font = font_small = ImageFont.load_default()
        # 1) Regions — пунктир
        for r in regions:
            bbox = r.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                _dashed_rect(draw, x1, y1, x2, y2, COLOR_PARENT_REGIONS)
        # 2) Atoms — цвет по ui_role, пунктир для weak_*, подпись atom.type / ui_role
        ui_role_colors: Dict[str, Tuple[int, int, int]] = {
            "button": (0, 200, 0),
            "weak_button": (100, 220, 100),
            "input": (0, 150, 255),
            "weak_input": (100, 180, 255),
            "link": (0, 100, 255),
            "weak_link": (100, 150, 255),
            "control_group": (180, 180, 0),
            "pagination": (200, 100, 0),
            "form": (0, 180, 120),
            "noise": (128, 128, 128),
        }
        type_colors_fallback: Dict[str, Tuple[int, int, int]] = {
            "button": (0, 200, 0),
            "input": (0, 150, 255),
            "weak_input": (100, 180, 255),
            "title": (200, 100, 0),
            "text_block": (255, 0, 0),
            "text": (255, 0, 0),
        }
        for a in atoms:
            bbox = a.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            t = a.get("type", "text_block")
            ui_role = a.get("ui_role") or ""
            color = ui_role_colors.get(ui_role) or type_colors_fallback.get(t, (128, 128, 128))
            is_weak = ui_role in ("weak_button", "weak_link", "weak_input")
            if is_weak:
                _dashed_rect(draw, x1, y1, x2, y2, color)
            else:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"{t} / {ui_role}" if ui_role else f"{t}"
            draw.text((x1, max(0, y1 - 14)), label[:60], fill=color, font=font)
        # 3) Сырой OCR-слой: RawOCRBox — оранжевый, подпись OCR: "text" (confidence)
        color_raw_ocr = (255, 140, 0)
        if raw_ocr_boxes:
            for box in raw_ocr_boxes:
                bbox = box.get("bbox", [0, 0, 0, 0])
                if len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                draw.rectangle([x1, y1, x2, y2], outline=color_raw_ocr, width=2)
                text = (box.get("text") or "").strip()[:50]
                conf = box.get("confidence", 0)
                label = f'OCR: "{text}" ({conf:.2f})' if text else f"OCR ({conf:.2f})"
                draw.text((x1, max(0, y1 - 12)), label, fill=color_raw_ocr, font=font_small)
        # 4) Merge Layer v2: линии от центра OCR-бокса к центру атома (label=синий, content=фиолетовый; standalone не рисуется)
        if text_ui_links and raw_ocr_boxes and atoms:
            ocr_by_id = {b.get("id", ""): b for b in raw_ocr_boxes}
            atom_by_id = {a.get("id", ""): a for a in atoms}
            color_label = (0, 100, 255)
            color_content = (180, 0, 180)
            for link in text_ui_links:
                atom_id = link.get("atom_id")
                if not atom_id:
                    continue
                ocr_id = link.get("ocr_box_id", "")
                lt = link.get("link_type", "content")
                cov = link.get("coverage_ratio", 0)
                ob = ocr_by_id.get(ocr_id)
                at = atom_by_id.get(atom_id)
                if not ob or not at:
                    continue
                obbox = ob.get("bbox", [0, 0, 0, 0])
                abbox = at.get("bbox", [0, 0, 0, 0])
                if len(obbox) < 4 or len(abbox) < 4:
                    continue
                cx_ocr = int((obbox[0] + obbox[2]) / 2)
                cy_ocr = int((obbox[1] + obbox[3]) / 2)
                cx_atom = int((abbox[0] + abbox[2]) / 2)
                cy_atom = int((abbox[1] + abbox[3]) / 2)
                color = color_label if lt == "label" else color_content
                draw.line([(cx_ocr, cy_ocr), (cx_atom, cy_atom)], fill=color, width=2)
                draw.text((cx_ocr, max(0, cy_ocr - 10)), f"link: {lt} ({cov:.2f})", fill=color, font=font_small)
        # 5) OCR lines (legacy) — рамка + текст
        color_ocr_line = (255, 165, 0)
        if lines:
            for line in lines:
                for box in line:
                    x, y = box.get("x", 0), box.get("y", 0)
                    w, h = box.get("w", 0), box.get("h", 0)
                    if w <= 0 or h <= 0:
                        continue
                    draw.rectangle([x, y, x + w, y + h], outline=color_ocr_line, width=2)
                    text = (box.get("text") or "").strip()[:50]
                    if text:
                        draw.text((x, max(0, y - 12)), text, fill=color_ocr_line, font=font_small)
        # 6) Независимые текстовые блоки (legacy paragraph/label) — рамка + тип + текст (зелёный)
        color_independent = (0, 180, 80)
        if independent_text_blocks:
            for blk in independent_text_blocks:
                bbox = blk.get("bbox", [0, 0, 0, 0])
                if len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                draw.rectangle([x1, y1, x2, y2], outline=color_independent, width=2)
                t = blk.get("type", "text")
                text = (blk.get("text") or "").strip()[:60]
                label = f"{t}: {text}" if text else t
                draw.text((x1, max(0, y1 - 14)), label, fill=color_independent, font=font_small)
        img.save(out_path, "PNG")
        logger.info("debug: saved atoms_v2 image to %s", out_path)
        return str(out_path)
    except Exception as e:
        logger.warning("debug: failed to save atoms_v2 image %s: %s", out_path, e)
        return None
