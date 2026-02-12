"""
S2 — OCR Extraction (State Machine Architecture)

OCR предоставляет ТОЛЬКО текст и его координаты.
Никаких изменений geometry!

Добавляет:
- language detection (ru/en/mixed)
- text blocks with bbox
- text binding hints (какой текст к какому элементу ближе)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class OCRBlock:
    """OCR text block."""
    text: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    
    # Computed
    line_height: float = 0.0
    is_label_hint: bool = False  # looks like a label
    is_value_hint: bool = False  # looks like a value


@dataclass
class LanguageInfo:
    """Detected language information."""
    primary: str  # 'ru', 'en', 'mixed'
    ru_ratio: float  # ratio of Russian text
    en_ratio: float  # ratio of English text
    confidence: float


@dataclass
class S2Result:
    """Result of S2 — OCR Extraction."""
    ocr_blocks: List[OCRBlock]
    language: LanguageInfo
    median_line_height: float
    diagnostics: Dict[str, Any]


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

# Russian characters pattern
RU_PATTERN = re.compile(r'[а-яА-ЯёЁ]')
# English characters pattern  
EN_PATTERN = re.compile(r'[a-zA-Z]')
# Common label patterns (Russian)
RU_LABEL_PATTERNS = [
    re.compile(r'^(Имя|Фамилия|Email|Телефон|Адрес|Город|Страна|Пароль|Логин|Почта|Дата|Комментарий|Описание|Название|Номер|ФИО)', re.IGNORECASE),
    re.compile(r'\*$'),  # ends with asterisk (required)
    re.compile(r':$'),   # ends with colon
]
# Common label patterns (English)
EN_LABEL_PATTERNS = [
    re.compile(r'^(Name|Email|Phone|Address|City|Country|Password|Login|Username|Date|Comment|Description|Title|Number)', re.IGNORECASE),
    re.compile(r'\*$'),
    re.compile(r':$'),
]
# Common button patterns (Russian)
RU_BUTTON_PATTERNS = [
    re.compile(r'^(Отправить|Сохранить|Отмена|Отменить|Войти|Выйти|Далее|Назад|Добавить|Удалить|Создать|OK|Да|Нет|Закрыть|Применить|Поиск)', re.IGNORECASE),
]
# Common button patterns (English)
EN_BUTTON_PATTERNS = [
    re.compile(r'^(Submit|Save|Cancel|Sign In|Sign Out|Login|Logout|Next|Back|Add|Delete|Create|OK|Yes|No|Close|Apply|Search)', re.IGNORECASE),
]


def detect_language(text: str) -> LanguageInfo:
    """
    Detect language from text.
    
    Returns LanguageInfo with primary language and ratios.
    """
    if not text:
        return LanguageInfo(primary="unknown", ru_ratio=0.0, en_ratio=0.0, confidence=0.0)
    
    # Count character types
    ru_chars = len(RU_PATTERN.findall(text))
    en_chars = len(EN_PATTERN.findall(text))
    total_alpha = ru_chars + en_chars
    
    if total_alpha == 0:
        # Only numbers/symbols
        return LanguageInfo(primary="unknown", ru_ratio=0.0, en_ratio=0.0, confidence=0.5)
    
    ru_ratio = ru_chars / total_alpha
    en_ratio = en_chars / total_alpha
    
    if ru_ratio > 0.7:
        primary = "ru"
        confidence = ru_ratio
    elif en_ratio > 0.7:
        primary = "en"
        confidence = en_ratio
    elif ru_ratio > 0.3 and en_ratio > 0.3:
        primary = "mixed"
        confidence = max(ru_ratio, en_ratio)
    else:
        primary = "ru" if ru_ratio > en_ratio else "en"
        confidence = max(ru_ratio, en_ratio)
    
    return LanguageInfo(
        primary=primary,
        ru_ratio=ru_ratio,
        en_ratio=en_ratio,
        confidence=confidence,
    )


def is_label_text(text: str, language: str) -> bool:
    """Check if text looks like a label."""
    if not text or len(text) < 2:
        return False
    
    # Check common patterns
    patterns = RU_LABEL_PATTERNS if language == "ru" else EN_LABEL_PATTERNS
    for p in patterns:
        if p.search(text):
            return True
    
    # Short text without value characters often is a label
    if len(text) < 30 and not any(c.isdigit() for c in text):
        return True
    
    return False


def is_button_text(text: str, language: str) -> bool:
    """Check if text looks like a button."""
    if not text or len(text) < 2:
        return False
    
    patterns = RU_BUTTON_PATTERNS if language == "ru" else EN_BUTTON_PATTERNS
    for p in patterns:
        if p.search(text.strip()):
            return True
    
    return False


def is_value_text(text: str) -> bool:
    """Check if text looks like a value (email, phone, date, etc.)."""
    if not text:
        return False
    
    # Email pattern
    if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', text):
        return True
    
    # Phone pattern
    if re.match(r'^[\d\s\+\-\(\)]{7,}$', text):
        return True
    
    # Date patterns
    if re.match(r'^\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4}$', text):
        return True
    
    return False


# =============================================================================
# OCR WRAPPER
# =============================================================================

def run_ocr(
    image,
    container_bbox: List[float],
    ocr_engine: str = "easyocr",
    languages: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Run OCR on image region.
    
    Returns (raw_results, detected_ocr_language)
    """
    if languages is None:
        languages = ["ru", "en"]
    
    x1, y1, x2, y2 = int(container_bbox[0]), int(container_bbox[1]), int(container_bbox[2]), int(container_bbox[3])
    
    # Clamp to image
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return [], "unknown"
    
    crop = image[y1:y2, x1:x2]
    
    results = []
    
    if ocr_engine == "easyocr":
        try:
            import easyocr
            reader = easyocr.Reader(languages, gpu=False, verbose=False)
            ocr_results = reader.readtext(crop)
            
            for (box, text, conf) in ocr_results:
                # Convert box to [x1, y1, x2, y2] relative to original image
                pts = [(p[0], p[1]) for p in box]
                bx1 = min(p[0] for p in pts) + x1
                by1 = min(p[1] for p in pts) + y1
                bx2 = max(p[0] for p in pts) + x1
                by2 = max(p[1] for p in pts) + y1
                
                results.append({
                    "text": text,
                    "bbox": [float(bx1), float(by1), float(bx2), float(by2)],
                    "confidence": float(conf),
                })
        except ImportError:
            logger.warning("easyocr not available, skipping OCR")
        except Exception as e:
            logger.error(f"OCR error: {e}")
    
    elif ocr_engine == "tesseract":
        try:
            import pytesseract
            # Run tesseract with output type = dict
            data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT, lang='+'.join(languages))
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text:
                    continue
                conf = int(data['conf'][i]) / 100.0 if data['conf'][i] != -1 else 0.5
                
                bx = data['left'][i] + x1
                by = data['top'][i] + y1
                bw = data['width'][i]
                bh = data['height'][i]
                
                results.append({
                    "text": text,
                    "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                    "confidence": float(conf),
                })
        except ImportError:
            logger.warning("pytesseract not available, skipping OCR")
        except Exception as e:
            logger.error(f"OCR error: {e}")
    
    # Determine primary language from results
    all_text = " ".join(r["text"] for r in results)
    lang_info = detect_language(all_text)
    
    return results, lang_info.primary


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def extract_ocr(
    image_path: str,
    container_bbox: List[float],
    ocr_engine: str = "easyocr",
    precomputed_ocr: Optional[List[Dict[str, Any]]] = None,
) -> S2Result:
    """
    S2 — OCR Extraction.
    
    Extracts text with coordinates. Does NOT modify geometry!
    
    Args:
        image_path: путь к изображению
        container_bbox: bbox контейнера формы
        ocr_engine: OCR движок ('easyocr' или 'tesseract')
        precomputed_ocr: предвычисленные OCR-результаты (опционально)
    
    Returns:
        S2Result с ocr_blocks и language info
    """
    import cv2
    
    diagnostics: Dict[str, Any] = {
        "total_blocks": 0,
        "labels_detected": 0,
        "buttons_detected": 0,
        "values_detected": 0,
    }
    
    # Load image if needed
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return S2Result(
            ocr_blocks=[],
            language=LanguageInfo(primary="unknown", ru_ratio=0.0, en_ratio=0.0, confidence=0.0),
            median_line_height=20.0,
            diagnostics={"error": "could not read image"},
        )
    
    # Get OCR results
    if precomputed_ocr is not None:
        raw_results = precomputed_ocr
        all_text = " ".join(r.get("text", "") for r in raw_results)
        lang_info = detect_language(all_text)
        detected_lang = lang_info.primary
    else:
        raw_results, detected_lang = run_ocr(image, container_bbox, ocr_engine)
    
    # Process results into OCRBlock objects
    ocr_blocks = []
    line_heights = []
    
    for r in raw_results:
        text = r.get("text", "").strip()
        if not text:
            continue
        
        bbox = r.get("bbox", [])
        if len(bbox) < 4:
            continue
        
        confidence = r.get("confidence", 0.5)
        
        # Compute line height
        line_height = bbox[3] - bbox[1]
        line_heights.append(line_height)
        
        # Determine hints
        is_label = is_label_text(text, detected_lang)
        is_button = is_button_text(text, detected_lang)
        is_value = is_value_text(text)
        
        block = OCRBlock(
            text=text,
            bbox=bbox,
            confidence=confidence,
            line_height=line_height,
            is_label_hint=is_label and not is_button,
            is_value_hint=is_value,
        )
        ocr_blocks.append(block)
        
        if is_label:
            diagnostics["labels_detected"] += 1
        if is_button:
            diagnostics["buttons_detected"] += 1
        if is_value:
            diagnostics["values_detected"] += 1
    
    diagnostics["total_blocks"] = len(ocr_blocks)
    
    # Compute median line height
    if line_heights:
        sorted_h = sorted(line_heights)
        median_line_height = sorted_h[len(sorted_h) // 2]
    else:
        median_line_height = 20.0
    
    # Full language detection
    all_text = " ".join(b.text for b in ocr_blocks)
    language = detect_language(all_text)
    
    logger.info(f"S2 completed: {len(ocr_blocks)} blocks, language={language.primary} (ru={language.ru_ratio:.2f})")
    
    return S2Result(
        ocr_blocks=ocr_blocks,
        language=language,
        median_line_height=median_line_height,
        diagnostics=diagnostics,
    )


def get_ocr_blocks_as_dicts(s2_result: S2Result) -> List[Dict[str, Any]]:
    """Convert OCR blocks to dict format for S1 OCR overlap check."""
    return [
        {
            "text": b.text,
            "bbox": b.bbox,
            "confidence": b.confidence,
        }
        for b in s2_result.ocr_blocks
    ]
