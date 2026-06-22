"""
OCR normalization for BPG detection pipeline.

Raw Tesseract output must not feed embeddings or cross-view matching directly.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Latin → Cyrillic lookalikes (common OCR confusion in RU UI)
_LATIN_TO_CYRILLIC = str.maketrans(
    {
        "A": "А",
        "a": "а",
        "B": "В",
        "E": "Е",
        "e": "е",
        "K": "К",
        "k": "к",
        "M": "М",
        "H": "Н",
        "O": "О",
        "o": "о",
        "P": "Р",
        "p": "р",
        "C": "С",
        "c": "с",
        "T": "Т",
        "y": "у",
        "X": "Х",
        "x": "х",
    }
)

# Cyrillic → Latin for mixed tokens that are mostly Latin UI labels
_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "А": "A",
        "а": "a",
        "В": "B",
        "Е": "E",
        "е": "e",
        "К": "K",
        "к": "k",
        "М": "M",
        "Н": "H",
        "О": "O",
        "о": "o",
        "Р": "P",
        "р": "p",
        "С": "C",
        "с": "c",
        "Т": "T",
        "У": "Y",
        "у": "y",
        "Х": "X",
        "х": "x",
    }
)

_NOISE_PATTERNS = (
    re.compile(r"^[\W_\d]+$"),
    re.compile(r"^(.)\1{4,}$"),
)


@dataclass
class OCRNormalizationResult:
    raw_text: str
    cleaned_text: str
    is_noisy: bool
    changes: list[str] = field(default_factory=list)


def _cyrillic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04FF")
    return cyr / len(letters)


def _fix_script_confusion(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    if not text:
        return text, changes

    ratio = _cyrillic_ratio(text)
    if ratio >= 0.55:
        fixed = text.translate(_LATIN_TO_CYRILLIC)
        if fixed != text:
            changes.append("latin_to_cyrillic")
        return fixed, changes
    if ratio <= 0.15 and re.search(r"[a-zA-Z]", text):
        fixed = text.translate(_CYRILLIC_TO_LATIN)
        if fixed != text:
            changes.append("cyrillic_to_latin")
        return fixed, changes
    return text, changes


def _merge_broken_tokens(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    # "С о х р а н и т ь" → attempt merge single-letter runs
    parts = text.split()
    if len(parts) >= 4 and all(len(p) == 1 for p in parts):
        merged = "".join(parts)
        changes.append("merge_single_char_tokens")
        return merged, changes

    # hyphen / newline artifacts
    merged = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    if merged != text:
        changes.append("merge_hyphen_breaks")
    return merged, changes


def _clean_whitespace(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized != text.strip():
        changes.append("whitespace_cleanup")
    return normalized, changes


def _is_noisy(cleaned: str) -> bool:
    if not cleaned:
        return False
    if len(cleaned) == 1 and not cleaned.isalnum():
        return True
    for pat in _NOISE_PATTERNS:
        if pat.match(cleaned):
            return True
    alnum = sum(1 for c in cleaned if c.isalnum())
    if len(cleaned) >= 6 and alnum / len(cleaned) < 0.35:
        return True
    return False


def normalize_ocr(text: str | None) -> OCRNormalizationResult:
    """
    Normalize raw OCR for structured element build and embeddings.

    Never drops the element — marks noisy output via is_noisy.
    """
    raw = (text or "").strip()
    if not raw:
        return OCRNormalizationResult(
            raw_text="",
            cleaned_text="",
            is_noisy=False,
            changes=[],
        )

    changes: list[str] = []
    current = raw

    current, ch = _clean_whitespace(current)
    changes.extend(ch)

    current, ch = _fix_script_confusion(current)
    changes.extend(ch)

    current, ch = _merge_broken_tokens(current)
    changes.extend(ch)

    current, ch = _clean_whitespace(current)
    changes.extend(ch)

    # Drop isolated punctuation-only fragments at edges
    trimmed = re.sub(r"^[\|\[\]\(\)\{\}\.,;:!?«»\"'`]+|[\|\[\]\(\)\{\}\.,;:!?«»\"'`]+$", "", current)
    if trimmed != current:
        changes.append("trim_edge_punctuation")
        current = trimmed.strip()

    noisy = _is_noisy(current)
    if noisy:
        changes.append("marked_noisy_ocr")

    return OCRNormalizationResult(
        raw_text=raw,
        cleaned_text=current,
        is_noisy=noisy,
        changes=changes,
    )


def build_embedding_input(class_name: str, cleaned_text: str) -> str:
    """Class-aware text for sentence-transformer (never raw OCR)."""
    cls = (class_name or "unknown").strip().lower()
    ct = (cleaned_text or "").strip()
    if ct:
        return f"{cls}: {ct}"
    return f"{cls}:"


def normalize_class_for_matching(class_name: str) -> str:
    """Align detection vs OCR-line class labels for constrained matching."""
    c = (class_name or "unknown").strip().lower()
    if c in ("text", "label"):
        return "text_block"
    return c
