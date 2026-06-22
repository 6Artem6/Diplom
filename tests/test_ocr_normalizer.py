"""Tests for OCR normalization (class-aware pipeline)."""

from src.application.ocr.ocr_normalizer import (
    build_embedding_input,
    normalize_class_for_matching,
    normalize_ocr,
)


def test_whitespace_cleanup() -> None:
    result = normalize_ocr("  Hello   world  \n")
    assert result.cleaned_text == "Hello world"
    assert "whitespace_cleanup" in result.changes


def test_merge_single_char_tokens() -> None:
    result = normalize_ocr("С о х р а н и т ь")
    assert result.cleaned_text == "Сохранить"
    assert "merge_single_char_tokens" in result.changes


def test_empty_ocr_not_noisy() -> None:
    result = normalize_ocr("")
    assert result.cleaned_text == ""
    assert result.is_noisy is False


def test_build_embedding_input_class_aware() -> None:
    assert build_embedding_input("button", "Отправить") == "button: Отправить"
    assert build_embedding_input("input", "") == "input:"


def test_normalize_class_for_matching() -> None:
    assert normalize_class_for_matching("text") == "text_block"
    assert normalize_class_for_matching("button") == "button"
