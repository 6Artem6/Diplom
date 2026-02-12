"""
Unit-тесты для LeafElementDetection (Stage 1).
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MockFormRow:
    """Mock FormRow для тестов."""
    row_index: int
    row_type: str
    y_min: float
    y_max: float
    x_min: float
    x_max: float
    input_bbox: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestLeafElementDetectorImport:
    """Тесты импорта модуля."""

    def test_import_module(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import (
            run_leaf_element_detection,
            detect_button_like,
            detect_checkbox_like,
            detect_radio_like,
            detect_textarea_like,
        )
        assert callable(run_leaf_element_detection)
        assert callable(detect_button_like)
        assert callable(detect_checkbox_like)
        assert callable(detect_radio_like)
        assert callable(detect_textarea_like)

    def test_import_constants(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import (
            BUTTON_MIN_WIDTH_RATIO,
            CHECKBOX_SIZE_MIN,
            CHECKBOX_SIZE_MAX,
            RADIO_DIAMETER_MIN,
            RADIO_DIAMETER_MAX,
            TEXTAREA_HEIGHT_RATIO,
            TEXTAREA_WIDTH_RATIO,
        )
        assert BUTTON_MIN_WIDTH_RATIO == 0.4
        assert CHECKBOX_SIZE_MIN == 12
        assert CHECKBOX_SIZE_MAX == 28
        assert RADIO_DIAMETER_MIN == 12
        assert RADIO_DIAMETER_MAX == 28
        assert TEXTAREA_HEIGHT_RATIO == 1.8
        assert TEXTAREA_WIDTH_RATIO == 0.5


class TestDetectButtonLike:
    """Тесты detect_button_like."""

    def test_invalid_input(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import detect_button_like
        result = detect_button_like(None, [], [], [])
        assert result["type"] == "button"
        assert result["confidence"] == 0.0
        assert result["reason"] == "invalid_input"

    def test_width_too_small(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import detect_button_like
        # Row width = 100, container = 400, ratio = 0.25 < 0.4
        result = detect_button_like(
            MagicMock(),
            row_bbox=[0, 0, 100, 50],
            container_bbox=[0, 0, 400, 500],
            ocr_in_row=[],
        )
        assert result["confidence"] == 0.0
        assert result["reason"] == "width_too_small"


class TestDetectCheckboxLike:
    """Тесты detect_checkbox_like."""

    def test_invalid_input(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import detect_checkbox_like
        result = detect_checkbox_like(None, [], [])
        assert result["type"] == "checkbox"
        assert result["confidence"] == 0.0


class TestDetectRadioLike:
    """Тесты detect_radio_like."""

    def test_invalid_input(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import detect_radio_like
        result = detect_radio_like(None, [])
        assert result["type"] == "radio"
        assert result["confidence"] == 0.0


class TestDetectTextareaLike:
    """Тесты detect_textarea_like."""

    def test_invalid_input(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import detect_textarea_like
        result = detect_textarea_like(None, [], [], 40.0)
        assert result["type"] == "textarea"
        assert result["confidence"] == 0.0

    def test_height_too_small(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import detect_textarea_like
        # Row height = 50, median = 40, ratio = 1.25 < 1.8
        result = detect_textarea_like(
            MagicMock(),
            row_bbox=[0, 0, 300, 50],
            container_bbox=[0, 0, 400, 500],
            median_input_height=40.0,
        )
        assert result["confidence"] == 0.0
        assert "height_too_small" in result["reason"]


class TestRunLeafElementDetection:
    """Тесты run_leaf_element_detection."""

    def test_empty_rows(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import run_leaf_element_detection
        # Не должен падать на пустых входных данных
        run_leaf_element_detection([], "", [], [])

    def test_metadata_added(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import run_leaf_element_detection

        # Mock row
        row = MockFormRow(
            row_index=0,
            row_type="FIELD_VERTICAL",
            y_min=100,
            y_max=150,
            x_min=50,
            x_max=350,
            input_bbox=[150, 100, 350, 150],
        )

        # Без реального изображения — все детекторы вернут invalid_input
        run_leaf_element_detection(
            rows=[row],
            image_path="nonexistent.png",
            raw_ocr_boxes=[{"bbox": [60, 110, 140, 140], "text": "Label", "confidence": 0.9}],
            container_bbox=[50, 50, 400, 500],
            median_input_height=40.0,
        )

        # metadata должен быть добавлен
        assert "leaf_candidates" in row.metadata
        assert "leaf_debug" in row.metadata
        assert isinstance(row.metadata["leaf_candidates"], list)
        assert isinstance(row.metadata["leaf_debug"], dict)

    def test_row_type_not_modified(self):
        """Критический тест: row_type не должен измениться."""
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import run_leaf_element_detection

        row = MockFormRow(
            row_index=0,
            row_type="FIELD_VERTICAL",
            y_min=100,
            y_max=150,
            x_min=50,
            x_max=350,
            input_bbox=[150, 100, 350, 150],
        )

        original_row_type = row.row_type
        original_y_min = row.y_min
        original_y_max = row.y_max
        original_input_bbox = row.input_bbox

        run_leaf_element_detection(
            rows=[row],
            image_path="test.png",
            raw_ocr_boxes=[],
            container_bbox=[0, 0, 400, 500],
        )

        # Инварианты: ничего не изменилось
        assert row.row_type == original_row_type, "row_type изменился!"
        assert row.y_min == original_y_min, "y_min изменился!"
        assert row.y_max == original_y_max, "y_max изменился!"
        assert row.input_bbox == original_input_bbox, "input_bbox изменился!"


class TestOcrHelpers:
    """Тесты вспомогательных функций для OCR."""

    def test_ocr_in_bbox(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import _ocr_in_bbox

        ocr_boxes = [
            {"bbox": [100, 100, 150, 130], "text": "inside"},
            {"bbox": [400, 400, 450, 430], "text": "outside"},
        ]
        bbox = [50, 50, 200, 200]

        result = _ocr_in_bbox(ocr_boxes, bbox)
        assert len(result) == 1
        assert result[0]["text"] == "inside"

    def test_is_ocr_centered(self):
        from src.infrastructure.atoms_v2.experimental_v2.leaf_element_detector import _is_ocr_centered

        # OCR центрирован
        ocr_boxes = [{"bbox": [90, 100, 110, 130]}]  # center = 100
        bbox = [50, 50, 150, 150]  # center = 100
        assert _is_ocr_centered(ocr_boxes, bbox) is True

        # OCR не центрирован
        ocr_boxes = [{"bbox": [10, 100, 30, 130]}]  # center = 20
        assert _is_ocr_centered(ocr_boxes, bbox) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
