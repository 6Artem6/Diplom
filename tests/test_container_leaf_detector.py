"""
Unit-тесты для ContainerLeafDetection (Stage 1.1).
"""

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


class TestContainerLeafDetectorImport:
    """Тесты импорта модуля."""

    def test_import_module(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import (
            run_container_leaf_detection,
            update_container_leaf_with_rows,
            DEBUG_CONTAINER_LEAF,
        )
        assert callable(run_container_leaf_detection)
        assert callable(update_container_leaf_with_rows)
        assert DEBUG_CONTAINER_LEAF is True

    def test_import_constants(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import (
            CONTAINER_BUTTON_MIN_WIDTH_RATIO,
            CONTAINER_CHECKBOX_SIZE_MIN,
            CONTAINER_TEXTAREA_HEIGHT_RATIO,
            IOU_THRESHOLD_INSIDE_ROW,
        )
        assert CONTAINER_BUTTON_MIN_WIDTH_RATIO == 0.25
        assert CONTAINER_CHECKBOX_SIZE_MIN == 10
        assert CONTAINER_TEXTAREA_HEIGHT_RATIO == 1.5
        assert IOU_THRESHOLD_INSIDE_ROW == 0.5


class TestRunContainerLeafDetection:
    """Тесты run_container_leaf_detection."""

    def test_empty_input(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import run_container_leaf_detection
        result = run_container_leaf_detection([], "", [])
        assert result == {"all_candidates": [], "inside_rows": [], "outside_rows": []}

    def test_invalid_container_bbox(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import run_container_leaf_detection
        result = run_container_leaf_detection([0, 0], "test.png", [])
        assert result["all_candidates"] == []

    def test_result_structure(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import run_container_leaf_detection
        result = run_container_leaf_detection([0, 0, 400, 500], "nonexistent.png", [])
        assert "all_candidates" in result
        assert "inside_rows" in result
        assert "outside_rows" in result
        assert isinstance(result["all_candidates"], list)
        assert isinstance(result["inside_rows"], list)
        assert isinstance(result["outside_rows"], list)


class TestUpdateContainerLeafWithRows:
    """Тесты update_container_leaf_with_rows."""

    def test_empty_result(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import update_container_leaf_with_rows
        result = update_container_leaf_with_rows({}, [])
        assert result == {}

    def test_no_rows(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import update_container_leaf_with_rows
        container_leaf = {
            "all_candidates": [{"type": "button", "confidence": 0.8, "bbox": [10, 10, 100, 50]}],
            "inside_rows": [],
            "outside_rows": [],
        }
        result = update_container_leaf_with_rows(container_leaf, [])
        # Без rows все кандидаты остаются в all_candidates, но inside/outside не пересчитываются
        assert result == container_leaf

    def test_candidate_inside_row(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import update_container_leaf_with_rows

        container_leaf = {
            "all_candidates": [{"type": "button", "confidence": 0.8, "bbox": [50, 100, 150, 140]}],
            "inside_rows": [],
            "outside_rows": [],
        }
        # Row совпадает с candidate — IoU = 1.0 > 0.5
        row = MockFormRow(
            row_index=0,
            row_type="FIELD_VERTICAL",
            y_min=100,
            y_max=140,
            x_min=50,
            x_max=150,
        )
        result = update_container_leaf_with_rows(container_leaf, [row])

        # IoU = 1.0 — кандидат внутри row
        assert len(result["inside_rows"]) == 1
        assert len(result["outside_rows"]) == 0

    def test_candidate_outside_row(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import update_container_leaf_with_rows

        container_leaf = {
            "all_candidates": [{"type": "button", "confidence": 0.8, "bbox": [50, 400, 150, 450]}],
            "inside_rows": [],
            "outside_rows": [],
        }
        row = MockFormRow(
            row_index=0,
            row_type="FIELD_VERTICAL",
            y_min=90,
            y_max=150,
            x_min=40,
            x_max=200,
        )
        result = update_container_leaf_with_rows(container_leaf, [row])

        # Кандидат далеко от row — outside
        assert len(result["inside_rows"]) == 0
        assert len(result["outside_rows"]) == 1


class TestIoU:
    """Тесты функции _iou."""

    def test_no_overlap(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _iou
        iou = _iou([0, 0, 10, 10], [20, 20, 30, 30])
        assert iou == 0.0

    def test_full_overlap(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _iou
        iou = _iou([0, 0, 10, 10], [0, 0, 10, 10])
        assert iou == 1.0

    def test_partial_overlap(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _iou
        iou = _iou([0, 0, 10, 10], [5, 5, 15, 15])
        # Intersection: [5,5,10,10] = 25, Union: 100 + 100 - 25 = 175
        expected = 25 / 175
        assert abs(iou - expected) < 0.01


class TestOcrFiltering:
    """Тесты фильтрации OCR-областей."""

    def test_get_reliable_ocr_bboxes(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _get_reliable_ocr_bboxes
        ocr_boxes = [
            {"bbox": [10, 10, 30, 30], "confidence": 0.9},
            {"bbox": [50, 50, 70, 70], "confidence": 0.4},  # low confidence
            {"bbox": [100, 100, 120, 120], "confidence": 0.8},
        ]
        reliable = _get_reliable_ocr_bboxes(ocr_boxes, min_confidence=0.6, padding=3)
        assert len(reliable) == 2
        # Check padding applied
        assert reliable[0] == [7, 7, 33, 33]  # 10-3, 10-3, 30+3, 30+3

    def test_is_text_region_overlap(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _is_text_region
        ocr_bboxes = [[10, 10, 30, 30], [100, 100, 120, 120]]
        # Full overlap
        assert _is_text_region([10, 10, 30, 30], ocr_bboxes, 0.3) is True

    def test_is_text_region_no_overlap(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _is_text_region
        ocr_bboxes = [[10, 10, 30, 30], [100, 100, 120, 120]]
        # No overlap
        assert _is_text_region([200, 200, 220, 220], ocr_bboxes, 0.3) is False

    def test_max_iou_with_ocr(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _max_iou_with_ocr
        ocr_bboxes = [[10, 10, 30, 30], [100, 100, 120, 120]]
        # Full overlap with first
        max_iou = _max_iou_with_ocr([10, 10, 30, 30], ocr_bboxes)
        assert max_iou == 1.0
        # No overlap
        max_iou = _max_iou_with_ocr([200, 200, 220, 220], ocr_bboxes)
        assert max_iou == 0.0


class TestGeometryFiltering:
    """Тесты геометрической фильтрации checkbox/radio."""

    def test_is_letter_like_high_fill_ratio(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _is_letter_like
        metrics = {
            "fill_ratio": 0.6, "inner_contours": 0, "edge_density": 0.1,
            "aspect_ratio": 1.0, "symmetry_score": 0.9, "max_dimension": 20
        }
        is_letter, reason = _is_letter_like(metrics)
        assert is_letter is True
        assert "fill_ratio" in reason

    def test_is_letter_like_many_contours(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _is_letter_like
        metrics = {
            "fill_ratio": 0.2, "inner_contours": 5, "edge_density": 0.1,
            "aspect_ratio": 1.0, "symmetry_score": 0.9, "max_dimension": 20
        }
        is_letter, reason = _is_letter_like(metrics)
        assert is_letter is True
        assert "inner_contours" in reason

    def test_is_letter_like_high_edge_density(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _is_letter_like
        metrics = {
            "fill_ratio": 0.2, "inner_contours": 1, "edge_density": 0.4,
            "aspect_ratio": 1.0, "symmetry_score": 0.9, "max_dimension": 20
        }
        is_letter, reason = _is_letter_like(metrics)
        assert is_letter is True
        assert "edge_density" in reason

    def test_is_letter_like_bad_aspect_ratio(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _is_letter_like
        metrics = {
            "fill_ratio": 0.2, "inner_contours": 1, "edge_density": 0.1,
            "aspect_ratio": 0.5, "symmetry_score": 0.9, "max_dimension": 20
        }
        is_letter, reason = _is_letter_like(metrics)
        assert is_letter is True
        assert "aspect_ratio" in reason

    def test_is_letter_like_low_symmetry(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _is_letter_like
        metrics = {
            "fill_ratio": 0.2, "inner_contours": 1, "edge_density": 0.1,
            "aspect_ratio": 1.0, "symmetry_score": 0.5, "max_dimension": 20
        }
        is_letter, reason = _is_letter_like(metrics)
        assert is_letter is True
        assert "symmetry_score" in reason

    def test_is_letter_like_too_big(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _is_letter_like
        metrics = {
            "fill_ratio": 0.2, "inner_contours": 1, "edge_density": 0.1,
            "aspect_ratio": 1.0, "symmetry_score": 0.9, "max_dimension": 60
        }
        is_letter, reason = _is_letter_like(metrics)
        assert is_letter is True
        assert "max_dimension" in reason

    def test_is_not_letter_like(self):
        from src.infrastructure.atoms_v2.experimental_v2.container_leaf_detector import _is_letter_like
        metrics = {
            "fill_ratio": 0.2, "inner_contours": 1, "edge_density": 0.1,
            "aspect_ratio": 1.0, "symmetry_score": 0.9, "max_dimension": 20
        }
        is_letter, reason = _is_letter_like(metrics)
        assert is_letter is False
        assert reason == ""


if __name__ == "__main__":
    # Простой запуск тестов
    import sys
    
    test_classes = [
        TestContainerLeafDetectorImport,
        TestRunContainerLeafDetection,
        TestUpdateContainerLeafWithRows,
        TestIoU,
        TestOcrFiltering,
        TestGeometryFiltering,
    ]
    
    all_passed = True
    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"✓ {cls.__name__}.{method_name}")
                except Exception as e:
                    print(f"✗ {cls.__name__}.{method_name}: {e}")
                    all_passed = False
    
    if all_passed:
        print("\n=== ALL TESTS PASSED ===")
    else:
        print("\n=== SOME TESTS FAILED ===")
        sys.exit(1)
