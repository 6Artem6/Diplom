"""
Модели данных для experimental_multilevel_v2.

Регионы, скелет формы, слоты и граф — без привязки к baseline (Macro/Meso/Micro).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

# --- Level 0: глобальная ориентация ---
# Сегмент страницы по цвету/плотности (не bbox атомов)
@dataclass
class PageSegment:
    """Крупная визуальная область страницы (цвет, плотность, отступы)."""
    y_min: float
    y_max: float
    x_min: float
    x_max: float
    dominant_bgr: Tuple[int, int, int]
    content_density: float  # 0..1
    is_likely_gap: bool


# --- Form Container First (ТЗ): контейнер формы — источник истины ---
@dataclass
class FormContainer:
    """Замкнутый контейнер формы (card/panel). Все строки/слоты/поля только внутри bbox."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormColumn:
    """Колонка формы (вертикальная ось)."""
    col_index: int
    x_min: float
    x_max: float


# --- Level 1: семантические регионы ---
RegionType = Literal["header", "footer", "sidebar", "content", "form_area", "unknown"]


@dataclass
class Region:
    """Семантический регион с типом и границами."""
    region_type: RegionType
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Level 2: скелет формы ---
RowType = Literal[
    "FIELD_HORIZONTAL",
    "FIELD_VERTICAL",
    "FIELD_INPUT_ONLY",
    "TEXTAREA",
    "ACTION",
    "TEXT",
    "HEADER",
    "FIELD",
    "SPACER",
]
HeightMode = Literal["fixed", "adaptive"]


@dataclass
class FormRow:
    """Строка формы (каркас). Внутренняя декомпозиция: label_bbox, input_bbox, helper_bbox по визуальным кандидатам и OCR."""
    row_index: int
    y_min: float
    y_max: float
    x_min: float
    x_max: float
    column_count: int = 1
    row_type: RowType = "FIELD_HORIZONTAL"
    height_mode: HeightMode = "adaptive"
    label_bbox: Optional[List[float]] = None
    input_bbox: Optional[List[float]] = None
    helper_bbox: Optional[List[float]] = None
    vertical_split_y: Optional[float] = None
    height_confidence: Optional[float] = None


@dataclass
class FormSkeleton:
    """Каркас формы: строки и колонки без полей. Привязка к FormContainer (не к Region)."""
    form_region: Any  # Region или FormContainer для совместимости
    rows: List[FormRow] = field(default_factory=list)
    columns: Optional[List[FormColumn]] = None  # FormColumn[] по ТЗ
    column_boundaries: Optional[List[Tuple[float, float]]] = None  # (x_min,x_max) для grid
    layout_type: Literal["vertical", "grid", "mixed"] = "vertical"


# --- Level 3: слоты ---
SlotRole = Literal["label_slot", "input_slot", "textarea_slot", "helper_slot", "action_slot"]


@dataclass
class Slot:
    """Ожидаемая зона (слот). Существует даже если поле не найдено. expected_bbox_hint — зона поиска поля."""
    slot_id: str
    role: SlotRole
    row_index: int
    column_index: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width_hint: float
    height_hint: float
    expected_bbox_hint: List[float] = field(default_factory=list)  # [x1,y1,x2,y2] для FieldLocator
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RowSlots:
    """Слоты одной строки формы."""
    row_index: int
    row_bbox: List[float]
    slots: List[Slot] = field(default_factory=list)


# --- Level 4–5: граф формы ---
@dataclass
class SlotAssignment:
    """Слот → не более одного bbox."""
    slot: Slot
    bbox: Optional[List[float]] = None
    field_type: Literal["input", "textarea"] = "input"
    confidence: float = 0.0


@dataclass
class FormGraph:
    """Граф формы: слоты, назначения, связи label→input, input→helper."""
    skeleton: FormSkeleton
    row_slots: List[RowSlots] = field(default_factory=list)
    assignments: List[SlotAssignment] = field(default_factory=list)
    label_to_input: List[Tuple[str, str]] = field(default_factory=list)  # slot_id -> slot_id
    input_to_helper: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
