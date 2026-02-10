"""
Модели логической схемы формы — без координат и bbox.

Form → Schema → Fields: схема описывает структуру формы (тип, колонки, строки, роли, слоты),
но не создаёт и не режет bbox. OCR используется только для роли строки, не для геометрии.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

# form_type: вертикальная форма (поток строк), сетка (матрица ячеек), смешанная
FormType = Literal["vertical", "grid", "mixed"]

# роль строки: поле, заголовок, кнопка, текст
RowRole = Literal["field_row", "header", "button_row", "text"]


@dataclass
class ColumnHint:
    """Подсказка по колонке: относительные позиции (0..1 от ширины card). Bbox builder использует для x1/x2."""
    x_center_ratio: float  # центр колонки, 0..1
    width_hint_ratio: float  # доля ширины card на колонку, 0..1


@dataclass
class SlotSchema:
    """Слот в строке (одна ячейка). Не содержит bbox."""
    column_index: int
    has_label: bool  # OCR подсказывает label (только для роли, не для геометрии)
    has_input: bool  # в этом слоте ожидается поле ввода
    expected_input_type: Literal["input", "textarea"] = "input"


@dataclass
class RowSchema:
    """Логическая строка схемы. Роль определяет, создавать ли поля; OCR только для роли."""
    role: RowRole
    slots: List[SlotSchema] = field(default_factory=list)
    # y_center_ratio / y_top_ratio — подсказка для bbox builder (0..1 от высоты card), опционально
    y_center_ratio: Optional[float] = None
    height_hint_ratio: Optional[float] = None


@dataclass
class FormSchema:
    """
    Логическая схема формы. Не содержит bbox.
    Вход для FieldBBoxBuilder(schema, card_bbox).
    """
    form_type: FormType
    columns: List[ColumnHint] = field(default_factory=list)  # для grid и mixed — явные колонки
    rows: List[RowSchema] = field(default_factory=list)
    flow_end_row_index: Optional[int] = None  # после этой строки (включительно) полей нет
    # Единая высота строки поля (доля высоты card), если задана — bbox builder использует
    default_row_height_ratio: Optional[float] = None
