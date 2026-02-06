"""
UI Structure Recovery — канонические форматы и контракты слоёв.

Цель: восстановление логической структуры UI по изображению как семантического дерева,
а не набора несвязанных bounding box'ов.

Разделение ответственности:
- CV (Detectron2): только геометрия и типы UI-элементов. Не источник текста.
- OCR: единственный источник текста. Полностранично, независимо от CV.
- Merge: геометрическое связывание текста и UI (IoU/покрытие).
- Semantic: логические роли текста (заголовок, описание, CTA) — отдельный слой (например LayoutLM3).

Каждый слой отключаем и заменяем без переписывания всего пайплайна.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# --- Инвариант CV-слоя (Detectron2) ---
#
# Допустимый выход CV: только id, source="detectron2", type, bbox [x1,y1,x2,y2], confidence.
# Запрещены на этом шаге: text, role, paragraph, label, lines_count, region_id,
# объединённые текстовые блоки. Текст живёт отдельно (OCR) и привязывается позже (Merge).


# --- Bbox: единый формат для геометрии ---

class BboxXYXY(BaseModel):
    """Bounding box: [x1, y1, x2, y2] в координатах изображения."""

    x1: float = Field(..., description="Left")
    y1: float = Field(..., description="Top")
    x2: float = Field(..., description="Right")
    y2: float = Field(..., description="Bottom")

    def to_xywh(self) -> Dict[str, float]:
        return {"x": self.x1, "y": self.y1, "width": self.x2 - self.x1, "height": self.y2 - self.y1}

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class BboxXYWH(BaseModel):
    """Bounding box: x, y, width, height."""

    x: float = Field(..., description="Left")
    y: float = Field(..., description="Top")
    width: float = Field(..., ge=0)
    height: float = Field(..., ge=0)

    def to_xyxy(self) -> BboxXYXY:
        return BboxXYXY(x1=self.x, y1=self.y, x2=self.x + self.width, y2=self.y + self.height)


# --- Raw UI Atom: строгий выход CV-слоя (Detectron2) ---

class RawUICVAtom(BaseModel):
    """
    Единственно допустимый формат выхода шага Detectron2 (CV-слой).

    Разрешённые поля: id, source, type, bbox, confidence.
    Запрещены: text, role, paragraph, label, lines_count, region_id, любые OCR/layout-поля.
    Наличие запрещённых полей = нарушение архитектуры.
    """

    id: str = Field(..., description="Уникальный идентификатор атома в рамках одного view")
    source: Literal["detectron2"] = Field(default="detectron2", description="Источник: только CV-слой")
    type: str = Field(
        ...,
        description="Визуальный тип: button, card, input, navbar, container, icon, checkbox, radio, dropdown, modal, text_block, unknown",
    )
    bbox: List[float] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2] в пикселях")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность детекции")

    class Config:
        extra = "forbid"  # запрет любых полей кроме перечисленных


# --- UI Atom: расширенная модель (для внутреннего использования) ---

class UIAtomType(str, Enum):
    """
    Визуальные типы UI-элементов от CV (сырая карта UI).

    Определяются только геометрией/внешним видом. Не OCR, не семантика текста.
    Карточки допустимо детектировать как контейнеры без подписей — подписи позже из OCR.
    """

    BUTTON = "button"
    INPUT = "input"
    CARD = "card"
    NAVBAR = "navbar"
    CONTAINER = "container"
    TITLE = "title"       # визуальный блок заголовка (геометрия)
    TEXT_BLOCK = "text_block"
    ICON = "icon"
    DROPDOWN = "dropdown"
    MODAL = "modal"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    UNKNOWN = "unknown"


class UIAtom(BaseModel):
    """
    Элемент сырой карты UI от CV (например Detectron2).

    Только геометрия и визуальный тип. Текст не извлекается; если внутри кнопки есть
    надпись — возвращается только бокс кнопки. Подписи и семантика — на этапах OCR и layout.
    """

    id: str = Field(..., description="Уникальный идентификатор атома в рамках одного view")
    type: UIAtomType = Field(..., description="Тип UI-элемента от CV")
    bbox: BboxXYXY = Field(..., description="Границы в координатах изображения")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность CV-модели")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Доп. данные (класс модели и т.п.)")


# --- Инвариант OCR-слоя ---
#
# OCR-слой возвращает только RawOCRBox. Запрещены: role, paragraph, lines_count,
# region_id, label, type, semantic_role и т.п. OCR ничего не знает про UI, регионы, кнопки, карточки.


# --- Raw OCR Box: строгий выход OCR-слоя ---

class RawOCRBox(BaseModel):
    """
    Единственно допустимый формат выхода шага OCR (сырой OCR).

    Разрешённые поля: id, source, bbox, text, confidence.
    Запрещены: role, paragraph, lines_count, region_id, label, type, semantic_role.
    OCR не знает про UI-типы, регионы, кнопки, карточки.
    """

    id: str = Field(..., description="Уникальный идентификатор бокса в рамках одного view")
    source: Literal["ocr"] = Field(default="ocr", description="Источник: только OCR-слой")
    bbox: List[float] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2] в пикселях")
    text: str = Field(..., description="Распознанный текст")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность OCR")

    class Config:
        extra = "forbid"


# --- OCR Box: расширенная модель (для внутреннего использования) ---

class OCRBox(BaseModel):
    """
    Текстовый бокс от OCR. Единственный источник текста в пайплайне.

    OCR запускается полностранично, независимо от CV.
    Текст может быть вне UI-элементов, внутри кнопок/полей, частью дизайна.
    """

    id: str = Field(..., description="Уникальный идентификатор бокса в рамках одного view")
    bbox: Union[BboxXYXY, BboxXYWH] = Field(..., description="Границы текста")
    text: str = Field(..., description="Распознанный текст")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность OCR")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def bbox_xyxy(self) -> BboxXYXY:
        if isinstance(self.bbox, BboxXYXY):
            return self.bbox
        return self.bbox.to_xyxy()


# --- Merge Layer v2: связь текста и UI (только геометрия) ---
#
# Инварианты: atom_id = None → текст вне UI; OCR-бокс всегда остаётся самостоятельным;
# связь не удаляет и не перемещает текст. Атомы и raw_ocr_boxes не модифицируются.


class TextUILinkType(str, Enum):
    """Тип связи текстового бокса с UI-элементом (эвристика, без ML)."""

    LABEL = "label"       # подпись элемента (кнопка, поле, title)
    CONTENT = "content"   # содержимое элемента (карточка, контейнер)
    STANDALONE = "standalone"  # текст не привязан к одному элементу или вне UI


class TextUILink(BaseModel):
    """
    Связь OCR-бокса с UI-атомом по геометрии (Merge Layer v2).

    Только геометрия: не группирует текст, не вводит семантику, не модифицирует атомы/OCR.
    atom_id = None → standalone (текст вне UI).
    """

    ocr_box_id: str = Field(..., description="ID RawOCRBox")
    atom_id: Optional[str] = Field(None, description="ID UIAtom; None = standalone")
    link_type: Literal["label", "content", "standalone"] = Field(
        ...,
        description="Эвристика: button/input/title→label; card/container→content; иначе content",
    )
    coverage_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="intersection(ocr_box, atom_bbox) / area(ocr_box)",
    )

    class Config:
        extra = "forbid"


# --- Semantic: логические роли текста (опциональный слой) ---

class TextSemanticRole(str, Enum):
    """Семантическая роль текста. Определяется отдельным слоем (например LayoutLM3), не CV."""

    TITLE = "title"
    DESCRIPTION = "description"
    CAPTION = "caption"
    CTA = "cta"           # call-to-action
    BODY = "body"
    LABEL = "label"      # подпись поля/кнопки
    PLACEHOLDER = "placeholder"
    FREE_TEXT = "free_text"
    UNKNOWN = "unknown"


class TextSemanticAnnotation(BaseModel):
    """
    Семантическая интерпретация текста (заголовок, описание, CTA и т.д.).

    Отдельный слой: работает с текстом, координатами и привязкой к UI-элементу.
    Detectron2 эту роль не выполняет.
    """

    ocr_box_id: str = Field(...)
    role: TextSemanticRole = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --- Итоговое семантическое дерево UI ---

class UINode(BaseModel):
    """
    Узел семантического дерева UI.

    Объединяет геометрию (CV), текст (OCR, привязанный через merge) и опционально роли (semantic).
    Используется для анализа, экспорта, автогенерации интерфейса.
    """

    id: str = Field(..., description="Уникальный ID узла")
    type: str = Field(..., description="Тип: button, input, card, navbar, panel, form, paragraph, label, ...")
    bbox: Optional[BboxXYXY] = Field(None, description="Границы узла (агрегат детей или атома)")
    label: str = Field("", description="Подпись элемента: текст, привязанный как label/content")
    text_content: List[str] = Field(
        default_factory=list,
        description="Весь текст, привязанный к этому узлу (не теряется из общего слоя)",
    )
    semantic_roles: List[TextSemanticRole] = Field(
        default_factory=list,
        description="Роли текста внутри узла (если включён semantic слой)",
    )
    children: List["UINode"] = Field(default_factory=list, description="Дочерние узлы")
    atom_id: Optional[str] = Field(None, description="ID UIAtom, если узел построен от атома")
    metadata: Dict[str, Any] = Field(default_factory=dict)


UINode.model_rebuild()
