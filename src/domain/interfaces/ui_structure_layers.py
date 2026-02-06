"""
Интерфейсы слоёв восстановления логической структуры UI.

Каждый слой отключаем и заменяем без переписывания всего пайплайна.
CV → геометрия UI; OCR → текст; Merge → связь; Semantic → роли текста.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..models.ui_structure import (
    OCRBox,
    TextSemanticAnnotation,
    TextUILink,
    UIAtom,
    UINode,
)


class UICVLayer(ABC):
    """
    CV-слой: только геометрия и типы UI-элементов.

    Detectron2 или другая модель. Не источник текста, не классификация текста.
    """

    @abstractmethod
    def detect_atoms(self, image_path: str) -> List[UIAtom]:
        """Детекция UI-атомов по изображению. Возвращает список UIAtom."""
        pass


class UIOCRLayer(ABC):
    """
    OCR-слой: единственный источник текста.

    Запускается полностранично, независимо от CV. Текст может быть вне UI, внутри кнопок/полей.
    """

    @abstractmethod
    def detect_text_boxes(self, image_path: str) -> List[OCRBox]:
        """Полностраничный OCR. Возвращает список OCRBox."""
        pass


class UIMergeLayer(ABC):
    """
    Merge-слой: геометрическое связывание текста и UI.

    Правила: IoU/покрытие, порог, доминирующий атом. Текст остаётся в общем слое.
    """

    @abstractmethod
    def link_text_to_ui(
        self,
        ocr_boxes: List[OCRBox],
        atoms: List[UIAtom],
        coverage_threshold: float = 0.5,
    ) -> List[TextUILink]:
        """Связывание OCR-боксов с атомами по геометрии. Возвращает список TextUILink."""
        pass


class UISemanticLayer(ABC):
    """
    Semantic-слой: логические роли текста (заголовок, описание, CTA и т.д.).

    Работает с текстом, координатами и привязкой к UI. Не строит геометрию. Опциональный.
    """

    @abstractmethod
    def assign_semantic_roles(
        self,
        ocr_boxes: List[OCRBox],
        links: List[TextUILink],
        atoms: List[UIAtom],
    ) -> List[TextSemanticAnnotation]:
        """Назначение семантических ролей тексту. Возвращает список аннотаций."""
        pass


class UITreeBuilder(ABC):
    """
    Построение семантического дерева UINode из атомов, OCR и связей.

    Объединяет выходы CV, OCR, Merge (и опционально Semantic) в дерево узлов.
    """

    @abstractmethod
    def build_tree(
        self,
        atoms: List[UIAtom],
        ocr_boxes: List[OCRBox],
        links: List[TextUILink],
        semantic_annotations: Optional[List[TextSemanticAnnotation]] = None,
    ) -> List[UINode]:
        """Строит список корневых узлов дерева UI (например navbar, card, form, standalone text)."""
        pass
