"""
Pipeline mode atoms_v2: Detectron2 (atomic UI) → CV visual regions → assign → OCR per region → text merge → logical grouping.

Не заменяет improved-full-pipeline. Включается отдельным эндпоинтом или переключателем режима.
"""

from .pipeline import run_atoms_v2_pipeline

__all__ = ["run_atoms_v2_pipeline"]
