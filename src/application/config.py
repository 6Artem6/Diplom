"""
Application-level configuration.

GUI_ANALYSIS_BACKEND switches between detection/layout backends
without changing pipeline code.
"""

import os
from typing import Literal

GUI_ANALYSIS_BACKEND_TYPE = Literal["yolo_clip", "pix2struct", "layoutlmv3", "flow"]

BACKENDS: tuple[GUI_ANALYSIS_BACKEND_TYPE, ...] = (
    "yolo_clip",
    "pix2struct",
    "layoutlmv3",
    "flow",
)
DEFAULT_BACKEND: GUI_ANALYSIS_BACKEND_TYPE = "yolo_clip"


def get_gui_analysis_backend() -> str:
    """
    Backend for GUI analysis: yolo_clip | pix2struct | layoutlmv3 | flow.
    Default: yolo_clip.
    """
    raw = os.getenv("GUI_ANALYSIS_BACKEND", DEFAULT_BACKEND).strip().lower()
    if raw in BACKENDS:
        return raw
    return DEFAULT_BACKEND

