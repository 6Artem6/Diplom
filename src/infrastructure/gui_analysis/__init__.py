"""
Alternative / experimental GUI-analysis backends.

Each implements GUIDetectionService and returns GUIBlock-compatible output.
Debug overlays saved under /app/debug/{bpg_id}/{backend}/.

Backends:
- pix2struct: VLM-based, JSON/line parsing (research only).
- layoutlmv3: OCR-based grouping into cards (research only).
- flow_layout: flow-based implicit containers (messages/sections) using OCR lines + YOLO.
"""

from .pix2struct_service import Pix2StructDetectionService
from .layoutlmv3_service import LayoutLMv3DetectionService
from .flow_layout_service import FlowLayoutDetectionService
from .debug_overlay import save_backend_debug_pngs

__all__ = [
    "Pix2StructDetectionService",
    "LayoutLMv3DetectionService",
    "FlowLayoutDetectionService",
    "save_backend_debug_pngs",
]
