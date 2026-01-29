"""
Domain Interfaces

Abstract interfaces for pipeline modules.
Implementation lives in infrastructure layer.
"""

from .preprocessing import PreprocessingService
from .gui_detection import GUIDetectionService
from .representation import RepresentationService
from .linking import EntityLinkingService
from .bpg_construction import BPGConstructionService
from .bpg_storage import BPGStorage
from .bpg_query import BPGQueryService
from .visual_layout_detector import VisualLayoutDetector, RawLayoutBox

__all__ = [
    "PreprocessingService",
    "GUIDetectionService",
    "RepresentationService",
    "EntityLinkingService",
    "BPGConstructionService",
    "BPGStorage",
    "BPGQueryService",
    "VisualLayoutDetector",
    "RawLayoutBox",
]
