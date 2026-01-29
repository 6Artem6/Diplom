"""
Dependency Injection

Initializes services for FastAPI endpoints.
GUI analysis backend is switchable via GUI_ANALYSIS_BACKEND (yolo_clip | pix2struct | layoutlmv3).
YOLO is only loaded for yolo_clip; CLIP is used for linking in all backends.
"""

import logging
from pathlib import Path
from typing import Optional

from src.application.config import get_gui_analysis_backend
from src.infrastructure.preprocessing import PreprocessingServiceImpl
from src.infrastructure.gui_detection import GUIDetectionServiceImpl, YOLODetectorImpl
from src.infrastructure.representation import RepresentationServiceImpl, CLIPEncoder
from src.infrastructure.storage import ChromaVectorStore, InMemoryBPGStore, BPGQueryServiceImpl
from src.infrastructure.linking import EntityLinkingServiceImpl
from src.infrastructure.bpg_construction import BPGConstructionServiceImpl
from src.infrastructure.gui_analysis import (
    Pix2StructDetectionService,
    LayoutLMv3DetectionService,
    FlowLayoutDetectionService,
)
from src.application.use_cases.bpg_pipeline import BuildBPGUseCase
from src.domain.interfaces.gui_detection import GUIDetectionService
from src.domain.interfaces.bpg_storage import BPGStorage
from src.domain.interfaces.bpg_query import BPGQueryService

logger = logging.getLogger(__name__)

_bpg_storage: Optional[BPGStorage] = None
_bpg_query_service: Optional[BPGQueryService] = None
_yolo_detector: Optional[YOLODetectorImpl] = None
_clip_encoder: Optional[CLIPEncoder] = None
_gui_detection_by_backend: dict[str, GUIDetectionService] = {}


def get_yolo_detector() -> YOLODetectorImpl:
    """Singleton UI YOLO detector; loads from UI_YOLO_MODEL_PATH, no network."""
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YOLODetectorImpl()
    return _yolo_detector


def get_clip_encoder() -> CLIPEncoder:
    """Singleton CLIP encoder; loads from TRANSFORMERS_CACHE, local_files_only."""
    global _clip_encoder
    if _clip_encoder is None:
        _clip_encoder = CLIPEncoder()
    return _clip_encoder


def get_gui_detection_service() -> GUIDetectionService:
    """
    Return GUI detection implementation for current GUI_ANALYSIS_BACKEND (cached per backend).
    """
    backend = get_gui_analysis_backend()
    if backend in _gui_detection_by_backend:
        return _gui_detection_by_backend[backend]
    if backend == "yolo_clip":
        svc = GUIDetectionServiceImpl(yolo_detector=get_yolo_detector())
    elif backend == "pix2struct":
        svc = Pix2StructDetectionService()
    elif backend == "layoutlmv3":
        svc = LayoutLMv3DetectionService()
    elif backend == "flow":
        # Flow-based layout: OCR lines + YOLO elements
        svc = FlowLayoutDetectionService(yolo_detector=get_yolo_detector())
    else:
        svc = GUIDetectionServiceImpl(yolo_detector=get_yolo_detector())
    _gui_detection_by_backend[backend] = svc
    logger.info("GUI detection backend=%s", backend)
    return svc


def ensure_models_loaded() -> None:
    """
    Load models required for selected backend.
    yolo_clip: YOLO + CLIP. pix2struct / layoutlmv3: CLIP only (for linking).
    """
    backend = get_gui_analysis_backend()
    get_clip_encoder()
    if backend == "yolo_clip":
        get_yolo_detector()
        logger.info("Models loaded (YOLO + CLIP); offline mode ok.")
    else:
        logger.info("Models loaded (CLIP for linking); backend=%s", backend)


def get_bpg_use_case(
    chroma_persist_dir: Optional[Path] = None,
) -> BuildBPGUseCase:
    """
    Initialize BPG construction use case. GUI detection backend from GUI_ANALYSIS_BACKEND.
    """
    preprocessing = PreprocessingServiceImpl()
    gui_detection = get_gui_detection_service()
    representation = RepresentationServiceImpl(embedding_service=get_clip_encoder())

    vector_store = ChromaVectorStore(persist_directory=None)
    linking = EntityLinkingServiceImpl(vector_store)
    bpg_construction = BPGConstructionServiceImpl()

    return BuildBPGUseCase(
        preprocessing=preprocessing,
        gui_detection=gui_detection,
        representation=representation,
        linking=linking,
        bpg_construction=bpg_construction,
    )


def get_bpg_storage() -> BPGStorage:
    """Get BPG storage instance (singleton)."""
    global _bpg_storage
    if _bpg_storage is None:
        _bpg_storage = InMemoryBPGStore()
    return _bpg_storage


def get_bpg_query_service() -> BPGQueryService:
    """Get BPG query service instance (singleton)."""
    global _bpg_query_service
    if _bpg_query_service is None:
        storage = get_bpg_storage()
        _bpg_query_service = BPGQueryServiceImpl(storage)
    return _bpg_query_service
