"""
Sentence-Transformer text encoder for class-aware cleaned-text embeddings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_DIM = 384


def _default_cache_dir() -> str:
    return os.getenv("TRANSFORMERS_CACHE", os.getenv("HF_HOME", "/app/models/transformers"))


class SentenceTransformerEncoder:
    """Lazy singleton — MiniLM on cleaned class-aware text only."""

    _instance: "SentenceTransformerEncoder | None" = None

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None) -> None:
        self.model_name = model_name or os.getenv("SENTENCE_TRANSFORMER_MODEL", _DEFAULT_MODEL)
        self.cache_dir = Path(cache_dir or _default_cache_dir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._load_error: str | None = None

    @classmethod
    def get_instance(cls) -> "SentenceTransformerEncoder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error:
            return False
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "SentenceTransformerEncoder: loading %s from %s",
                self.model_name,
                self.cache_dir,
            )
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
            )
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error(
                "SentenceTransformerEncoder: failed to load model: %s. "
                "Install sentence-transformers and cache model offline.",
                e,
            )
            return False

    async def embed_text(self, text: str) -> List[float]:
        """L2-normalized embedding; empty input → zero vector."""
        stripped = (text or "").strip()
        if not stripped:
            return [0.0] * _EMBED_DIM
        if not self._ensure_model():
            return [0.0] * _EMBED_DIM
        try:
            vec = self._model.encode(  # type: ignore[union-attr]
                stripped,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            arr = np.asarray(vec, dtype=np.float32)
            return arr.tolist()
        except Exception as e:
            logger.error("SentenceTransformerEncoder: encode failed: %s", e)
            return [0.0] * _EMBED_DIM
