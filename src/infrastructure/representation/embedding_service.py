"""
Embedding Service Interface

Placeholder for CLIP and SentenceTransformer integration.
"""

from abc import ABC, abstractmethod
from typing import List
from src.domain.interfaces.gui_detection import GUIBlock


class EmbeddingService(ABC):
    """Interface for embedding services."""

    @abstractmethod
    async def embed_visual(self, block: GUIBlock) -> List[float]:
        """Generate visual embedding (CLIP)."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate text embedding (SentenceTransformer)."""
        pass
