"""
Representation Service Interface

Handles multimodal embeddings (visual, text, layout).
"""

from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

from ..models.gui_block import GUIBlock


class MultimodalEmbedding(BaseModel):
    """
    Multimodal representation of a GUI block.

    Domain rationale:
    - Enables similarity search for entity linking
    - LLM agents can query "find similar blocks" for pattern recognition
    - Chroma vector store uses these embeddings
    """

    block_id: str
    visual_embedding: List[float]  # CLIP embedding (placeholder)
    text_embedding: List[float]  # SentenceTransformer embedding (placeholder)
    layout_features: dict  # Position, size, neighbors, etc.


class RepresentationService(ABC):
    """
    Interface for generating multimodal embeddings.

    Domain rationale:
    - Abstracts CLIP/SentenceTransformer implementations
    - Enables swapping embedding models
    - Allows testing with mock embeddings
    """

    @abstractmethod
    async def generate_embeddings(
        self,
        blocks: List[GUIBlock],
    ) -> List[MultimodalEmbedding]:
        """
        Generate multimodal embeddings for GUI blocks.

        Args:
            blocks: List of GUI blocks to embed

        Returns:
            List of embeddings with visual, text, and layout features
        """
        pass
