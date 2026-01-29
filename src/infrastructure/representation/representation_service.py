"""
Representation Service Implementation

Real implementation with CLIP encoder.
"""

from typing import List
import logging

from src.domain.interfaces.representation import (
    RepresentationService,
    MultimodalEmbedding,
)
from src.domain.interfaces.gui_detection import GUIBlock
from .embedding_service import EmbeddingService
from .clip_encoder import CLIPEncoder

logger = logging.getLogger(__name__)


class RepresentationServiceImpl(RepresentationService):
    """
    Real implementation of representation service with CLIP.

    Architecture rationale:
    - Uses real CLIP encoder for visual embeddings
    - Generates 512-dimensional L2-normalized vectors
    - Enables cosine similarity for cross-view matching
    """

    def __init__(self, embedding_service: EmbeddingService = None):
        """
        Initialize representation service.

        Args:
            embedding_service: Optional embedding service (default: creates CLIPEncoder)
        """
        if embedding_service is None:
            # Create real CLIP encoder
            self.embedding_service = CLIPEncoder()
        else:
            self.embedding_service = embedding_service

    async def generate_embeddings(
        self,
        blocks: List[GUIBlock],
    ) -> List[MultimodalEmbedding]:
        """
        Generate CLIP embeddings for GUI blocks.

        Real implementation:
        1. Extract visual features with CLIP (crop from bounding box)
        2. L2-normalize embeddings for cosine similarity
        3. Compute layout features (position, size)
        """
        logger.info(f"Representation: Generating embeddings for {len(blocks)} blocks")
        
        embeddings = []
        for block in blocks:
            # Generate CLIP visual embedding
            visual_emb = await self.embedding_service.embed_visual(block)
            
            # Text embedding not used for cross-view matching
            text_emb = await self.embedding_service.embed_text(block.ocr_text)
            
            # Store embedding in block for later use
            block.visual_features = visual_emb

            layout_features = {
                "x": block.bounding_box.get("x", 0.0),
                "y": block.bounding_box.get("y", 0.0),
                "width": block.bounding_box.get("width", 0.0),
                "height": block.bounding_box.get("height", 0.0),
                "class_label": block.element_types[0] if block.element_types else "unknown",
            }

            embedding = MultimodalEmbedding(
                block_id=block.id,
                visual_embedding=visual_emb,
                text_embedding=text_emb,
                layout_features=layout_features,
            )
            embeddings.append(embedding)
        
        logger.info(f"Representation: Generated {len(embeddings)} embeddings")
        
        return embeddings
