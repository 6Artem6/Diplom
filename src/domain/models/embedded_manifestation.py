"""
Embedded Manifestation Model

Explicit structure linking GUIManifestation with its embedding.
Guarantees 1:1 correspondence and prevents metadata/embedding desynchronization.
"""

from uuid import UUID
from pydantic import BaseModel, Field, field_validator

from .bpg_models import GUIManifestation
from ..interfaces.representation import MultimodalEmbedding


class EmbeddedManifestation(BaseModel):
    """
    GUIManifestation with its corresponding embedding.

    Domain rationale:
    - Guarantees 1:1 correspondence between manifestation and embedding
    - Prevents metadata/embedding desynchronization
    - Enables type-safe passing through pipeline
    - Research: ensures reproducibility and debuggability
    """

    manifestation: GUIManifestation = Field(..., description="GUI manifestation")
    embedding: MultimodalEmbedding = Field(..., description="Corresponding multimodal embedding")

    @field_validator("embedding")
    @classmethod
    def validate_embedding_not_empty(cls, v: MultimodalEmbedding) -> MultimodalEmbedding:
        """Validate that embedding is not empty."""
        # For cross-view matching, visual_embedding is required (CLIP)
        if not v.visual_embedding or len(v.visual_embedding) == 0:
            raise ValueError("EmbeddedManifestation: visual_embedding cannot be empty (required for CLIP cross-view matching)")
        return v

    def to_chroma_metadata(self, phase: str) -> dict:
        """
        Convert to Chroma metadata format.

        Args:
            phase: "within_view" or "cross_view"

        Returns:
            Metadata dict for Chroma storage
        """
        return {
            "manifestation_id": str(self.manifestation.id),
            "view_id": str(self.manifestation.view_id),
            "screenshot_id": self.manifestation.screenshot_id,
            "phase": phase,
            "entity_instance_id": str(self.manifestation.entity_instance_id),
        }

    def get_embedding_vector(self) -> list[float]:
        """Get visual embedding vector for similarity search (CLIP)."""
        # Use visual embedding for cross-view matching (CLIP)
        if self.embedding.visual_embedding:
            return self.embedding.visual_embedding
        # Fallback to text embedding if visual not available
        return self.embedding.text_embedding
