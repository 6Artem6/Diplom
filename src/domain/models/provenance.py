"""
Provenance and Confidence Models

First-class support for tracking evidence sources and uncertainty.
Critical for research evaluation and explainability.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class InferenceMethod(str, Enum):
    """Method used to infer a fact from GUI evidence."""

    HEURISTIC = "heuristic"
    ML_MODEL = "ml_model"
    LLM_ASSISTED = "llm_assisted"
    TEMPORAL_PATTERN = "temporal_pattern"
    MULTIMODAL_SIMILARITY = "multimodal_similarity"
    USER_FEEDBACK = "user_feedback"


class Confidence(BaseModel):
    """
    Confidence score for an inferred fact.

    Domain rationale:
    - LLM agents need to know how certain the system is about a fact
    - Runtime validation can reject low-confidence edges
    - Research evaluation requires explicit uncertainty tracking
    """

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score in [0, 1]",
    )
    method: InferenceMethod = Field(
        ...,
        description="How this confidence was computed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (e.g., similarity scores, model outputs)",
    )


class Provenance(BaseModel):
    """
    Provenance tracks where a fact came from.

    Domain rationale:
    - LLM agents need to explain "why" a constraint exists
    - Research requires traceability to original GUI evidence
    - Debugging and validation depend on evidence sources
    """

    evidence_sources: List[str] = Field(
        ...,
        description="IDs of screenshots, sessions, or other evidence",
    )
    inference_method: InferenceMethod = Field(
        ...,
        description="How this fact was inferred",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this fact was inferred",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (model versions, parameters, etc.)",
    )

    def add_evidence(self, source_id: str) -> None:
        """Add an evidence source."""
        if source_id not in self.evidence_sources:
            self.evidence_sources.append(source_id)
