"""
BPG Construction Service Interface

Handles building BPG from entities, actions, and clickstreams.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field

from ..models.bpg_models import (
    EntityType,
    EntityInstance,
    GUIManifestation,
    Action,
    PatternNode,
    Rule,
)
from ..models.detected_element import DetectedElement
from ..models.bpg_edges import (
    FunctionalEdge,
    TemporalEdge,
    ConditionalEdge,
    CompositionalEdge,
    CrossViewEdge,
)


class BusinessProcessGraph(BaseModel):
    """
    Complete Business Process Graph.

    Domain rationale:
    - Runtime knowledge base for LLM agents
    - Enables context selection and constraint validation
    - Provides explainability surface
    """

    id: UUID = Field(default_factory=uuid4, description="Unique BPG identifier")
    entity_types: List[EntityType]
    entity_instances: List[EntityInstance]
    actions: List[Action]
    patterns: List[PatternNode]
    rules: List[Rule]
    edges: List[
        Union[
            FunctionalEdge,
            TemporalEdge,
            ConditionalEdge,
            CompositionalEdge,
        ]
    ]
    cross_view_edges: List[CrossViewEdge] = Field(
        default_factory=list,
        description="Cross-view edges linking manifestations of same entities",
    )
    detected_elements: List[DetectedElement] = Field(
        default_factory=list,
        description="Primary detection layer: class, bbox, text per GUI element",
    )
    gui_manifestations: List[GUIManifestation] = Field(
        default_factory=list,
        description="GUI manifestations with embeddings and layout features",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when BPG was created",
    )


class BPGConstructionService(ABC):
    """
    Interface for constructing BPG from processed data.

    Domain rationale:
    - Orchestrates final graph assembly
    - Infers functional/temporal edges from clickstreams
    - Creates patterns and rules from observed behavior
    """

    @abstractmethod
    async def build_bpg(
        self,
        entity_instances: List[EntityInstance],
        actions: List[Action],
        clickstream_data: List[dict],  # Placeholder: structured clickstream
        cross_view_edges: List[CrossViewEdge],
    ) -> BusinessProcessGraph:
        """
        Build complete BPG from entities and actions.

        Args:
            entity_instances: Detected entity instances
            actions: Detected actions
            clickstream_data: User interaction sequences

        Returns:
            Complete Business Process Graph
        """
        pass
