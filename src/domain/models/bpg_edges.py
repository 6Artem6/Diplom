"""
BPG Edge Models

Edge types connecting nodes in Business Process Graph.
Each edge type has specific semantic meaning for LLM runtime usage.
"""

from enum import Enum
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from .provenance import Provenance, Confidence


class EdgeType(str, Enum):
    """Types of edges in BPG."""

    CROSS_VIEW = "cross_view"
    COMPOSITIONAL = "compositional"
    FUNCTIONAL = "functional"
    TEMPORAL = "temporal"
    CONDITIONAL = "conditional"
    ROLE = "role"


class BPGEdge(BaseModel):
    """
    Base edge model for BPG.

    Domain rationale:
    - Edges encode relationships critical for LLM understanding
    - Each edge type has specific runtime semantics
    - Provenance enables explainability ("why does this edge exist?")
    """

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID = Field(..., description="Source node ID")
    target_id: UUID = Field(..., description="Target node ID")
    edge_type: EdgeType = Field(..., description="Type of relationship")
    confidence: Confidence = Field(..., description="Confidence in edge inference")
    provenance: Provenance = Field(..., description="Evidence for this edge")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional edge-specific data",
    )

    class Config:
        frozen = True


class CrossViewEdge(BPGEdge):
    """
    Links different GUIManifestations of the same EntityInstance.

    Domain rationale:
    - LLM agents need to track entities across UI states
    - Enables robust entity linking when UI changes
    - Runtime: "What other views show this same entity?"
    """

    edge_type: EdgeType = Field(default=EdgeType.CROSS_VIEW)
    similarity_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Multimodal similarity score between manifestations",
    )


class FunctionalEdge(BPGEdge):
    """
    Represents action → result relation.

    Domain rationale:
    - LLM agents need to understand "what happens when I do X?"
    - Enables causal reasoning for action planning
    - Runtime: "What entities result from this action?"
    """

    edge_type: EdgeType = Field(default=EdgeType.FUNCTIONAL)
    action_id: UUID = Field(..., description="Action that causes this relation")


class TemporalEdge(BPGEdge):
    """
    Represents frequent or causal ordering from clickstreams.

    Domain rationale:
    - LLM agents benefit from understanding workflow sequences
    - Enables pattern-based action planning
    - Runtime: "What typically happens after this view?"
    """

    edge_type: EdgeType = Field(default=EdgeType.TEMPORAL)
    frequency: Optional[float] = Field(
        None,
        ge=0.0,
        description="Observed frequency in clickstreams",
    )
    temporal_order: Optional[int] = Field(
        None,
        description="Typical position in sequence",
    )


class ConditionalEdge(BPGEdge):
    """
    Represents preconditions and postconditions.

    Domain rationale:
    - LLM agents need to validate action plans against constraints
    - Enables explainable rejection of invalid actions
    - Runtime: "What conditions must be met for this action?"
    """

    edge_type: EdgeType = Field(default=EdgeType.CONDITIONAL)
    condition_type: str = Field(
        ...,
        description="'precondition' or 'postcondition'",
    )
    rule_id: Optional[UUID] = Field(
        None,
        description="Reference to Rule node if applicable",
    )


class CompositionalEdge(BPGEdge):
    """
    Represents containment or structural hierarchy.

    Domain rationale:
    - LLM agents need to understand entity relationships
    - Enables hierarchical reasoning (e.g., "Order contains Items")
    - Runtime: "What entities are part of this entity?"
    """

    edge_type: EdgeType = Field(default=EdgeType.COMPOSITIONAL)
    relationship_type: str = Field(
        ...,
        description="Type of composition (e.g., 'contains', 'belongs_to')",
    )


class RoleEdge(BPGEdge):
    """
    Assigns semantic role to GUI elements.

    Domain rationale:
    - LLM agents benefit from understanding UI element semantics
    - Enables role-based action selection
    - Runtime: "What elements have role 'primary_action'?"
    """

    edge_type: EdgeType = Field(default=EdgeType.ROLE)
    role: str = Field(
        ...,
        description="Semantic role (e.g., 'filter', 'selector', 'primary_action')",
    )
