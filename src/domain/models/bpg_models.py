"""
BPG Node Models

Core node types for Business Process Graph.
Each node represents a business-level concept inferred from GUI.
"""

from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from .provenance import Provenance, Confidence


class EntityType(BaseModel):
    """
    Business-level concept inferred from GUI patterns.

    Domain rationale:
    - LLM agents need to understand "what" entities exist in the domain
    - EntityType provides semantic abstraction over concrete instances
    - Runtime: LLM can query "what entity types exist?" to understand domain
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Human-readable name (e.g., 'Product', 'Order')")
    description: Optional[str] = Field(
        None,
        description="Inferred description from GUI patterns",
    )
    confidence: Confidence = Field(..., description="Confidence in entity type inference")
    provenance: Provenance = Field(..., description="How this entity type was inferred")

    class Config:
        frozen = True


class EntityInstance(BaseModel):
    """
    Concrete instance of an EntityType observed in GUI.

    Domain rationale:
    - LLM agents need to reason about specific instances (e.g., "Order #123")
    - Links abstract EntityType to concrete GUI manifestations
    - Runtime: LLM can query "what instances of Product exist?" for context
    """

    id: UUID = Field(default_factory=uuid4)
    entity_type_id: UUID = Field(..., description="Reference to EntityType")
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Inferred attributes (e.g., {'id': '123', 'name': 'Widget'})",
    )
    confidence: Confidence = Field(
        ...,
        description="Confidence in instance identification",
    )
    provenance: Provenance = Field(..., description="Evidence sources for this instance")

    class Config:
        frozen = True


class GUIManifestation(BaseModel):
    """
    Visual representation of an entity in a specific GUI view.

    Domain rationale:
    - Same entity (e.g., Product #123) appears differently across screens
    - LLM agents need to understand "where" an entity appears visually
    - Cross-view linking enables robust entity tracking across UI states
    - view_id ensures cross_view edges are only between DIFFERENT views
    """

    id: UUID = Field(default_factory=uuid4)
    entity_instance_id: UUID = Field(
        ...,
        description="Reference to EntityInstance this manifestation represents",
    )
    view_id: UUID = Field(
        ...,
        description="ID of View (one View = one screenshot). Critical for cross-view semantics.",
    )
    screenshot_id: str = Field(
        ...,
        description="ID of screenshot (legacy, kept for compatibility). Use view_id for cross-view logic.",
    )
    bounding_box: Dict[str, float] = Field(
        ...,
        description="Bounding box: {'x': float, 'y': float, 'width': float, 'height': float}",
    )
    visual_embedding: Optional[List[float]] = Field(
        None,
        description="CLIP visual embedding (placeholder in skeleton)",
    )
    text_embedding: Optional[List[float]] = Field(
        None,
        description="Text embedding from OCR (placeholder in skeleton)",
    )
    layout_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Layout features (position, size, neighbors, etc.)",
    )

    class Config:
        frozen = True


class Action(BaseModel):
    """
    Action affordance inferred from GUI elements.

    Domain rationale:
    - LLM agents need to know "what actions are possible?"
    - Actions link to entities (functional edges: Action → EntityInstance)
    - Runtime: LLM validates plans against available actions in BPG
    """

    id: UUID = Field(default_factory=uuid4)
    action_type: str = Field(
        ...,
        description="Type of action (e.g., 'click', 'submit', 'select')",
    )
    trigger_element: Dict[str, Any] = Field(
        ...,
        description="GUI element that triggers this action (bounding box, text, etc.)",
    )
    confidence: Confidence = Field(..., description="Confidence in action inference")
    provenance: Provenance = Field(..., description="Evidence for this action")

    class Config:
        frozen = True


class PatternNode(BaseModel):
    """
    Higher-level workflow or reusable interaction pattern.

    Domain rationale:
    - LLM agents benefit from recognizing common patterns (e.g., "checkout flow")
    - Patterns provide reusable templates for action planning
    - Runtime: LLM can match current context to known patterns
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Pattern name (e.g., 'Checkout Flow')")
    steps: List[UUID] = Field(
        ...,
        description="Ordered references to Actions or other PatternNodes",
    )
    confidence: Confidence = Field(..., description="Confidence in pattern recognition")
    provenance: Provenance = Field(..., description="Evidence for this pattern")

    class Config:
        frozen = True


class Rule(BaseModel):
    """
    Explicit constraint or condition inferred from GUI behavior.

    Domain rationale:
    - LLM agents need to know "what constraints exist?"
    - Rules enable runtime validation of action plans
    - Explainability: LLM can cite rules when rejecting invalid actions
    """

    id: UUID = Field(default_factory=uuid4)
    rule_type: str = Field(
        ...,
        description="Type of rule (e.g., 'precondition', 'postcondition', 'invariant')",
    )
    condition: str = Field(
        ...,
        description="Human-readable condition (e.g., 'User must be authenticated')",
    )
    scope: Optional[str] = Field(
        None,
        description="Scope of rule (e.g., 'checkout', 'cart')",
    )
    confidence: Confidence = Field(..., description="Confidence in rule inference")
    provenance: Provenance = Field(..., description="Evidence for this rule")

    class Config:
        frozen = True


# Union type for all node types
BPGNode = EntityType | EntityInstance | GUIManifestation | Action | PatternNode | Rule
