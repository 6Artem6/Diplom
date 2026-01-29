"""
Domain Models

Pydantic models representing core BPG entities:
- Nodes (EntityType, EntityInstance, GUIManifestation, Action, PatternNode, Rule)
- Edges (with types: cross_view, functional, temporal, etc.)
- Provenance and Confidence
- View (one View = one screenshot, critical for cross-view semantics)
"""

from .bpg_models import (
    EntityType,
    EntityInstance,
    GUIManifestation,
    Action,
    PatternNode,
    Rule,
    BPGNode,
)
from .bpg_edges import (
    BPGEdge,
    EdgeType,
    CrossViewEdge,
    FunctionalEdge,
    TemporalEdge,
    ConditionalEdge,
    CompositionalEdge,
    RoleEdge,
)
from .provenance import Provenance, Confidence, InferenceMethod
from .view import View
from .embedded_manifestation import EmbeddedManifestation

__all__ = [
    # Nodes
    "EntityType",
    "EntityInstance",
    "GUIManifestation",
    "Action",
    "PatternNode",
    "Rule",
    "BPGNode",
    # Edges
    "BPGEdge",
    "EdgeType",
    "CrossViewEdge",
    "FunctionalEdge",
    "TemporalEdge",
    "ConditionalEdge",
    "CompositionalEdge",
    "RoleEdge",
    # Metadata
    "Provenance",
    "Confidence",
    "InferenceMethod",
    # View
    "View",
    # Embedded Manifestation
    "EmbeddedManifestation",
]
