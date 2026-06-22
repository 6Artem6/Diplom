"""
BPG Query Service Implementation

Extracts relevant subgraph from BPG for LLM context.
"""

from typing import Optional
from uuid import UUID

from src.domain.interfaces.bpg_query import BPGQueryService
from src.domain.interfaces.bpg_storage import BPGStorage
from src.domain.interfaces.bpg_construction import BusinessProcessGraph
from src.domain.models.bpg_models import EntityType, EntityInstance, Action, Rule
from src.domain.models.bpg_edges import BPGEdge


class BPGQueryServiceImpl(BPGQueryService):
    """
    Implementation of BPG query service.

    Architecture rationale:
    - Filters BPG by confidence and entity type
    - Returns minimal relevant subgraph for LLM context
    - Enables prompt enrichment with domain knowledge
    """

    def __init__(self, bpg_storage: BPGStorage):
        """
        Initialize query service.

        Args:
            bpg_storage: BPG storage for retrieval
        """
        self.bpg_storage = bpg_storage

    async def find_relevant_context(
        self,
        bpg_id: UUID,
        query: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_confidence: float = 0.5,
    ) -> BusinessProcessGraph:
        """
        Find relevant BPG subgraph for LLM context.

        Skeleton implementation:
        - Filters by entity_type if provided
        - Filters by min_confidence
        - Returns subgraph with relevant nodes and edges
        """
        bpg = await self.bpg_storage.get(bpg_id)
        if not bpg:
            raise ValueError(f"BPG not found: {bpg_id}")

        # Filter entity types
        filtered_entity_types = [
            et for et in bpg.entity_types
            if et.confidence.score >= min_confidence
            and (entity_type is None or et.name.lower() == entity_type.lower())
        ]

        # Get entity type IDs
        entity_type_ids = {et.id for et in filtered_entity_types}

        # Filter entity instances
        filtered_instances = [
            ei for ei in bpg.entity_instances
            if ei.confidence.score >= min_confidence
            and (not entity_type_ids or ei.entity_type_id in entity_type_ids)
        ]

        instance_ids = {ei.id for ei in filtered_instances}

        # Filter actions
        filtered_actions = [
            a for a in bpg.actions
            if a.confidence.score >= min_confidence
        ]

        # Filter edges (only edges connecting filtered nodes)
        filtered_edges = [
            e for e in bpg.edges
            if e.confidence.score >= min_confidence
            and (
                (hasattr(e, 'source_id') and e.source_id in instance_ids)
                or (hasattr(e, 'target_id') and e.target_id in instance_ids)
            )
        ]

        # Filter cross_view edges
        filtered_cross_view = [
            e for e in bpg.cross_view_edges
            if e.confidence.score >= min_confidence
        ]

        # Create subgraph
        element_ids = set()
        for ei in filtered_instances:
            for eid in ei.attributes.get("element_ids", []):
                element_ids.add(str(eid))

        filtered_detected = [
            el
            for el in bpg.detected_elements
            if str(el.element_id) in element_ids
        ]
        filtered_manifestations = [
            m
            for m in bpg.gui_manifestations
            if str(m.id) in element_ids
        ]

        subgraph = BusinessProcessGraph(
            id=bpg.id,
            entity_types=filtered_entity_types,
            entity_instances=filtered_instances,
            actions=filtered_actions,
            patterns=[],  # Simplified: patterns not filtered in skeleton
            rules=[],  # Simplified: rules not filtered in skeleton
            edges=filtered_edges,
            cross_view_edges=filtered_cross_view,
            detected_elements=filtered_detected,
            gui_manifestations=filtered_manifestations,
            created_at=bpg.created_at,
        )

        return subgraph
