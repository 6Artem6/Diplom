"""
BPG Query Service Interface

Interface for querying BPG to extract relevant context for LLM agents.
"""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from .bpg_construction import BusinessProcessGraph


class BPGQueryService(ABC):
    """
    Interface for querying BPG.

    Domain rationale:
    - LLM agents need to extract relevant subgraph from BPG
    - Enables context enrichment with minimal relevant knowledge
    - Filters by confidence and provenance for quality
    """

    @abstractmethod
    async def find_relevant_context(
        self,
        bpg_id: UUID,
        query: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_confidence: float = 0.5,
    ) -> BusinessProcessGraph:
        """
        Find relevant BPG subgraph for LLM context.

        Args:
            bpg_id: BPG identifier
            query: Optional text query for semantic search
            entity_type: Optional filter by entity type name
            min_confidence: Minimum confidence threshold

        Returns:
            Subgraph with relevant nodes and edges
        """
        pass
