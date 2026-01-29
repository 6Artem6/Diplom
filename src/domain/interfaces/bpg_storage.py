"""
BPG Storage Interface

Interface for persisting and retrieving Business Process Graphs.
"""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from .bpg_construction import BusinessProcessGraph


class BPGStorage(ABC):
    """
    Interface for BPG persistence.

    Domain rationale:
    - BPG must be stored for runtime queries by LLM agents
    - Enables retrieval by ID for context enrichment
    - Allows swapping storage implementations (in-memory → Neo4j → PostgreSQL)
    """

    @abstractmethod
    async def save(self, bpg: BusinessProcessGraph) -> UUID:
        """
        Save BPG and return its ID.

        Args:
            bpg: Business Process Graph to save

        Returns:
            BPG ID for later retrieval
        """
        pass

    @abstractmethod
    async def get(self, bpg_id: UUID) -> Optional[BusinessProcessGraph]:
        """
        Get BPG by ID.

        Args:
            bpg_id: BPG identifier

        Returns:
            Business Process Graph or None if not found
        """
        pass

    @abstractmethod
    async def list_all(self) -> list[UUID]:
        """
        List all BPG IDs.

        Returns:
            List of BPG IDs
        """
        pass
