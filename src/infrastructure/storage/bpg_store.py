"""
In-Memory BPG Storage

Simple in-memory storage for PoC. Can be replaced with Neo4j/PostgreSQL later.
"""

from typing import Optional
from uuid import UUID

from src.domain.interfaces.bpg_storage import BPGStorage
from src.domain.interfaces.bpg_construction import BusinessProcessGraph


class InMemoryBPGStore(BPGStorage):
    """
    In-memory BPG storage implementation.

    Architecture rationale:
    - Simple storage for PoC/research
    - Can be replaced with Neo4j/PostgreSQL without changing domain logic
    - Enables testing without database setup
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._storage: dict[UUID, BusinessProcessGraph] = {}

    async def save(self, bpg: BusinessProcessGraph) -> UUID:
        """Save BPG and return its ID."""
        self._storage[bpg.id] = bpg
        return bpg.id

    async def get(self, bpg_id: UUID) -> Optional[BusinessProcessGraph]:
        """Get BPG by ID."""
        return self._storage.get(bpg_id)

    async def list_all(self) -> list[UUID]:
        """List all BPG IDs."""
        return list(self._storage.keys())
