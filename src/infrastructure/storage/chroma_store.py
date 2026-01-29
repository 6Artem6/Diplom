"""
Chroma Vector Store

Storage for GUI block embeddings and entity linking.

Critical guarantees:
- Every embedding has corresponding metadata
- Metadata includes phase ("within_view" | "cross_view")
- Collections are cleared between builds for determinism
"""

from typing import List, Optional, Dict, Any, Literal
import chromadb
from chromadb.config import Settings
import logging

from src.domain.interfaces.representation import MultimodalEmbedding
from src.domain.models.embedded_manifestation import EmbeddedManifestation

logger = logging.getLogger(__name__)

Phase = Literal["within_view", "cross_view"]


class ChromaVectorStore:
    """
    Chroma-based vector store for embeddings with strict metadata guarantees.

    Architecture rationale:
    - Stores GUI block embeddings for similarity search
    - Enables entity linking via vector similarity
    - LLM runtime can query "find similar blocks" for context
    - Research: enables evaluation of embedding quality
    - Guarantees: embeddings and metadatas are always synchronized
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize Chroma vector store.

        Args:
            persist_directory: Optional directory for persistence
        """
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )

        # Collection for GUI block embeddings
        self.blocks_collection = self.client.get_or_create_collection(
            name="gui_blocks",
            metadata={"description": "GUI block embeddings for entity linking"},
        )

    def clear_collection(self) -> None:
        """
        Clear all embeddings from collection.

        Architecture rationale:
        - Ensures determinism between builds
        - Prevents stale embeddings from previous runs
        - Critical for PoC reproducibility
        """
        try:
            # Delete and recreate collection
            self.client.delete_collection(name="gui_blocks")
            logger.info("ChromaVectorStore: Cleared gui_blocks collection")
        except Exception:
            # Collection might not exist, that's OK
            pass

        self.blocks_collection = self.client.get_or_create_collection(
            name="gui_blocks",
            metadata={"description": "GUI block embeddings for entity linking"},
        )
        logger.info("ChromaVectorStore: Recreated gui_blocks collection")

    async def store_embedded_manifestations(
        self,
        embedded_manifestations: List[EmbeddedManifestation],
        phase: Phase,
    ) -> None:
        """
        Store embedded manifestations in Chroma with strict guarantees.

        Args:
            embedded_manifestations: List of EmbeddedManifestation (guaranteed 1:1 correspondence)
            phase: "within_view" or "cross_view"

        Raises:
            ValueError: If embeddings or metadatas are empty or mismatched
        """
        if not embedded_manifestations:
            logger.warning("ChromaVectorStore: store_embedded_manifestations called with empty list")
            return

        # Extract embeddings and metadatas with guaranteed correspondence
        ids = []
        vectors = []
        metadatas = []

        view_ids = set()
        for emb_man in embedded_manifestations:
            ids.append(str(emb_man.manifestation.id))
            vectors.append(emb_man.get_embedding_vector())
            metadata = emb_man.to_chroma_metadata(phase)
            metadatas.append(metadata)
            view_ids.add(metadata["view_id"])

        # Strict validation
        if len(ids) == 0:
            raise ValueError("ChromaVectorStore: Cannot store empty embeddings list")

        if len(ids) != len(vectors) or len(ids) != len(metadatas):
            raise ValueError(
                f"ChromaVectorStore: Length mismatch - ids={len(ids)}, "
                f"vectors={len(vectors)}, metadatas={len(metadatas)}"
            )

        # Validate all embeddings are non-empty
        for i, vec in enumerate(vectors):
            if not vec or len(vec) == 0:
                raise ValueError(
                    f"ChromaVectorStore: Empty embedding at index {i} "
                    f"(manifestation_id={ids[i]})"
                )

        # Validate all metadatas are non-empty
        for i, meta in enumerate(metadatas):
            if not meta:
                raise ValueError(
                    f"ChromaVectorStore: Empty metadata at index {i} "
                    f"(manifestation_id={ids[i]})"
                )
            if "phase" not in meta:
                raise ValueError(
                    f"ChromaVectorStore: Missing 'phase' in metadata at index {i} "
                    f"(manifestation_id={ids[i]})"
                )

        # Store in Chroma
        self.blocks_collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
        )

        logger.info(
            f"ChromaVectorStore: Stored {len(ids)} embeddings "
            f"(phase={phase}, views={len(view_ids)}, view_ids={sorted(view_ids)})"
        )

    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        phase: Optional[Phase] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector (must be non-empty)
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            phase: Optional phase filter ("within_view" or "cross_view")

        Returns:
            List of similar blocks with similarity scores and metadata

        Raises:
            ValueError: If query_embedding is empty
        """
        if not query_embedding or len(query_embedding) == 0:
            raise ValueError("ChromaVectorStore: query_embedding cannot be empty")

        # Build where clause in Chroma's operator format: {"field": {"$eq": value}}
        # Multiple conditions: {"$and": [{"a": {"$eq": x}}, {"b": {"$eq": y}}]}
        conditions: List[Dict[str, Any]] = []
        if filter_metadata:
            for k, v in filter_metadata.items():
                conditions.append({k: {"$eq": v}})
        if phase:
            conditions.append({"phase": {"$eq": phase}})
        if len(conditions) == 0:
            where_clause = None
        elif len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = {"$and": conditions}

        # Query Chroma
        results = self.blocks_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
        )

        # Format results with strict validation
        similar_blocks = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, manifestation_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                
                if not metadata:
                    logger.warning(
                        f"ChromaVectorStore: Missing metadata for result {i} "
                        f"(manifestation_id={manifestation_id})"
                    )
                    continue

                similar_blocks.append({
                    "manifestation_id": manifestation_id,
                    "similarity": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                    "metadata": metadata,
                })

        logger.debug(
            f"ChromaVectorStore: Found {len(similar_blocks)} similar embeddings "
            f"(phase={phase}, filter={filter_metadata})"
        )

        return similar_blocks
