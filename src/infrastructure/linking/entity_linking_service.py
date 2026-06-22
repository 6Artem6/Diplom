"""
Entity Linking Service Implementation

Uses Chroma vector store for cross-view entity linking.

Critical semantics:
- Cross-view linking = linking entities ACROSS different views
- Same-view similarity is handled separately (cluster_within_views)
- CrossViewEdge is only created if source.view_id != target.view_id
- Uses EmbeddedManifestation to guarantee 1:1 correspondence
"""

from typing import List, Dict, Tuple
from uuid import UUID, uuid4
from collections import defaultdict
import logging
import os
import numpy as np

from src.application.ocr.ocr_normalizer import normalize_class_for_matching
from src.domain.interfaces.linking import EntityLinkingService
from src.domain.models.bpg_models import EntityInstance, GUIManifestation
from src.domain.models.bpg_edges import CrossViewEdge
from src.domain.models.provenance import Provenance, Confidence, InferenceMethod
from src.domain.models.embedded_manifestation import EmbeddedManifestation
from ..storage.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class EntityLinkingServiceImpl(EntityLinkingService):
    """
    Implementation of entity linking with explicit cross-view semantics.

    Architecture rationale:
    - Separates within-view clustering from cross-view linking
    - Ensures cross_view edges only between different view_id
    - Enables user verification of cross-view correctness
    - Research: enables evaluation of linking accuracy
    - Uses EmbeddedManifestation to guarantee embeddings/metadata correspondence
    """

    def __init__(self, vector_store: ChromaVectorStore):
        """
        Initialize entity linking service.

        Args:
            vector_store: Chroma vector store for similarity search
        """
        self.vector_store = vector_store

    @staticmethod
    def _manifestation_class(man: GUIManifestation) -> str:
        return normalize_class_for_matching(
            str(man.layout_features.get("class_label", "unknown"))
        )

    @staticmethod
    def _cleaned_text(man: GUIManifestation) -> str:
        return str(man.layout_features.get("cleaned_text", "")).strip()

    @staticmethod
    def _embedding_input(man: GUIManifestation) -> str:
        return str(man.layout_features.get("embedding_input_text", "")).strip()

    @staticmethod
    def _vector_usable(vec: List[float]) -> bool:
        return bool(vec) and any(v != 0.0 for v in vec)

    async def cluster_within_views(
        self,
        embedded_manifestations_by_view: Dict[UUID, List[EmbeddedManifestation]],
    ) -> Dict[UUID, List[List[UUID]]]:
        """
        Cluster manifestations within each view separately.

        Skeleton implementation:
        - Uses vector similarity to find similar manifestations within same view
        - Returns clusters per view
        - Real implementation would use HDBSCAN for clustering
        - Uses phase="within_view" for Chroma metadata
        """
        # Clear Chroma collection for determinism
        self.vector_store.clear_collection()

        clusters_by_view: Dict[UUID, List[List[UUID]]] = {}

        for view_id, embedded_manifestations in embedded_manifestations_by_view.items():
            if not embedded_manifestations:
                continue

            if len(embedded_manifestations) < 2:
                # Single manifestation = single cluster
                clusters_by_view[view_id] = [[emb_man.manifestation.id] for emb_man in embedded_manifestations]
                logger.debug(f"EntityLinking: View {view_id} has {len(embedded_manifestations)} manifestation(s), creating single cluster")
                continue

            # Store embeddings for this view with phase="within_view"
            await self.vector_store.store_embedded_manifestations(
                embedded_manifestations,
                phase="within_view",
            )

            similarity_threshold = float(
                os.getenv("WITHIN_VIEW_SIMILARITY_THRESHOLD", "0.75")
            )
            clusters: List[List[UUID]] = []
            processed = set()

            for emb_man in embedded_manifestations:
                man_id = emb_man.manifestation.id
                if man_id in processed:
                    continue

                source_class = self._manifestation_class(emb_man.manifestation)
                similar = await self.vector_store.search_similar(
                    query_embedding=emb_man.get_embedding_vector(),
                    top_k=len(embedded_manifestations),
                    filter_metadata={"view_id": str(view_id)},
                    phase="within_view",
                )

                cluster = [man_id]
                processed.add(man_id)

                for similar_block in similar:
                    if similar_block["similarity"] < similarity_threshold:
                        continue
                    similar_man_id = UUID(similar_block["metadata"]["manifestation_id"])
                    similar_view_id = UUID(similar_block["metadata"]["view_id"])
                    if similar_view_id != view_id:
                        logger.error(
                            "EntityLinking: within-view search returned other view %s (expected %s)",
                            similar_view_id,
                            view_id,
                        )
                        continue
                    target_emb = next(
                        (e for e in embedded_manifestations if e.manifestation.id == similar_man_id),
                        None,
                    )
                    if target_emb is None:
                        continue
                    if self._manifestation_class(target_emb.manifestation) != source_class:
                        continue
                    if similar_man_id not in processed:
                        cluster.append(similar_man_id)
                        processed.add(similar_man_id)

                clusters.append(cluster)

            clusters_by_view[view_id] = clusters
            logger.info(
                f"EntityLinking: View {view_id} clustered into {len(clusters)} cluster(s) "
                f"from {len(embedded_manifestations)} manifestations"
            )

        return clusters_by_view

    async def link_cross_view(
        self,
        embedded_manifestations: List[EmbeddedManifestation],
        within_view_clusters: Dict[UUID, List[List[UUID]]],
    ) -> List[CrossViewEdge]:
        """
        Link manifestations of the same entity ACROSS different views.

        Critical constraint:
        - Cross_view edges are ONLY created if source.view_id != target.view_id
        - Same-view similarity is handled by cluster_within_views, not here
        - Uses phase="cross_view" for Chroma metadata
        - MUST assert: source.view_id != target.view_id for every edge
        """
        if not embedded_manifestations:
            logger.warning("EntityLinking: link_cross_view called with empty list")
            return []

        # Store all embeddings with phase="cross_view"
        await self.vector_store.store_embedded_manifestations(
            embedded_manifestations,
            phase="cross_view",
        )

        # Create mapping: manifestation_id -> EmbeddedManifestation
        emb_man_map = {emb_man.manifestation.id: emb_man for emb_man in embedded_manifestations}

        edges: List[CrossViewEdge] = []
        similarity_threshold = float(os.getenv("CROSS_VIEW_SIMILARITY_THRESHOLD", "0.72"))
        max_per_source_view = int(os.getenv("CROSS_VIEW_MAX_PER_SOURCE_VIEW", "1"))

        all_similarities: List[float] = []
        similarities_above_threshold: List[float] = []
        unmatched_pairs: List[tuple] = []
        rejected_reasons = {
            "same_view": 0,
            "class_mismatch": 0,
            "empty_embedding": 0,
            "below_threshold": 0,
        }
        top_k_unmatched = int(os.getenv("CROSS_VIEW_LOG_TOP_K_UNMATCHED", "10"))

        # Collect candidates; apply top-K per (source, target_view) after filtering
        candidates: List[tuple] = []

        for i, source_emb_man in enumerate(embedded_manifestations):
            source_man = source_emb_man.manifestation
            source_view_id = source_man.view_id
            source_embedding = source_emb_man.get_embedding_vector()
            source_class = self._manifestation_class(source_man)

            if not self._vector_usable(source_embedding):
                continue

            for j, target_emb_man in enumerate(embedded_manifestations):
                if i >= j:
                    continue

                target_man = target_emb_man.manifestation
                target_view_id = target_man.view_id

                if source_view_id == target_view_id:
                    rejected_reasons["same_view"] += 1
                    continue

                target_class = self._manifestation_class(target_man)
                if source_class != target_class:
                    rejected_reasons["class_mismatch"] += 1
                    logger.debug(
                        "EntityLinking: rejected class mismatch %s vs %s "
                        "(src=%s tgt=%s)",
                        source_class,
                        target_class,
                        self._embedding_input(source_man)[:60],
                        self._embedding_input(target_man)[:60],
                    )
                    continue

                target_embedding = target_emb_man.get_embedding_vector()
                if not self._vector_usable(target_embedding):
                    rejected_reasons["empty_embedding"] += 1
                    continue

                similarity = self._cosine_similarity(source_embedding, target_embedding)
                all_similarities.append(similarity)

                if similarity >= similarity_threshold:
                    similarities_above_threshold.append(similarity)
                    candidates.append(
                        (
                            similarity,
                            source_man,
                            target_man,
                            source_view_id,
                            target_view_id,
                            source_class,
                        )
                    )
                else:
                    rejected_reasons["below_threshold"] += 1
                    unmatched_pairs.append(
                        (
                            str(source_view_id),
                            str(target_view_id),
                            similarity,
                            source_class,
                            target_class,
                        )
                    )

        # Top-1 (or K) match per (source_id, target_view_id)
        best_by_pair: Dict[tuple, tuple] = {}
        for cand in candidates:
            sim, src_man, tgt_man, src_vid, tgt_vid, cls = cand
            key = (src_man.id, tgt_vid)
            if key not in best_by_pair or sim > best_by_pair[key][0]:
                best_by_pair[key] = cand

        selected = sorted(best_by_pair.values(), key=lambda c: c[0], reverse=True)
        if max_per_source_view > 0:
            trimmed: List[tuple] = []
            per_key_count: Dict[tuple, int] = defaultdict(int)
            for cand in selected:
                key = (cand[1].id, cand[4])
                if per_key_count[key] >= max_per_source_view:
                    continue
                per_key_count[key] += 1
                trimmed.append(cand)
            selected = trimmed

        processed_pairs: set = set()
        for (
            similarity,
            source_man,
            target_man,
            source_view_id,
            target_view_id,
            source_class,
        ) in selected:
            pair = tuple(sorted([source_man.id, target_man.id]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            edge = CrossViewEdge(
                id=uuid4(),
                source_id=source_man.id,
                target_id=target_man.id,
                similarity_score=similarity,
                confidence=Confidence(
                    score=min(1.0, max(0.0, similarity)),
                    method=InferenceMethod.MULTIMODAL_SIMILARITY,
                ),
                provenance=Provenance(
                    evidence_sources=[source_man.screenshot_id, target_man.screenshot_id],
                    inference_method=InferenceMethod.MULTIMODAL_SIMILARITY,
                    metadata={
                        "source_view_id": str(source_view_id),
                        "target_view_id": str(target_view_id),
                        "source_detected_element_id": str(source_man.id),
                        "target_detected_element_id": str(target_man.id),
                        "phase": "cross_view",
                        "method": "sentence_transformer_class_constrained",
                        "source_class": source_class,
                        "target_class": source_class,
                        "source_cleaned_text": self._cleaned_text(source_man),
                        "target_cleaned_text": self._cleaned_text(target_man),
                        "source_embedding_input": self._embedding_input(source_man),
                        "target_embedding_input": self._embedding_input(target_man),
                        "class_constraint": True,
                        "similarity": similarity,
                    },
                ),
            )
            edges.append(edge)
            logger.info(
                "EntityLinking: cross-view edge class=%s sim=%.3f "
                "src_input=%r tgt_input=%r",
                source_class,
                similarity,
                self._embedding_input(source_man)[:80],
                self._embedding_input(target_man)[:80],
            )

        # Log similarity distribution and top-K cross-view pair similarities
        top_k = int(os.getenv("CROSS_VIEW_LOG_TOP_K", "10"))
        if all_similarities:
            max_sim = max(all_similarities)
            mean_sim = sum(all_similarities) / len(all_similarities)
            count_above = len(similarities_above_threshold)
            sorted_sims = sorted(all_similarities, reverse=True)[:top_k]
            logger.info(
                "EntityLinking: Similarity distribution - "
                "max=%.3f, mean=%.3f, count_above_threshold=%d/%d (threshold=%.2f)",
                max_sim, mean_sim, count_above, len(all_similarities), similarity_threshold,
            )
            logger.info(
                "EntityLinking: Top-%d cross-view similarities: %s",
                top_k,
                [f"{s:.3f}" for s in sorted_sims],
            )
            logger.info(
                "EntityLinking: Rejection reasons - "
                "same_view=%d, class_mismatch=%d, empty_embedding=%d, below_threshold=%d",
                rejected_reasons["same_view"],
                rejected_reasons["class_mismatch"],
                rejected_reasons["empty_embedding"],
                rejected_reasons["below_threshold"],
            )
            if unmatched_pairs:
                top_unmatched = sorted(unmatched_pairs, key=lambda x: x[2], reverse=True)[:top_k_unmatched]
                logger.info(
                    "EntityLinking: Top-%d unmatched cross-view pairs (below threshold): %s",
                    top_k_unmatched,
                    [(v1, v2, f"{s:.3f}", c1, c2) for (v1, v2, s, c1, c2) in top_unmatched],
                )
                if len(edges) == 0:
                    logger.warning(
                        "EntityLinking: No cross-view edges; %d pairs below threshold. "
                        "Consider lowering CROSS_VIEW_SIMILARITY_THRESHOLD or check overlay PNG.",
                        len(unmatched_pairs),
                    )
        else:
            logger.warning("EntityLinking: No cross-view similarity computations (views or blocks missing?)")

        logger.info(
            f"EntityLinking: Created {len(edges)} cross-view edge(s) "
            f"from {len(embedded_manifestations)} manifestations "
            f"(threshold={similarity_threshold})"
        )

        return edges

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector (L2-normalized)
            vec2: Second vector (L2-normalized)
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if len(vec1) != len(vec2):
            logger.warning(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
            return 0.0
        
        # Dot product (since vectors are L2-normalized, this is cosine similarity)
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return max(0.0, min(1.0, dot_product))  # Clamp to [0, 1]

    async def create_entity_instances(
        self,
        manifestations: List[GUIManifestation],
        cross_view_edges: List[CrossViewEdge],
        within_view_clusters: Dict[UUID, List[List[UUID]]],
    ) -> Tuple[List[EntityInstance], Dict[UUID, UUID]]:
        """
        Create EntityInstances from manifestations using cross_view edges and within-view clusters.

        Architecture rationale:
        - Combines within-view clusters and cross-view edges
        - Manifestations in same within-view cluster OR connected by cross-view edge = same entity
        """
        # Build graph: cross-view edges + within-view clusters
        graph = defaultdict(set)

        # Add cross-view edges
        for edge in cross_view_edges:
            graph[edge.source_id].add(edge.target_id)
            graph[edge.target_id].add(edge.source_id)

        # Add within-view clusters (all manifestations in same cluster are connected)
        for view_id, clusters in within_view_clusters.items():
            for cluster in clusters:
                for i, man_id in enumerate(cluster):
                    for j, other_man_id in enumerate(cluster):
                        if i != j:
                            graph[man_id].add(other_man_id)
                            graph[other_man_id].add(man_id)

        # Find connected components (DFS)
        visited = set()
        components = []

        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.add(node)
            for neighbor in graph.get(node, set()):
                dfs(neighbor, component)

        # Group manifestations by connected component
        man_map = {man.id: man for man in manifestations}
        for man in manifestations:
            if man.id not in visited:
                component = set()
                dfs(man.id, component)
                components.append(component)

        # Create EntityInstance for each component
        entity_instances = []
        man_to_entity: Dict[UUID, UUID] = {}
        for component in components:
            component_manifestations = [man_map[mid] for mid in component if mid in man_map]

            # Extract attributes (simplified)
            view_ids_in_component = set(man.view_id for man in component_manifestations)
            element_ids = [str(mid) for mid in component if mid in man_map]
            attributes = {
                "component_size": len(component_manifestations),
                "view_count": len(view_ids_in_component),
                "element_ids": element_ids,
            }

            instance = EntityInstance(
                id=uuid4(),
                entity_type_id=uuid4(),  # Placeholder: would infer from patterns
                attributes=attributes,
                confidence=Confidence(
                    score=0.8,
                    method=InferenceMethod.MULTIMODAL_SIMILARITY,
                ),
                provenance=Provenance(
                    evidence_sources=[man.screenshot_id for man in component_manifestations],
                    inference_method=InferenceMethod.MULTIMODAL_SIMILARITY,
                ),
            )
            entity_instances.append(instance)
            for mid in component:
                if mid in man_map:
                    man_to_entity[mid] = instance.id

            logger.debug(
                f"EntityLinking: Created EntityInstance {instance.id} "
                f"with {len(component_manifestations)} manifestations "
                f"across {len(view_ids_in_component)} view(s)"
            )

        logger.info(
            f"EntityLinking: Created {len(entity_instances)} EntityInstance(s) "
            f"from {len(manifestations)} manifestations"
        )

        return entity_instances, man_to_entity
