"""
Entity Linking Service Interface

Handles cross-view entity linking using multimodal similarity.

Critical semantics:
- Cross-view linking = linking entities ACROSS different views
- Same-view similarity is NOT cross-view (clustering within view)
- Cross_viewEdge is only created if source.view_id != target.view_id
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from uuid import UUID

from ..models.bpg_models import EntityInstance, GUIManifestation
from ..models.bpg_edges import CrossViewEdge
from ..models.embedded_manifestation import EmbeddedManifestation


class EntityLinkingService(ABC):
    """
    Interface for cross-view entity linking.

    Domain rationale:
    - Core domain logic: "same entity, different views"
    - Enables robust entity tracking across UI states
    - LLM agents need this for understanding entity persistence
    - Research: enables user verification of cross-view correctness
    """

    @abstractmethod
    async def cluster_within_views(
        self,
        embedded_manifestations_by_view: Dict[UUID, List[EmbeddedManifestation]],
    ) -> Dict[UUID, List[List[UUID]]]:
        """
        Cluster manifestations within each view separately.

        Architecture rationale:
        - First step: find similar manifestations within same view
        - Returns clusters per view (list of manifestation ID groups)
        - This is NOT cross-view linking, just similarity within view
        - Uses EmbeddedManifestation to guarantee 1:1 correspondence

        Args:
            embedded_manifestations_by_view: EmbeddedManifestations grouped by view_id

        Returns:
            Dict mapping view_id -> list of clusters (each cluster is list of manifestation IDs)
        """
        pass

    @abstractmethod
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
        - Uses EmbeddedManifestation to guarantee 1:1 correspondence
        - MUST assert: source.view_id != target.view_id for every edge

        Args:
            embedded_manifestations: All EmbeddedManifestations
            within_view_clusters: Clusters from cluster_within_views step

        Returns:
            List of cross_view edges (only between different view_id)

        Raises:
            AssertionError: If any edge has source.view_id == target.view_id
        """
        pass

    @abstractmethod
    async def create_entity_instances(
        self,
        manifestations: List[GUIManifestation],
        cross_view_edges: List[CrossViewEdge],
        within_view_clusters: Dict[UUID, List[List[UUID]]],
    ) -> Tuple[List[EntityInstance], Dict[UUID, UUID]]:
        """
        Create EntityInstances from manifestations using cross_view edges and within-view clusters.

        Returns:
            Tuple of (entity_instances, manifestation_id -> entity_instance_id map)

        Architecture rationale:
        - Correct order: within-view clustering → cross-view linking → entity instances
        - Manifestations connected by cross_view edges OR in same within-view cluster belong to same entity
        - This ensures causal data flow

        Args:
            manifestations: GUIManifestations to group
            cross_view_edges: Edges connecting manifestations ACROSS different views
            within_view_clusters: Clusters within each view (manifestations in same cluster = same entity)
        """
        pass
