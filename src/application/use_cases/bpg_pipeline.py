"""
BPG Construction Pipeline Use Case

Orchestrates the complete pipeline from screenshots to BPG.
"""

from typing import List
from pathlib import Path
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

from src.application.config import get_gui_analysis_backend
from src.domain.interfaces import (
    PreprocessingService,
    GUIDetectionService,
    RepresentationService,
    EntityLinkingService,
    BPGConstructionService,
)
from src.domain.interfaces.bpg_construction import BusinessProcessGraph


class BuildBPGRequest(BaseModel):
    """Request to build BPG from screenshots."""

    screenshot_paths: List[Path]
    clickstream_data: List[dict] = []  # Optional clickstream for temporal edges


class BuildBPGUseCase:
    """
    Use case: Build BPG from GUI screenshots.

    Domain rationale:
    - Encapsulates end-to-end pipeline orchestration
    - Separates application logic from domain logic
    - Enables testing with mock services
    - LLM runtime will query the resulting BPG for context
    """

    def __init__(
        self,
        preprocessing: PreprocessingService,
        gui_detection: GUIDetectionService,
        representation: RepresentationService,
        linking: EntityLinkingService,
        bpg_construction: BPGConstructionService,
    ):
        self.preprocessing = preprocessing
        self.gui_detection = gui_detection
        self.representation = representation
        self.linking = linking
        self.bpg_construction = bpg_construction

    async def execute(self, request: BuildBPGRequest) -> BusinessProcessGraph:
        """
        Execute the complete BPG construction pipeline.

        Pipeline steps:
        1. Preprocessing: Load screenshots, run OCR
        2. GUI Detection: Detect elements, group into blocks
        3. Representation: Generate multimodal embeddings
        4. Linking: Link entities across views
        5. BPG Construction: Build graph with edges

        Returns:
            Complete Business Process Graph
        """
        # Step 1: Preprocessing
        logger.info(f"BuildBPG: Starting pipeline with {len(request.screenshot_paths)} screenshot(s)")
        screenshots = await self.preprocessing.load_screenshots(
            request.screenshot_paths
        )
        logger.info(f"BuildBPG: Loaded {len(screenshots)} screenshot(s)")

        # Step 2: GUI Detection (flatten if backends return nested children)
        from src.domain.models.gui_block import flatten_gui_blocks
        all_blocks = []
        for screenshot in screenshots:
            blocks = await self.gui_detection.detect_gui_blocks(
                str(screenshot.image_path),
                screenshot.ocr_text,
            )
            all_blocks.extend(flatten_gui_blocks(blocks))
        backend = get_gui_analysis_backend()
        logger.info("BuildBPG: backend=%s, detected %d GUI block(s)", backend, len(all_blocks))

        # Step 2b: OCR normalization (raw → cleaned; never feed raw OCR to embeddings)
        from src.application.ocr.block_ocr_pipeline import normalize_blocks_ocr

        normalize_blocks_ocr(all_blocks)
        logger.info("BuildBPG: OCR normalization applied to %d block(s)", len(all_blocks))

        # Step 3: Representation
        embeddings = await self.representation.generate_embeddings(all_blocks)
        logger.info(f"BuildBPG: Generated {len(embeddings)} embedding(s)")
        
        # Validate: blocks and embeddings must match
        if len(all_blocks) != len(embeddings):
            raise ValueError(
                f"BuildBPG: Blocks count ({len(all_blocks)}) != embeddings count ({len(embeddings)})"
            )

        # Step 4: Entity Linking
        # Create Views from screenshots
        views_by_screenshot = self._create_views(screenshots)
        
        # Create EmbeddedManifestations (guaranteed 1:1 correspondence)
        embedded_manifestations = self._create_embedded_manifestations(
            all_blocks, embeddings, views_by_screenshot
        )
        
        # Group embedded manifestations by view_id
        embedded_manifestations_by_view = self._group_embedded_by_view(embedded_manifestations)
        
        # Step 4a: Cluster within each view separately
        within_view_clusters = await self.linking.cluster_within_views(
            embedded_manifestations_by_view
        )
        
        # Step 4b: Link across different views
        cross_view_edges = await self.linking.link_cross_view(
            embedded_manifestations, within_view_clusters
        )
        
        # Step 4c: Create entity instances from clusters and cross-view edges
        manifestations = [emb_man.manifestation for emb_man in embedded_manifestations]
        entity_instances, man_to_entity = await self.linking.create_entity_instances(
            manifestations, cross_view_edges, within_view_clusters
        )

        from src.application.detection.element_builder import (
            build_detected_element,
            assign_entity_ids_to_elements,
        )

        detected_elements = []
        for block, embedding, emb_man in zip(
            all_blocks, embeddings, embedded_manifestations
        ):
            view = views_by_screenshot.get(block.screenshot_id) or next(
                (
                    v
                    for v in views_by_screenshot.values()
                    if Path(v.screenshot_path).stem == block.screenshot_id
                ),
                None,
            )
            if not view:
                continue
            man_id = emb_man.manifestation.id
            detected_elements.append(
                build_detected_element(
                    block,
                    view,
                    man_id,
                    embedding.layout_features,
                )
            )

        detected_elements = assign_entity_ids_to_elements(
            detected_elements, man_to_entity
        )

        manifestations = [
            man.model_copy(
                update={"entity_instance_id": man_to_entity[man.id]},
            )
            if man.id in man_to_entity
            else man
            for man in manifestations
        ]

        # Step 5: BPG Construction
        actions = self._extract_actions(all_blocks)

        bpg = await self.bpg_construction.build_bpg(
            entity_instances=entity_instances,
            actions=actions,
            clickstream_data=request.clickstream_data,
            cross_view_edges=cross_view_edges,
        )

        bpg = bpg.model_copy(
            update={
                "detected_elements": detected_elements,
                "gui_manifestations": manifestations,
            }
        )

        logger.info(
            f"BuildBPG: Built BPG {bpg.id} with "
            f"{len(bpg.entity_types)} entity type(s), "
            f"{len(bpg.entity_instances)} entity instance(s), "
            f"{len(bpg.actions)} action(s), "
            f"{len(bpg.cross_view_edges)} cross-view edge(s)"
        )

        # FAIL-FAST for yolo_clip only; alt backends get WARNING
        if len(request.screenshot_paths) >= 2 and len(cross_view_edges) == 0:
            import os
            from collections import Counter
            blocks_per_screenshot = Counter(b.screenshot_id for b in all_blocks)
            threshold = float(os.getenv("CROSS_VIEW_SIMILARITY_THRESHOLD", "0.78"))
            msg = (
                "Cross-view linking: %d view(s) but 0 cross-view edges. "
                "Diagnostics: blocks_per_screenshot=%s, similarity_threshold=%.2f."
            ) % (len(request.screenshot_paths), dict(blocks_per_screenshot), threshold)
            if backend == "yolo_clip":
                logger.error("BuildBPG: %s", msg)
                raise RuntimeError(msg)
            logger.warning("BuildBPG: %s (backend=%s; insufficient data, not failing)", msg, backend)

        # Alt-backend debug PNGs: /app/debug/{bpg_id}/pix2struct|layoutlmv3/
        if backend in ("pix2struct", "layoutlmv3"):
            try:
                from src.infrastructure.gui_analysis import save_backend_debug_pngs
                for screenshot in screenshots:
                    stem = Path(screenshot.image_path).stem
                    blocks_for_view = [b for b in all_blocks if b.screenshot_id == stem]
                    save_backend_debug_pngs(
                        bpg.id, backend, str(screenshot.image_path), blocks_for_view
                    )
                logger.info("BuildBPG: Saved %s debug overlays for %d view(s)", backend, len(screenshots))
            except Exception as e:
                logger.warning("BuildBPG: Failed to save %s debug overlays: %s", backend, e)

        # Generate debug visualizations (side-effect, doesn't affect business logic)
        try:
            from src.infrastructure.visualization.debug_visualizer import DebugVisualizer
            visualizer = DebugVisualizer()
            
            # Generate colors for entity instances
            import colorsys
            entity_colors = {}
            if entity_instances:
                colors = []
                for i in range(len(entity_instances)):
                    hue = i / max(len(entity_instances), 1)
                    saturation = 0.7
                    value = 0.9
                    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                    colors.append(tuple(int(c * 255) for c in rgb))
                
                for i, ei in enumerate(entity_instances):
                    entity_colors[ei.id] = colors[i]
            
            # Visualize each view
            for view in views_by_screenshot.values():
                visualizer.visualize_view(
                    bpg_id=bpg.id,
                    view=view,
                    manifestations=manifestations,
                    entity_colors=entity_colors,
                    cross_view_edges=cross_view_edges,
                )
            
            logger.info(f"BuildBPG: Generated debug visualizations for {len(views_by_screenshot)} view(s)")
        except Exception as e:
            logger.warning(f"BuildBPG: Failed to generate visualizations (non-critical): {e}", exc_info=True)

        return bpg

    def _create_views(self, screenshots):
        """Create View objects from screenshots."""
        from src.domain.models.view import View
        from pathlib import Path
        
        # Map screenshot_id -> View
        views_by_screenshot = {}
        
        for screenshot in screenshots:
            if screenshot.screenshot_id not in views_by_screenshot:
                # Extract view_id from filename (list/details)
                screenshot_path = Path(screenshot.image_path)
                filename = screenshot_path.stem.lower()
                
                if "list" in filename:
                    view_id_name = "list"
                elif "detail" in filename:
                    view_id_name = "details"
                else:
                    view_id_name = filename
                
                view = View(
                    screenshot_id=screenshot.screenshot_id,
                    screenshot_path=str(screenshot.image_path),
                )
                views_by_screenshot[screenshot.screenshot_id] = view
                logger.debug(f"BuildBPG: Created view {view.id} for {screenshot.screenshot_id} (view_id={view_id_name})")
        
        return views_by_screenshot

    def _create_embedded_manifestations(self, blocks, embeddings, views_by_screenshot):
        """
        Create EmbeddedManifestations from blocks with view_id.
        
        Guarantees 1:1 correspondence between manifestation and embedding.
        """
        from src.domain.models.bpg_models import GUIManifestation
        from src.domain.models.embedded_manifestation import EmbeddedManifestation
        from uuid import uuid4

        embedded_manifestations = []
        for block, embedding in zip(blocks, embeddings):
            view = views_by_screenshot.get(block.screenshot_id)
            if not view:
                # Fallback: match by Path(screenshot_path).stem (blocks use stem as screenshot_id)
                view = next(
                    (v for v in views_by_screenshot.values()
                     if Path(v.screenshot_path).stem == block.screenshot_id),
                    None,
                )
            if not view:
                continue  # Skip if view not found

            # bounding_box must be Dict[str, float]; extract container_bbox if present
            bb = getattr(block, "bounding_box", None) or {}
            bbox_flat = {k: float(v) for k, v in bb.items() if k != "container_bbox" and not isinstance(v, dict)}
            container = bb.get("container_bbox")
            container_flat = None
            if isinstance(container, dict):
                container_flat = {k: float(v) for k, v in container.items()}

            manifestation = GUIManifestation(
                id=uuid4(),
                entity_instance_id=uuid4(),  # Placeholder, will be updated after clustering
                view_id=view.id,  # CRITICAL: view_id for cross-view semantics
                screenshot_id=block.screenshot_id,
                bounding_box=bbox_flat,
                container_bbox=container_flat,
                visual_embedding=embedding.visual_embedding,
                text_embedding=embedding.text_embedding,
                layout_features=embedding.layout_features,
            )
            
            # Create EmbeddedManifestation (guaranteed 1:1 correspondence)
            embedded_manifestation = EmbeddedManifestation(
                manifestation=manifestation,
                embedding=embedding,
            )
            embedded_manifestations.append(embedded_manifestation)
        
        return embedded_manifestations

    def _group_embedded_by_view(self, embedded_manifestations):
        """Group EmbeddedManifestations by view_id."""
        from collections import defaultdict
        
        grouped = defaultdict(list)
        for emb_man in embedded_manifestations:
            grouped[emb_man.manifestation.view_id].append(emb_man)
        return dict(grouped)

    def _extract_actions(self, blocks):
        """Extract actions from GUI blocks (placeholder)."""
        from src.domain.models.bpg_models import Action
        from src.domain.models.provenance import Provenance, Confidence, InferenceMethod
        from uuid import uuid4

        actions = []
        for block in blocks:
            # Placeholder: In real implementation, would detect action affordances
            if "button" in block.element_types:
                action = Action(
                    id=uuid4(),
                    action_type="click",
                    trigger_element={
                        "bounding_box": block.bounding_box,
                        "text": block.ocr_text,
                    },
                    confidence=Confidence(
                        score=0.7,
                        method=InferenceMethod.HEURISTIC,
                    ),
                    provenance=Provenance(
                        evidence_sources=[block.screenshot_id],
                        inference_method=InferenceMethod.HEURISTIC,
                    ),
                )
                actions.append(action)
        return actions
