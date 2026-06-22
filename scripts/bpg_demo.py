#!/usr/bin/env python3
"""
BPG demo: shows within-view clusters + cross-view linking step by step.
Run inside container: python3 /app/scripts/bpg_demo.py
"""
import asyncio
import json
import os
import sys

os.environ["CROSS_VIEW_SIMILARITY_THRESHOLD"] = "0.95"
sys.path.insert(0, "/app")

from pathlib import Path
from uuid import UUID, uuid4
from collections import defaultdict


async def main():
    from src.api.dependencies import get_gui_detection_service, get_clip_encoder
    from src.infrastructure.preprocessing import PreprocessingServiceImpl
    from src.infrastructure.representation import RepresentationServiceImpl
    from src.infrastructure.storage import ChromaVectorStore
    from src.infrastructure.linking import EntityLinkingServiceImpl
    from src.domain.models.gui_block import flatten_gui_blocks
    from src.domain.models.bpg_models import GUIManifestation
    from src.domain.models.embedded_manifestation import EmbeddedManifestation
    from src.domain.models.view import View

    screenshots_paths = [
        Path("/app/data/demo_forms/images/demo_form_16.png"),
        Path("/app/data/demo_forms/images/demo_form_11.png"),
    ]

    preprocessing = PreprocessingServiceImpl()
    gui_detection = get_gui_detection_service()
    representation = RepresentationServiceImpl(embedding_service=get_clip_encoder())
    vector_store = ChromaVectorStore(persist_directory=None)
    linking = EntityLinkingServiceImpl(vector_store)

    # Step 1: Load
    screenshots = await preprocessing.load_screenshots(screenshots_paths)
    print(f"=== Step 1: Loaded {len(screenshots)} screenshots ===")

    # Step 2: Detect GUI blocks
    all_blocks = []
    for s in screenshots:
        blocks = await gui_detection.detect_gui_blocks(str(s.image_path), s.ocr_text)
        all_blocks.extend(flatten_gui_blocks(blocks))
    print(f"\n=== Step 2: Detected {len(all_blocks)} GUI blocks ===")
    for b in all_blocks[:5]:
        print(f"  {b.screenshot_id} | types={b.element_types} | text={b.ocr_text[:50]}...")

    # Step 3: Embeddings
    embeddings = await representation.generate_embeddings(all_blocks)
    print(f"\n=== Step 3: Generated {len(embeddings)} embeddings ===")

    # Build Views and EmbeddedManifestations
    views = {}
    for s in screenshots:
        views[s.screenshot_id] = View(
            screenshot_id=s.screenshot_id,
            screenshot_path=str(s.image_path),
        )

    embedded_manifestations = []
    for block, emb in zip(all_blocks, embeddings):
        view = views.get(block.screenshot_id)
        if not view:
            continue
        bb = getattr(block, "bounding_box", None) or {}
        bbox_flat = {k: float(v) for k, v in bb.items() if k != "container_bbox" and not isinstance(v, dict)}
        man = GUIManifestation(
            id=uuid4(),
            entity_instance_id=uuid4(),
            view_id=view.id,
            screenshot_id=block.screenshot_id,
            bounding_box=bbox_flat,
            visual_embedding=emb.visual_embedding,
            text_embedding=emb.text_embedding,
            layout_features=emb.layout_features,
        )
        embedded_manifestations.append(EmbeddedManifestation(manifestation=man, embedding=emb))

    emb_by_view = defaultdict(list)
    for em in embedded_manifestations:
        emb_by_view[em.manifestation.view_id].append(em)

    # Step 4a: Within-view clustering (WHAT YOU WANT TO SEE)
    within_view_clusters = await linking.cluster_within_views(dict(emb_by_view))

    print(f"\n{'='*60}")
    print(f"=== Step 4a: WITHIN-VIEW CLUSTERS ===")
    print(f"{'='*60}")
    for view_id, clusters in within_view_clusters.items():
        view_name = next((v.screenshot_id for v in views.values() if v.id == view_id), "?")
        print(f"\n  Screen: {view_name}")
        print(f"  Total clusters: {len(clusters)}")
        for i, cluster in enumerate(clusters):
            # Find manifestation details for this cluster
            cluster_elements = []
            for em in embedded_manifestations:
                if em.manifestation.id in cluster:
                    lf = em.manifestation.layout_features
                    cluster_elements.append({
                        "class": lf.get("class_label", "?"),
                        "bbox": em.manifestation.bounding_box,
                    })
            classes = [e["class"] for e in cluster_elements]
            print(f"    Cluster {i}: {len(cluster)} elements | classes={classes}")

    # Step 4b: Cross-view linking
    cross_view_edges = await linking.link_cross_view(embedded_manifestations, within_view_clusters)

    print(f"\n{'='*60}")
    print(f"=== Step 4b: CROSS-VIEW EDGES ({len(cross_view_edges)}) ===")
    print(f"{'='*60}")
    for e in sorted(cross_view_edges, key=lambda x: x.similarity_score, reverse=True)[:15]:
        meta = e.provenance.metadata
        print(f"  sim={e.similarity_score:.3f} | {meta.get('source_class','?')} <-> {meta.get('target_class','?')}")

    # Step 4c: Entity instances
    manifestations = [em.manifestation for em in embedded_manifestations]
    entity_instances, _man_map = await linking.create_entity_instances(
        manifestations, cross_view_edges, within_view_clusters
    )

    print(f"\n{'='*60}")
    print(f"=== Step 4c: ENTITY INSTANCES ({len(entity_instances)}) ===")
    print(f"{'='*60}")
    for ei in entity_instances:
        print(f"  {str(ei.id)[:8]}... | view_count={ei.attributes.get('view_count')} | component_size={ei.attributes.get('component_size')}")


if __name__ == "__main__":
    asyncio.run(main())
