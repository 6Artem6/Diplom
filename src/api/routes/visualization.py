"""
BPG Visualization API Routes

Endpoints for visual verification of cross-view entity linking.
Returns PNG images with bounding boxes, entity colors, and cross-view connections.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response, FileResponse
from typing import Optional
from uuid import UUID
import json
import logging
import os
from pathlib import Path
import colorsys

from ...domain.interfaces.bpg_storage import BPGStorage
from ..dependencies import get_bpg_storage

logger = logging.getLogger(__name__)

router = APIRouter()
DEBUG_BASE = Path("/app/debug")


def _generate_colors(n: int) -> list:
    """Generate n distinct colors for entity instances."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


@router.get("/bpg/{bpg_id}/debug/visualization")
async def visualize_bpg(
    bpg_id: str,
    storage: BPGStorage = Depends(get_bpg_storage),
):
    """
    Visualize BPG with cross-view entity linking.

    Returns PNG image with:
    - Bounding boxes for each GUIManifestation
    - Color-coded entity instances (same color = same entity)
    - Labels: view_id, class_label, similarity
    - Cross-view connections between different screenshots

    Architecture rationale:
    - Enables user verification of cross-view correctness
    - Research: critical for validating entity linking accuracy
    - Visual feedback for debugging and evaluation
    """
    try:
        bpg_uuid = UUID(bpg_id)
        bpg = await storage.get(bpg_uuid)
        if not bpg:
            raise HTTPException(
                status_code=404,
                detail=f"BPG not found: {bpg_id}",
            )

        # Get all manifestations from entity instances
        # Note: In full implementation, would load from storage
        # For now, reconstruct from cross_view_edges
        
        # Group manifestations by entity_instance_id
        entity_to_manifestations = {}
        for edge in bpg.cross_view_edges:
            source_id = edge.source_id
            target_id = edge.target_id
            
            # Find entity instance for this edge
            for ei in bpg.entity_instances:
                if ei.id not in entity_to_manifestations:
                    entity_to_manifestations[ei.id] = []
                # Add source and target (would need to load actual manifestations)
                entity_to_manifestations[ei.id].extend([source_id, target_id])
        
        # Generate colors for entity instances
        entity_colors = {}
        colors = _generate_colors(len(bpg.entity_instances))
        for i, ei in enumerate(bpg.entity_instances):
            entity_colors[ei.id] = colors[i]
        
        # Collect similarity statistics
        similarity_scores = [edge.similarity_score for edge in bpg.cross_view_edges]
        similarity_summary = {}
        if similarity_scores:
            similarity_summary = {
                "threshold": float(os.getenv("CROSS_VIEW_SIMILARITY_THRESHOLD", "0.78")),
                "max": max(similarity_scores),
                "mean": sum(similarity_scores) / len(similarity_scores),
                "min": min(similarity_scores),
                "count": len(similarity_scores),
            }
        
        # Find PNG visualization files and their API URLs
        debug_dir = DEBUG_BASE / str(bpg_id)
        png_files = []
        if debug_dir.exists():
            for p in sorted(debug_dir.glob("*.png")):
                name = p.name
                png_files.append({
                    "filename": name,
                    "url": f"/api/v1/bpg/{bpg_id}/debug/image/{name}",
                })
        
        visualization_data = {
            "bpg_id": str(bpg_id),
            "cross_view_edges": [
                {
                    "source_id": str(edge.source_id),
                    "target_id": str(edge.target_id),
                    "similarity_score": edge.similarity_score,
                    "confidence": edge.confidence.score,
                    "source_view_id": edge.provenance.metadata.get("source_view_id"),
                    "target_view_id": edge.provenance.metadata.get("target_view_id"),
                    "validation": "✅ Different views" if (
                        edge.provenance.metadata.get("source_view_id") != 
                        edge.provenance.metadata.get("target_view_id")
                    ) else "❌ SAME VIEW (ERROR!)",
                }
                for edge in bpg.cross_view_edges
            ],
            "entity_instances": [
                {
                    "id": str(ei.id),
                    "attributes": ei.attributes,
                    "view_count": ei.attributes.get("view_count", 0),
                    "is_cross_view": ei.attributes.get("view_count", 0) >= 2,
                    "color": f"rgb{entity_colors.get(ei.id, (128, 128, 128))}",
                }
                for ei in bpg.entity_instances
            ],
            "summary": {
                "total_cross_view_edges": len(bpg.cross_view_edges),
                "cross_view_entities": len([
                    ei for ei in bpg.entity_instances
                    if ei.attributes.get("view_count", 0) >= 2
                ]),
                "validation_passed": all(
                    edge.provenance.metadata.get("source_view_id") != 
                    edge.provenance.metadata.get("target_view_id")
                    for edge in bpg.cross_view_edges
                ),
            },
            "similarity_stats": similarity_summary,
            "visualization_files": png_files,
        }

        return Response(
            content=json.dumps(visualization_data, indent=2),
            media_type="application/json",
        )

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid BPG ID format: {bpg_id}",
        )


@router.get("/bpg/{bpg_id}/debug/image/{filename}", response_class=FileResponse)
async def get_debug_image(
    bpg_id: str,
    filename: str,
):
    """
    Serve a single debug PNG (e.g. view visualization with bounding boxes).
    filename must be a plain basename (e.g. view-uuid.png); path traversal is disallowed.
    """
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    try:
        UUID(bpg_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid BPG ID: {bpg_id}")
    path = (DEBUG_BASE / bpg_id / filename).resolve()
    base = DEBUG_BASE.resolve()
    if not path.is_file() or base not in path.parents:
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    return FileResponse(path, media_type="image/png")
