"""
BPG Construction API Routes

Endpoints for building and querying Business Process Graph.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel
from uuid import UUID

from src.application.use_cases.bpg_pipeline import BuildBPGUseCase, BuildBPGRequest
from src.domain.interfaces.bpg_construction import BusinessProcessGraph
from src.domain.interfaces.bpg_storage import BPGStorage
from src.domain.interfaces.bpg_query import BPGQueryService
from ..dependencies import get_bpg_use_case, get_bpg_storage, get_bpg_query_service

router = APIRouter()


class BuildBPGRequestModel(BaseModel):
    """Request model for BPG construction."""

    screenshot_paths: List[str]
    clickstream_data: Optional[List[dict]] = None


class BuildBPGResponse(BaseModel):
    """Response for BPG construction."""

    bpg_id: str
    entity_types_count: int
    entity_instances_count: int
    actions_count: int
    patterns_count: int
    rules_count: int
    edges_count: int
    cross_view_edges_count: int
    message: str


@router.post("/bpg/build", response_model=BuildBPGResponse)
async def build_bpg(
    request: BuildBPGRequestModel,
    use_case: BuildBPGUseCase = Depends(get_bpg_use_case),
    storage: BPGStorage = Depends(get_bpg_storage),
):
    """
    Build BPG from screenshot paths.

    Architecture rationale:
    - Accepts file paths (not file uploads) for simplicity in skeleton
    - Real implementation would handle file uploads
    - Saves BPG to storage and returns summary statistics
    """
    try:
        paths = [Path(p) for p in request.screenshot_paths]
        bpg_request = BuildBPGRequest(
            screenshot_paths=paths,
            clickstream_data=request.clickstream_data or [],
        )
        bpg = await use_case.execute(bpg_request)
        bpg_id = await storage.save(bpg)
        return BuildBPGResponse(
            bpg_id=str(bpg_id),
            entity_types_count=len(bpg.entity_types),
            entity_instances_count=len(bpg.entity_instances),
            actions_count=len(bpg.actions),
            patterns_count=len(bpg.patterns),
            rules_count=len(bpg.rules),
            edges_count=len(bpg.edges),
            cross_view_edges_count=len(bpg.cross_view_edges),
            message="BPG built successfully",
        )
    except RuntimeError as e:
        if "Cross-view linking failed" in str(e):
            raise HTTPException(status_code=422, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build BPG: {str(e)}")


@router.get("/bpg/{bpg_id}", response_model=dict)
async def get_bpg(
    bpg_id: str,
    storage: BPGStorage = Depends(get_bpg_storage),
):
    """Get BPG by ID."""
    try:
        bpg_uuid = UUID(bpg_id)
        bpg = await storage.get(bpg_uuid)
        if not bpg:
            raise HTTPException(
                status_code=404,
                detail=f"BPG not found: {bpg_id}",
            )
        return bpg.model_dump()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid BPG ID format: {bpg_id}",
        )


@router.get("/bpg/{bpg_id}/context", response_model=dict)
async def get_bpg_context(
    bpg_id: str,
    query: Optional[str] = None,
    entity_type: Optional[str] = None,
    min_confidence: float = 0.5,
    query_service: BPGQueryService = Depends(get_bpg_query_service),
):
    """
    Get relevant BPG context for LLM runtime.

    Architecture rationale:
    - LLM agents query this endpoint for domain context
    - Returns minimal relevant subgraph based on query
    - Enables prompt enrichment with BPG knowledge
    """
    try:
        bpg_uuid = UUID(bpg_id)
        subgraph = await query_service.find_relevant_context(
            bpg_id=bpg_uuid,
            query=query,
            entity_type=entity_type,
            min_confidence=min_confidence,
        )
        return subgraph.model_dump()
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}",
        )
