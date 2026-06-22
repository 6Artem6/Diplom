"""
Detected GUI Element — primary truth layer from detection pipeline.

Class + bbox + text originate here; entity/manifestation layers are aggregations.
"""

from typing import Dict, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class DetectedElement(BaseModel):
    """
    One UI element as produced by detection + OCR, before BPG aggregation.

    element_id aligns with GUIManifestation.id (1:1).
    """

    element_id: UUID = Field(..., description="Unique element ID (= manifestation id)")
    view_id: UUID = Field(..., description="View (screenshot) UUID")
    screenshot_id: str = Field(..., description="Screenshot stem, e.g. demo_form_16")
    bbox: Dict[str, float] = Field(
        ...,
        description="Bounding box: x, y, width, height and/or x1,y1,x2,y2",
    )
    class_name: str = Field(
        ...,
        alias="class",
        description="UI class: button, input, text_block, etc.",
    )
    role: Optional[str] = Field(
        None,
        description="Derived semantic role (text, action, input, visual, …)",
    )
    text: Optional[str] = Field(None, description="Cleaned display text")
    cleaned_text: Optional[str] = Field(None, description="Normalized OCR (embedding source)")
    ocr_text: Optional[str] = Field(None, description="Raw OCR text before normalization")
    noisy_ocr: bool = Field(False, description="True if OCR could not be reliably cleaned")
    embedding_input: Optional[str] = Field(
        None, description="Class-aware text fed to sentence-transformer"
    )
    embedding_source: Optional[str] = Field(
        None, description="Embedding model id, e.g. sentence_transformer"
    )
    confidence: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="Detection / classification confidence",
    )
    pipeline_stage: str = Field(
        default="detection",
        description="Pipeline stage that produced this record",
    )
    entity_instance_id: Optional[UUID] = Field(
        None,
        description="Assigned after entity linking",
    )
    manifestation_id: Optional[UUID] = Field(
        None,
        description="Linked GUIManifestation id (same as element_id)",
    )

    model_config = ConfigDict(populate_by_name=True)
