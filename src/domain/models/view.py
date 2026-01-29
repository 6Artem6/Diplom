"""
View Model

Explicit model for GUI views (one View = one screenshot).
Critical for cross-view entity linking semantics.
"""

from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class View(BaseModel):
    """
    Represents a single GUI view (one screenshot).

    Domain rationale:
    - Cross-view linking means linking entities ACROSS different views
    - Same-view similarity is NOT cross-view linking
    - Enables explicit validation: cross_view edges only between different view_id
    - Research: enables user verification of cross-view correctness
    """

    id: UUID = Field(default_factory=uuid4, description="Unique view identifier")
    screenshot_id: str = Field(..., description="ID of screenshot (from preprocessing)")
    screenshot_path: str = Field(..., description="Path to screenshot file")

    class Config:
        frozen = True
