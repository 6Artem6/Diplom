"""
Experimental multilevel v2 и Form Container First (ТЗ).

- run_experimental_multilevel_v2: PageOrientation → SemanticRegions → FormSkeleton → Slots → RoleBasedField → FormGraph.
- run_form_container_first_inference: FormContainerDetector → FormInnerLayout → SlotDetector → FieldLocator → FormGraph.
  Инвариант №0: все уровни только внутри FormContainer.bbox.
"""

from src.infrastructure.atoms_v2.experimental_v2.run_experimental_v2 import (
    run_experimental_multilevel_v2,
)
from src.infrastructure.atoms_v2.experimental_v2.run_form_container_first_inference import (
    run_form_container_first_inference,
)

__all__ = ["run_experimental_multilevel_v2", "run_form_container_first_inference"]
