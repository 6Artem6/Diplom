"""
Representation Service Implementation

Class-aware cleaned-text embeddings + CLIP visual (visual not used for cross-view).
"""

from typing import List
import logging

from src.application.detection.element_builder import resolve_element_class
from src.application.ocr.ocr_normalizer import build_embedding_input, normalize_ocr
from src.domain.interfaces.representation import (
    RepresentationService,
    MultimodalEmbedding,
)
from src.domain.interfaces.gui_detection import GUIBlock
from .embedding_service import EmbeddingService
from .clip_encoder import CLIPEncoder
from .sentence_transformer_encoder import SentenceTransformerEncoder

logger = logging.getLogger(__name__)


class RepresentationServiceImpl(RepresentationService):
    """
    Multimodal embeddings with class-aware cleaned-text for cross-view matching.

    Architecture rationale:
    - Sentence-transformer on f"{class}: {cleaned_text}" only
    - Raw OCR never enters embedding input
    - Visual CLIP kept for debug / future fusion, not cross-view primary
    """

    def __init__(self, embedding_service: EmbeddingService = None):
        if embedding_service is None:
            self.embedding_service = CLIPEncoder()
        else:
            self.embedding_service = embedding_service
        self.text_encoder = SentenceTransformerEncoder.get_instance()

    async def generate_embeddings(
        self,
        blocks: List[GUIBlock],
    ) -> List[MultimodalEmbedding]:
        logger.info("Representation: Generating embeddings for %d blocks", len(blocks))

        embeddings: List[MultimodalEmbedding] = []
        for block in blocks:
            raw_ocr = (block.ocr_text_raw or block.ocr_text or "").strip()
            if block.ocr_text_raw:
                cleaned = (block.ocr_text or "").strip()
                noisy = block.ocr_noisy
            else:
                norm = normalize_ocr(block.ocr_text)
                raw_ocr = norm.raw_text
                cleaned = norm.cleaned_text
                noisy = norm.is_noisy
                block.ocr_text_raw = raw_ocr
                block.ocr_text = cleaned
                block.ocr_noisy = noisy

            layout_stub: dict = {
                "x": block.bounding_box.get("x", 0.0),
                "y": block.bounding_box.get("y", 0.0),
                "width": block.bounding_box.get("width", 0.0),
                "height": block.bounding_box.get("height", 0.0),
            }
            class_label = resolve_element_class(block, layout_stub)

            embedding_input = build_embedding_input(class_label, cleaned)
            text_emb = await self.text_encoder.embed_text(embedding_input)

            visual_emb = await self.embedding_service.embed_visual(block)
            block.visual_features = visual_emb

            layout_features = {
                **layout_stub,
                "class_label": class_label,
                "raw_ocr": raw_ocr,
                "cleaned_text": cleaned,
                "noisy_ocr": noisy,
                "embedding_input_text": embedding_input,
                "embedding_source": "sentence_transformer",
            }

            if cleaned or raw_ocr:
                logger.debug(
                    "Representation: block=%s class=%s embedding_input=%r noisy=%s",
                    block.id,
                    class_label,
                    embedding_input[:80],
                    noisy,
                )

            embeddings.append(
                MultimodalEmbedding(
                    block_id=block.id,
                    visual_embedding=visual_emb,
                    text_embedding=text_emb,
                    layout_features=layout_features,
                )
            )

        logger.info("Representation: Generated %d embeddings (text-primary for linking)", len(embeddings))
        return embeddings
