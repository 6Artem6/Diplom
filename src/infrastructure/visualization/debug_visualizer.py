"""
Debug Visualization Service

Generates PNG images with bounding boxes, entity colors, and cross-view connections.
Side-effect service that doesn't affect business logic.
"""

from typing import List, Dict, Optional
from uuid import UUID
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont
import colorsys

from src.domain.models.bpg_models import GUIManifestation
from src.domain.models.bpg_edges import CrossViewEdge
from src.domain.models.view import View

logger = logging.getLogger(__name__)


def _generate_colors(n: int) -> List[tuple]:
    """Generate n distinct colors for entity instances."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        saturation = 0.7
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


class DebugVisualizer:
    """
    Service for generating debug visualization images.
    
    Architecture rationale:
    - Side-effect service: doesn't affect business logic
    - Generates PNG images for debugging and explainability
    - Handles errors gracefully (pipeline doesn't fail if visualization fails)
    """

    def __init__(self, debug_output_dir: str = "/app/debug"):
        """
        Initialize debug visualizer.
        
        Args:
            debug_output_dir: Directory for saving visualization images
        """
        self.debug_output_dir = Path(debug_output_dir)
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load font (fallback to default if not available)
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            logger.warning("DebugVisualizer: Could not load font, using default")
            self.font = ImageFont.load_default()
            self.font_large = ImageFont.load_default()

    def visualize_view(
        self,
        bpg_id: UUID,
        view: View,
        manifestations: List[GUIManifestation],
        entity_colors: Dict[UUID, tuple],
        cross_view_edges: List[CrossViewEdge],
    ) -> Optional[Path]:
        """
        Generate PNG visualization for a single view.
        
        Args:
            bpg_id: BPG identifier
            view: View to visualize
            manifestations: Manifestations in this view
            entity_colors: Color mapping for entity instances
            cross_view_edges: Cross-view edges (for highlighting matches)
            
        Returns:
            Path to generated PNG file, or None if failed
        """
        try:
            # Load screenshot
            screenshot_path = Path(view.screenshot_path)
            if not screenshot_path.exists():
                logger.warning(f"DebugVisualizer: Screenshot not found: {screenshot_path}")
                return self._create_no_detections_image(bpg_id, view)
            
            image = Image.open(screenshot_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            
            # Filter manifestations for this view
            view_manifestations = [
                m for m in manifestations
                if m.view_id == view.id
            ]
            
            if not view_manifestations:
                logger.warning(f"DebugVisualizer: No manifestations for view {view.id}")
                return self._create_no_detections_image(bpg_id, view)
            
            # Draw bounding boxes and labels
            for man in view_manifestations:
                bbox = man.bounding_box
                x1 = bbox.get("x1", bbox.get("x", 0))
                y1 = bbox.get("y1", bbox.get("y", 0))
                x2 = bbox.get("x2", x1 + bbox.get("width", 0))
                y2 = bbox.get("y2", y1 + bbox.get("height", 0))
                
                # Get entity color
                entity_color = entity_colors.get(man.entity_instance_id, (128, 128, 128))
                
                # Draw bounding box
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline=entity_color,
                    width=3,
                )
                
                # Draw label background
                label_text = f"{man.id.hex[:8]}"
                class_label = man.layout_features.get("class_label", "unknown")
                if class_label:
                    label_text += f" | {class_label}"
                
                # Find cross-view edge for this manifestation
                for edge in cross_view_edges:
                    if edge.source_id == man.id or edge.target_id == man.id:
                        label_text += f" | sim={edge.similarity_score:.2f}"
                        break
                
                # Get text size for background
                bbox_text = draw.textbbox((0, 0), label_text, font=self.font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # Draw label background (ensure it's visible)
                label_bg_y1 = max(0, y1 - text_height - 4)
                label_bg_y2 = max(text_height, y1)
                
                draw.rectangle(
                    [x1, label_bg_y1, x1 + text_width + 4, label_bg_y2],
                    fill=entity_color,
                    outline=entity_color,
                )
                
                # Draw label text
                draw.text(
                    (x1 + 2, label_bg_y1 + 2),
                    label_text,
                    fill=(255, 255, 255),
                    font=self.font,
                )
            
            # Save image
            output_path = self.debug_output_dir / str(bpg_id) / f"{view.id}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, "PNG")
            
            logger.info(f"DebugVisualizer: Saved visualization to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"DebugVisualizer: Failed to visualize view {view.id}: {e}", exc_info=True)
            return None

    def _create_no_detections_image(self, bpg_id: UUID, view: View) -> Optional[Path]:
        """Create placeholder image when no detections found."""
        try:
            screenshot_path = Path(view.screenshot_path)
            if screenshot_path.exists():
                image = Image.open(screenshot_path).convert("RGB")
            else:
                # Create blank image
                image = Image.new("RGB", (800, 600), color=(240, 240, 240))
            
            draw = ImageDraw.Draw(image)
            
            # Draw "NO DETECTIONS" text
            text = "NO DETECTIONS"
            bbox = draw.textbbox((0, 0), text, font=self.font_large)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            img_width, img_height = image.size
            x = (img_width - text_width) // 2
            y = (img_height - text_height) // 2
            
            # Draw background
            draw.rectangle(
                [x - 10, y - 10, x + text_width + 10, y + text_height + 10],
                fill=(255, 200, 200),
                outline=(255, 0, 0),
                width=2,
            )
            
            # Draw text
            draw.text(
                (x, y),
                text,
                fill=(0, 0, 0),
                font=self.font_large,
            )
            
            # Save
            output_path = self.debug_output_dir / str(bpg_id) / f"{view.id}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, "PNG")
            
            logger.info(f"DebugVisualizer: Created no-detections image at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"DebugVisualizer: Failed to create no-detections image: {e}", exc_info=True)
            return None
