"""
State Machine Pipeline — Entry Point

Детерминированный pipeline для анализа форм по State Machine архитектуре.

Этапы:
S0: Container Detection (внешний)
S1: Visual Geometry Extraction (immutable после этого)
S2: OCR Extraction
S3: Structural Segmentation
S4: Slot Assignment  
S5: Pattern Analysis
S6: Semantic Validation
S7: Graph Assembly (этот файл)

Ключевые принципы:
- Строгий forward flow без циклов
- visual_elements immutable после S1
- NMS только в S1
- Только относительные размеры
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Stage imports
from .visual_geometry_extractor import extract_visual_geometry, S1Result
from .ocr_extractor import extract_ocr, get_ocr_blocks_as_dicts, S2Result
from .structural_segmentation import segment_into_rows, S3Result
from .slot_assignment import assign_slots, get_form_atoms, S4Result
from .pattern_analysis import analyze_patterns, S5Result
from .semantic_validation import validate_form, format_flags_report, S6Result

# S0: Container Detection
from .form_container_detector import detect_form_containers, get_best_container

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    ocr_engine: str = "easyocr"  # 'easyocr', 'tesseract', 'paddleocr_service'
    demo_mode: bool = True
    output_dir: Optional[str] = None
    
    # OCR options
    skip_ocr: bool = False
    precomputed_ocr: Optional[List[Dict[str, Any]]] = None
    ocr_service_url: Optional[str] = None  # URL для paddleocr_service (напр. http://paddleocr_service:8000)
    
    # S0: Container Detection options
    auto_detect_container: bool = True  # если container_bbox не передан — детектить автоматически
    use_full_image_fallback: bool = True  # если детекция не нашла контейнер — использовать всё изображение


@dataclass
class PipelineResult:
    """Full pipeline result."""
    # Intermediate results
    s0_container_detected: bool = False  # был ли контейнер найден автоматически
    s0_container_confidence: float = 0.0
    s1_result: Optional[S1Result] = None
    s2_result: Optional[S2Result] = None
    s3_result: Optional[S3Result] = None
    s4_result: Optional[S4Result] = None
    s5_result: Optional[S5Result] = None
    s6_result: Optional[S6Result] = None
    
    # Final outputs
    atoms: List[Dict[str, Any]] = None
    is_valid: bool = False
    confidence: float = 0.0
    
    # Metadata
    image_path: str = ""
    container_bbox: List[float] = None
    container_source: str = ""  # "provided", "auto_detected", "full_image"
    
    # Errors
    error: Optional[str] = None
    stage_failed: Optional[str] = None
    
    def __post_init__(self):
        if self.atoms is None:
            self.atoms = []
        if self.container_bbox is None:
            self.container_bbox = []


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_pipeline_result(
    image_path: str,
    result: PipelineResult,
    output_dir: str,
) -> None:
    """Generate visualization images for debugging."""
    import cv2
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Could not read image for visualization: {image_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem
    
    # 1. S1 — Visual Elements
    if result.s1_result:
        img_s1 = image.copy()
        for elem in result.s1_result.visual_elements:
            bbox = elem.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Color by type
            colors = {
                "input": (0, 255, 0),      # green
                "textarea": (0, 165, 255),  # orange
                "button": (0, 0, 255),      # red
                "checkbox": (255, 0, 255),  # magenta
                "radio": (255, 0, 128),     # pink
                "container": (128, 128, 128),  # gray
                "label": (255, 255, 0),     # cyan
            }
            color = colors.get(elem.element_type, (200, 200, 200))
            
            cv2.rectangle(img_s1, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{elem.element_type} ({elem.confidence:.2f})"
            cv2.putText(img_s1, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_s1_visual.png"), img_s1)
    
    # 2. S3 — Rows
    if result.s3_result:
        img_s3 = image.copy()
        
        row_colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
        ]
        
        for i, row in enumerate(result.s3_result.rows):
            color = row_colors[i % len(row_colors)]
            
            # Draw row bounds
            x1 = int(row.x_min)
            y1 = int(row.y_min)
            x2 = int(row.x_max)
            y2 = int(row.y_max)
            
            cv2.rectangle(img_s3, (x1, y1), (x2, y2), color, 1)
            cv2.putText(img_s3, f"Row {i} ({row.row_type})", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw elements
            for elem in row.elements:
                eb = elem.bbox
                ex1, ey1, ex2, ey2 = int(eb[0]), int(eb[1]), int(eb[2]), int(eb[3])
                cv2.rectangle(img_s3, (ex1, ey1), (ex2, ey2), color, 2)
        
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_s3_rows.png"), img_s3)
    
    # 3. S4 — Slots
    if result.s4_result:
        img_s4 = image.copy()
        
        slot_colors = {
            "LABEL": (255, 255, 0),    # cyan
            "INPUT": (0, 255, 0),       # green
            "TEXTAREA": (0, 165, 255),  # orange
            "ACTION": (0, 0, 255),      # red
            "CHECKBOX": (255, 0, 255),  # magenta
            "RADIO": (255, 0, 128),     # pink
            "HEADER": (255, 128, 0),    # blue-ish
            "UNKNOWN": (128, 128, 128), # gray
        }
        
        for rs in result.s4_result.row_slots:
            for a in rs.assignments:
                bbox = a.element.bbox
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                color = slot_colors.get(a.slot, (200, 200, 200))
                
                # Fill with transparency
                overlay = img_s4.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.3, img_s4, 0.7, 0, img_s4)
                
                # Border
                cv2.rectangle(img_s4, (x1, y1), (x2, y2), color, 2)
                
                # Label
                cv2.putText(img_s4, a.slot, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw binding line
                if a.bound_to:
                    b_bbox = a.bound_to.element.bbox
                    bx = int((b_bbox[0] + b_bbox[2]) / 2)
                    by = int((b_bbox[1] + b_bbox[3]) / 2)
                    ax = int(x2)
                    ay = int((y1 + y2) / 2)
                    cv2.line(img_s4, (ax, ay), (bx, by), (0, 255, 255), 1)
        
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_s4_slots.png"), img_s4)
    
    # 4. S5 — Patterns
    if result.s5_result:
        img_s5 = image.copy()
        
        pattern_colors = {
            "checkbox_group": (255, 0, 255),
            "radio_group": (255, 128, 0),
            "field_pair": (0, 255, 128),
            "button_group": (128, 0, 255),
            "repeating_row": (255, 255, 128),
        }
        
        for pattern in result.s5_result.patterns:
            color = pattern_colors.get(pattern.pattern_type, (200, 200, 200))
            
            # Draw bounding box around all elements in pattern
            all_bboxes = [e.element.bbox for e in pattern.elements]
            if all_bboxes:
                px1 = int(min(b[0] for b in all_bboxes)) - 5
                py1 = int(min(b[1] for b in all_bboxes)) - 5
                px2 = int(max(b[2] for b in all_bboxes)) + 5
                py2 = int(max(b[3] for b in all_bboxes)) + 5
                
                cv2.rectangle(img_s5, (px1, py1), (px2, py2), color, 2)
                cv2.putText(img_s5, pattern.pattern_type, (px1, py1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_s5_patterns.png"), img_s5)
    
    logger.info(f"Visualizations saved to {output_dir}")


def save_diagnostics(
    result: PipelineResult,
    output_dir: str,
    image_name: str,
) -> None:
    """Save diagnostics to text file."""
    os.makedirs(output_dir, exist_ok=True)
    
    lines = []
    lines.append(f"=== State Machine Pipeline Report ===")
    lines.append(f"Image: {image_name}")
    lines.append(f"Container: {result.container_bbox}")
    lines.append(f"Container source: {result.container_source}")
    if result.s0_container_detected:
        lines.append(f"Container confidence: {result.s0_container_confidence:.2f}")
    lines.append("")
    
    if result.error:
        lines.append(f"ERROR: {result.error}")
        lines.append(f"Stage failed: {result.stage_failed}")
    else:
        lines.append(f"Valid: {result.is_valid}")
        lines.append(f"Confidence: {result.confidence:.2f}")
        lines.append(f"Total atoms: {len(result.atoms)}")
        lines.append("")
    
    # S1 diagnostics
    if result.s1_result:
        lines.append("--- S1: Visual Geometry ---")
        for k, v in result.s1_result.diagnostics.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    
    # S2 diagnostics
    if result.s2_result:
        lines.append("--- S2: OCR Extraction ---")
        lines.append(f"  Language: {result.s2_result.language.primary}")
        lines.append(f"  RU ratio: {result.s2_result.language.ru_ratio:.2f}")
        for k, v in result.s2_result.diagnostics.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    
    # S3 diagnostics
    if result.s3_result:
        lines.append("--- S3: Structural Segmentation ---")
        for k, v in result.s3_result.diagnostics.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    
    # S4 diagnostics
    if result.s4_result:
        lines.append("--- S4: Slot Assignment ---")
        for k, v in result.s4_result.diagnostics.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    
    # S5 diagnostics
    if result.s5_result:
        lines.append("--- S5: Pattern Analysis ---")
        for k, v in result.s5_result.diagnostics.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    
    # S6 report
    if result.s6_result:
        lines.append("--- S6: Semantic Validation ---")
        lines.append(format_flags_report(result.s6_result))
        lines.append("")
    
    # Atoms
    if result.atoms:
        lines.append("--- Atoms ---")
        for atom in result.atoms:
            lines.append(f"  {atom.get('slot', '?')}: {atom.get('bbox', [])} (row={atom.get('row_index', '?')})")
    
    # Write
    output_path = os.path.join(output_dir, f"{image_name}_diagnostics.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Diagnostics saved to {output_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_state_machine_pipeline(
    image_path: str,
    container_bbox: Optional[List[float]] = None,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """
    Run full State Machine pipeline.
    
    Args:
        image_path: путь к изображению
        container_bbox: bbox контейнера формы [x1, y1, x2, y2] (опционально - если не передан, определяется автоматически)
        config: конфигурация pipeline
    
    Returns:
        PipelineResult со всеми результатами
    """
    import cv2
    
    if config is None:
        config = PipelineConfig()
    
    result = PipelineResult(
        image_path=image_path,
    )
    
    logger.info(f"Starting State Machine Pipeline for {image_path}")
    
    # =========================================================================
    # S0: Container Detection (если bbox не передан)
    # =========================================================================
    if container_bbox is not None and len(container_bbox) >= 4:
        # Контейнер передан явно
        result.container_bbox = list(container_bbox)
        result.container_source = "provided"
        result.s0_container_detected = False
        logger.debug(f"S0: using provided container bbox: {container_bbox}")
    else:
        # Автоматическое определение контейнера
        if config.auto_detect_container:
            try:
                containers, diag = detect_form_containers(image_path)
                best_container = get_best_container(containers, demo_mode=config.demo_mode)
                
                if best_container:
                    result.container_bbox = list(best_container.bbox)
                    result.container_source = "auto_detected"
                    result.s0_container_detected = True
                    result.s0_container_confidence = best_container.confidence
                    logger.info(f"S0: auto-detected container bbox: {best_container.bbox} (conf={best_container.confidence:.2f})")
                else:
                    logger.warning("S0: no container detected")
                    
            except Exception as e:
                logger.warning(f"S0: container detection failed: {e}")
        
        # Fallback: использовать всё изображение
        if not result.container_bbox and config.use_full_image_fallback:
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
                result.container_bbox = [0.0, 0.0, float(w), float(h)]
                result.container_source = "full_image"
                logger.info(f"S0: using full image as container: {result.container_bbox}")
            else:
                result.error = "Could not read image"
                result.stage_failed = "S0"
                return result
    
    container_bbox = result.container_bbox
    if not container_bbox or len(container_bbox) < 4:
        result.error = "No container bbox available"
        result.stage_failed = "S0"
        return result
    
    # =========================================================================
    # S2: OCR Extraction (run first to get OCR blocks for S1)
    # =========================================================================
    try:
        if config.skip_ocr:
            s2_result = S2Result(
                ocr_blocks=[],
                language=None,  # type: ignore
                median_line_height=20.0,
                diagnostics={"skipped": True},
            )
            # Create minimal language info
            from .ocr_extractor import LanguageInfo
            s2_result.language = LanguageInfo(primary="unknown", ru_ratio=0.0, en_ratio=0.0, confidence=0.0)
        else:
            s2_result = extract_ocr(
                image_path=image_path,
                container_bbox=container_bbox,
                ocr_engine=config.ocr_engine,
                precomputed_ocr=config.precomputed_ocr,
                ocr_service_url=config.ocr_service_url,
            )
        result.s2_result = s2_result
        logger.debug(f"S2 completed: {len(s2_result.ocr_blocks)} OCR blocks")
        # #region agent log — OCR bboxes for verification
        _log_path = os.environ.get("DEBUG_BBOX_LOG")
        if _log_path:
            import json
            import time
            _dir = os.path.dirname(_log_path)
            if _dir:
                os.makedirs(_dir, exist_ok=True)
            _payload = {
                "timestamp": int(time.time() * 1000),
                "source": "ocr",
                "image_path": image_path,
                "stage": "S2",
                "message": "ocr_bboxes",
                "data": {
                    "count": len(s2_result.ocr_blocks),
                    "bboxes": [{"bbox": list(b.bbox), "text": b.text[:80]} for b in s2_result.ocr_blocks],
                },
            }
            with open(_log_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(_payload, ensure_ascii=False) + "\n")
        # #endregion
    except Exception as e:
        logger.error(f"S2 failed: {e}")
        result.error = str(e)
        result.stage_failed = "S2"
        return result
    
    # =========================================================================
    # S1: Visual Geometry Extraction
    # =========================================================================
    try:
        ocr_blocks_for_s1 = get_ocr_blocks_as_dicts(s2_result) if s2_result else []
        
        s1_result = extract_visual_geometry(
            image_path=image_path,
            container_bbox=container_bbox,
            ocr_blocks=ocr_blocks_for_s1,
        )
        result.s1_result = s1_result
        logger.debug(f"S1 completed: {len(s1_result.visual_elements)} visual elements")
        # #region agent log — CV bboxes for verification
        _log_path = os.environ.get("DEBUG_BBOX_LOG")
        if _log_path:
            import json
            import time
            _payload = {
                "timestamp": int(time.time() * 1000),
                "source": "cv",
                "image_path": image_path,
                "stage": "S1",
                "message": "cv_bboxes",
                "data": {
                    "count": len(s1_result.visual_elements),
                    "bboxes": [
                        {"bbox": list(e.bbox), "element_type": e.element_type}
                        for e in s1_result.visual_elements
                    ],
                },
            }
            with open(_log_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(_payload, ensure_ascii=False) + "\n")
        # #endregion
        if not s1_result.visual_elements:
            logger.warning("S1 found no visual elements")
    except Exception as e:
        logger.error(f"S1 failed: {e}")
        result.error = str(e)
        result.stage_failed = "S1"
        return result
    
    # =========================================================================
    # S3: Structural Segmentation
    # =========================================================================
    try:
        s3_result = segment_into_rows(
            visual_elements=s1_result.visual_elements,
            ocr_blocks=s2_result.ocr_blocks,
            context=s1_result.context,
        )
        result.s3_result = s3_result
        logger.debug(f"S3 completed: {len(s3_result.rows)} rows")
    except Exception as e:
        logger.error(f"S3 failed: {e}")
        result.error = str(e)
        result.stage_failed = "S3"
        return result
    
    # =========================================================================
    # S4: Slot Assignment
    # =========================================================================
    try:
        s4_result = assign_slots(
            rows=s3_result.rows,
            context=s1_result.context,
            language=s2_result.language,
            all_elements=s1_result.visual_elements,  # for container expansion
        )
        result.s4_result = s4_result
        logger.debug(f"S4 completed: {sum(len(rs.assignments) for rs in s4_result.row_slots)} assignments")
    except Exception as e:
        logger.error(f"S4 failed: {e}")
        result.error = str(e)
        result.stage_failed = "S4"
        return result
    
    # =========================================================================
    # S5: Pattern Analysis
    # =========================================================================
    try:
        s5_result = analyze_patterns(
            s4_result=s4_result,
            context=s1_result.context,
        )
        result.s5_result = s5_result
        logger.debug(f"S5 completed: {len(s5_result.patterns)} patterns")
    except Exception as e:
        logger.error(f"S5 failed: {e}")
        result.error = str(e)
        result.stage_failed = "S5"
        return result
    
    # =========================================================================
    # S6: Semantic Validation
    # =========================================================================
    try:
        s6_result = validate_form(
            s4_result=s4_result,
            s5_result=s5_result,
            language=s2_result.language,
        )
        result.s6_result = s6_result
        logger.debug(f"S6 completed: {len(s6_result.flags)} flags, valid={s6_result.is_valid}")
    except Exception as e:
        logger.error(f"S6 failed: {e}")
        result.error = str(e)
        result.stage_failed = "S6"
        return result
    
    # =========================================================================
    # S7: Graph Assembly (extract atoms)
    # =========================================================================
    try:
        atoms = get_form_atoms(s4_result)
        result.atoms = atoms
        result.is_valid = s6_result.is_valid
        result.confidence = s6_result.confidence_score
        logger.info(f"S7 completed: {len(atoms)} atoms, valid={result.is_valid}")
    except Exception as e:
        logger.error(f"S7 failed: {e}")
        result.error = str(e)
        result.stage_failed = "S7"
        return result
    
    # =========================================================================
    # Visualization (demo mode)
    # =========================================================================
    if config.demo_mode and config.output_dir:
        try:
            visualize_pipeline_result(image_path, result, config.output_dir)
            save_diagnostics(result, config.output_dir, Path(image_path).stem)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    
    return result


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def run_pipeline_batch(
    image_paths: List[str],
    container_bboxes: Optional[List[Optional[List[float]]]] = None,
    config: Optional[PipelineConfig] = None,
) -> List[PipelineResult]:
    """
    Run pipeline on multiple images.
    
    Args:
        image_paths: список путей к изображениям
        container_bboxes: опциональный список bbox (если None — все авто-детект)
        config: конфигурация
    """
    if container_bboxes is None:
        container_bboxes = [None] * len(image_paths)
    
    if len(image_paths) != len(container_bboxes):
        raise ValueError("image_paths and container_bboxes must have same length")
    
    results = []
    for img_path, bbox in zip(image_paths, container_bboxes):
        result = run_state_machine_pipeline(img_path, bbox, config)
        results.append(result)
    
    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="State Machine Pipeline for Form Analysis")
    parser.add_argument("image", help="Path to form image")
    parser.add_argument("--bbox", nargs=4, type=float, default=None,
                       help="Container bbox: x1 y1 x2 y2 (optional - auto-detect if not provided)")
    parser.add_argument("--output", "-o", default="./output",
                       help="Output directory for visualizations")
    parser.add_argument("--ocr-engine", default="paddleocr_service",
                       choices=["easyocr", "tesseract", "paddleocr_service"],
                       help="OCR engine to use (default: paddleocr_service)")
    parser.add_argument("--ocr-service-url", default="http://paddleocr_service:8000",
                       help="URL of OCR service (for paddleocr_service engine)")
    parser.add_argument("--no-demo", action="store_true",
                       help="Disable visualization output")
    parser.add_argument("--skip-ocr", action="store_true",
                       help="Skip OCR stage")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    
    # Run pipeline
    config = PipelineConfig(
        ocr_engine=args.ocr_engine,
        demo_mode=not args.no_demo,
        output_dir=args.output,
        skip_ocr=args.skip_ocr,
        ocr_service_url=args.ocr_service_url,
    )
    
    result = run_state_machine_pipeline(
        image_path=args.image,
        container_bbox=args.bbox,
        config=config,
    )
    
    # Print summary
    print(f"\n=== Pipeline Result ===")
    print(f"Container: {result.container_bbox}")
    print(f"Container source: {result.container_source}")
    if result.s0_container_detected:
        print(f"Container confidence: {result.s0_container_confidence:.2f}")
    print(f"Valid: {result.is_valid}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Atoms: {len(result.atoms)}")
    
    if result.error:
        print(f"Error: {result.error} (stage: {result.stage_failed})")
    
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    exit(main())
