#!/usr/bin/env python3
"""
Тестовый скрипт для State Machine Pipeline.

Использование:
    python scripts/test_state_machine_pipeline.py

Или с конкретным изображением:
    python scripts/test_state_machine_pipeline.py path/to/image.png --bbox x1 y1 x2 y2
"""

import sys
import os
import logging
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.atoms_v2.experimental_v2 import (
    run_state_machine_pipeline,
    PipelineConfig,
    PipelineResult,
)


def print_result(result: PipelineResult, verbose: bool = True) -> None:
    """Красивый вывод результата."""
    print("\n" + "=" * 60)
    print("STATE MACHINE PIPELINE RESULT")
    print("=" * 60)
    
    if result.error:
        print(f"\n❌ ERROR at stage {result.stage_failed}:")
        print(f"   {result.error}")
        return
    
    print(f"\n✅ Pipeline completed successfully")
    print(f"   Container: {result.container_bbox}")
    print(f"   Container source: {result.container_source}")
    if result.s0_container_detected:
        print(f"   Container confidence: {result.s0_container_confidence:.2%}")
    print(f"   Valid: {result.is_valid}")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Total atoms: {len(result.atoms)}")
    
    # S1 Summary
    if result.s1_result:
        print(f"\n📐 S1 Visual Geometry:")
        print(f"   Elements detected: {len(result.s1_result.visual_elements)}")
        if result.s1_result.diagnostics.get("by_type"):
            print(f"   By type: {result.s1_result.diagnostics['by_type']}")
    
    # S2 Summary
    if result.s2_result:
        print(f"\n📝 S2 OCR:")
        print(f"   Blocks: {len(result.s2_result.ocr_blocks)}")
        print(f"   Language: {result.s2_result.language.primary}")
    
    # S3 Summary
    if result.s3_result:
        print(f"\n📊 S3 Rows:")
        print(f"   Rows: {len(result.s3_result.rows)}")
        if result.s3_result.diagnostics.get("row_types"):
            print(f"   Types: {result.s3_result.diagnostics['row_types']}")
    
    # S4 Summary
    if result.s4_result:
        print(f"\n🏷️  S4 Slots:")
        print(f"   Assignments: {result.s4_result.diagnostics.get('slot_counts', {})}")
        print(f"   Bindings: {result.s4_result.diagnostics.get('bindings', 0)}")
    
    # S5 Summary
    if result.s5_result:
        print(f"\n🔄 S5 Patterns:")
        print(f"   Patterns: {len(result.s5_result.patterns)}")
        if result.s5_result.diagnostics.get("by_type"):
            print(f"   Types: {result.s5_result.diagnostics['by_type']}")
    
    # S6 Summary
    if result.s6_result:
        print(f"\n✔️  S6 Validation:")
        print(f"   Errors: {result.s6_result.diagnostics.get('errors', 0)}")
        print(f"   Warnings: {result.s6_result.diagnostics.get('warnings', 0)}")
        print(f"   Info: {result.s6_result.diagnostics.get('info', 0)}")
        
        if verbose and result.s6_result.flags:
            print("\n   Flags:")
            for flag in result.s6_result.flags:
                icon = "❌" if flag.flag_type == "error" else "⚠️" if flag.flag_type == "warning" else "ℹ️"
                print(f"     {icon} [{flag.code}] {flag.message}")
    
    # Atoms
    if verbose and result.atoms:
        print(f"\n📦 Atoms ({len(result.atoms)}):")
        for i, atom in enumerate(result.atoms[:10]):  # show first 10
            slot = atom.get("slot", "?")
            row = atom.get("row_index", "?")
            text = atom.get("text", "")[:30] if atom.get("text") else ""
            print(f"   {i+1}. {slot} (row {row})" + (f' "{text}..."' if text else ""))
        if len(result.atoms) > 10:
            print(f"   ... and {len(result.atoms) - 10} more")
    
    print("\n" + "=" * 60)


def test_with_demo_forms():
    """Тест на demo формах."""
    demo_dir = project_root / "data" / "demo_forms" / "images"
    
    if not demo_dir.exists():
        print(f"Demo forms directory not found: {demo_dir}")
        return
    
    images = list(demo_dir.glob("*.png"))[:3]  # первые 3 формы
    
    if not images:
        print("No PNG images found in demo directory")
        return
    
    print(f"\nTesting with {len(images)} demo forms from {demo_dir}\n")
    
    output_dir = project_root / "data" / "state_machine_output"
    output_dir.mkdir(exist_ok=True)
    
    config = PipelineConfig(
        demo_mode=True,
        output_dir=str(output_dir),
        ocr_engine="easyocr",
        skip_ocr=True,  # пока без OCR для быстрого теста
        auto_detect_container=True,  # автоматическое определение контейнера
        use_full_image_fallback=True,
    )
    
    for img_path in images:
        print(f"\n{'='*60}")
        print(f"Testing: {img_path.name}")
        print("=" * 60)
        
        # Контейнер определяется автоматически (S0)
        result = run_state_machine_pipeline(
            image_path=str(img_path),
            container_bbox=None,  # авто-детект!
            config=config,
        )
        
        print_result(result, verbose=False)
    
    print(f"\n\n📁 Visualizations saved to: {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test State Machine Pipeline")
    parser.add_argument("image", nargs="?", help="Path to form image (optional)")
    parser.add_argument("--bbox", nargs=4, type=float, default=None,
                       help="Container bbox: x1 y1 x2 y2 (auto-detect if not provided)")
    parser.add_argument("-o", "--output", default="./state_machine_output", help="Output directory")
    parser.add_argument("--ocr", choices=["easyocr", "tesseract", "skip"], default="skip",
                       help="OCR engine (default: skip)")
    parser.add_argument("--no-auto-detect", action="store_true",
                       help="Disable auto-detection, use full image instead")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    
    if args.image:
        # Тест конкретного изображения
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            return 1
        
        # Контейнер: переданный bbox или auto-detect
        container_bbox = args.bbox  # может быть None → auto-detect
        
        config = PipelineConfig(
            demo_mode=True,
            output_dir=args.output,
            skip_ocr=(args.ocr == "skip"),
            ocr_engine=args.ocr if args.ocr != "skip" else "easyocr",
            auto_detect_container=not args.no_auto_detect,
            use_full_image_fallback=True,
        )
        
        result = run_state_machine_pipeline(
            image_path=str(img_path),
            container_bbox=container_bbox,
            config=config,
        )
        
        print_result(result, verbose=args.verbose)
        
        print(f"\n📁 Visualizations saved to: {args.output}")
        
        return 0 if result.is_valid else 1
    else:
        # Тест на demo формах
        test_with_demo_forms()
        return 0


if __name__ == "__main__":
    sys.exit(main())
