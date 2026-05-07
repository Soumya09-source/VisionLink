#!/usr/bin/env python3
"""
test_ocr_final.py — Final OCR System Test with Enhanced Engine
================================================================

PURPOSE
-------
Test the enhanced OCR system with improved preprocessing, lower confidence
thresholds, and comprehensive debugging.

TESTS
------
1. Basic OCR functionality with synthetic scenes
2. ROI extraction and preprocessing verification
3. Confidence threshold validation
4. Text-likely object filtering
5. Scene Builder integration
6. Speech output generation
"""

import cv2
import numpy as np
import time
from pathlib import Path

# Import enhanced OCR engine
from core.ocr_engine_fixed import OCREngine, OCRResult
from core.scene_builder import SceneBuilder, summarize_scene

def create_comprehensive_test_scenes():
    """Create comprehensive test scenes with various text challenges."""
    scenes = []
    
    # Scene 1: High contrast EXIT sign
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (200, 150), (440, 250), (0, 0, 255), -1)
    cv2.putText(frame1, "EXIT", (250, 220), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
    scenes.append(("High Contrast EXIT", frame1, "stop sign"))
    
    # Scene 2: Laptop screen with text
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (150, 100), (490, 350), (50, 50, 50), -1)
    cv2.rectangle(frame2, (160, 110), (480, 340), (200, 200, 200), -1)
    cv2.putText(frame2, "HELLO WORLD", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    scenes.append(("Laptop Screen", frame2, "laptop"))
    
    # Scene 3: Bottle with small label
    frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame3, (280, 200), (360, 400), (0, 100, 255), -1)
    cv2.putText(frame3, "WATER", (285, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    scenes.append(("Bottle Label", frame3, "bottle"))
    
    # Scene 4: Low contrast sign
    frame4 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame4, (200, 150), (440, 250), (100, 100, 100), -1)
    cv2.putText(frame4, "PUSH", (250, 220), cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 200, 200), 8)
    scenes.append(("Low Contrast PUSH", frame4, "sign"))
    
    # Scene 5: Multiple text objects
    frame5 = np.zeros((480, 640, 3), dtype=np.uint8)
    # EXIT sign
    cv2.rectangle(frame5, (50, 100), (200, 180), (0, 0, 255), -1)
    cv2.putText(frame5, "EXIT", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    # Laptop screen
    cv2.rectangle(frame5, (250, 120), (450, 300), (50, 50, 50), -1)
    cv2.putText(frame5, "MENU", (300, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    scenes.append(("Multiple Objects", frame5, "mixed"))
    
    return scenes

def test_enhanced_ocr_system():
    """Test the enhanced OCR system comprehensively."""
    print("=== Enhanced OCR System Test ===")
    
    # Initialize enhanced OCR engine
    ocr_engine = OCREngine(
        ocr_interval=1,  # OCR every frame for testing
        text_cooldown_seconds=0.1,  # Very short cooldown for testing
        gpu=False,
        debug=True
    )
    
    # Initialize scene builder
    scene_builder = SceneBuilder(persistence_frames=3)
    
    # Create test scenes
    scenes = create_comprehensive_test_scenes()
    
    debug_dir = Path("debug_ocr_final")
    debug_dir.mkdir(exist_ok=True)
    
    total_tests = 0
    successful_detections = 0
    successful_ocr = 0
    
    for scene_name, frame, expected_object in scenes:
        print(f"\n--- Testing Scene: {scene_name} ---")
        total_tests += 1
        
        # Create appropriate detections
        if expected_object == "mixed":
            detections = [
                {"class_name": "stop sign", "direction": "left", "confidence": 0.95, "box": (50, 100, 200, 180)},
                {"class_name": "laptop", "direction": "center", "confidence": 0.90, "box": (250, 120, 450, 300)}
            ]
        else:
            detections = [{
                "class_name": expected_object,
                "direction": "center",
                "confidence": 0.95,
                "box": (200, 150, 440, 250)
            }]
        
        # Save original frame
        frame_path = debug_dir / f"scene_{scene_name.replace(' ', '_')}_original.png"
        cv2.imwrite(str(frame_path), frame)
        
        # Test OCR processing
        try:
            ocr_results = ocr_engine.process_frame(frame, detections, total_tests)
            
            if ocr_results:
                successful_ocr += 1
                print(f"  ✓ OCR Results: {[r.text for r in ocr_results]}")
                
                # Save ROI with OCR results overlay
                overlay_frame = frame.copy()
                for result in ocr_results:
                    cv2.putText(overlay_frame, f"OCR: {result.text} ({result.confidence:.2f})", 
                               (10, 30 + len(ocr_results) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                overlay_path = debug_dir / f"scene_{scene_name.replace(' ', '_')}_with_ocr.png"
                cv2.imwrite(str(overlay_path), overlay_frame)
                
            else:
                print(f"  ✗ No OCR results detected")
            
            # Test scene builder integration
            speakable_ocr = ocr_engine.get_speakable_results()
            scene = scene_builder.process_frame(detections, ocr_results=speakable_ocr)
            speech = summarize_scene(scene)
            
            if speech:
                print(f"  ✓ Speech: {speech}")
                successful_detections += 1
                ocr_engine.mark_spoken(speakable_ocr)
            else:
                print(f"  ✗ No speech generated")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Total Scenes Tested: {total_tests}")
    print(f"Successful OCR Detections: {successful_ocr}/{total_tests}")
    print(f"Successful Speech Generation: {successful_detections}/{total_tests}")
    print(f"Debug Images Saved: {debug_dir.absolute()}")
    
    # Test specific scenarios
    print(f"\n=== Specific Test Results ===")
    test_specific_scenarios(ocr_engine, scene_builder, debug_dir)

def test_specific_scenarios(ocr_engine, scene_builder, debug_dir):
    """Test specific OCR scenarios."""
    
    print("\n--- Testing Low Confidence Threshold ---")
    # Create very low contrast text
    low_contrast_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(low_contrast_frame, (200, 150), (440, 250), (80, 80, 80), -1)
    cv2.putText(low_contrast_frame, "TEST", (250, 220), cv2.FONT_HERSHEY_SIMPLEX, 3, (90, 90, 90), 8)
    
    detections = [{
        "class_name": "sign",
        "direction": "center",
        "confidence": 0.95,
        "box": (200, 150, 440, 250)
    }]
    
    ocr_results = ocr_engine.process_frame(low_contrast_frame, detections, 999)
    if ocr_results:
        print(f"  ✓ Low confidence text detected: {[r.text for r in ocr_results]}")
    else:
        print(f"  ✗ Low confidence text not detected")
    
    print("\n--- Testing Small Text Enhancement ---")
    # Create very small text
    small_text_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(small_text_frame, (280, 200), (360, 280), (0, 100, 255), -1)
    cv2.putText(small_text_frame, "TINY", (290, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    small_detections = [{
        "class_name": "bottle",
        "direction": "center",
        "confidence": 0.95,
        "box": (280, 200, 360, 280)
    }]
    
    small_ocr_results = ocr_engine.process_frame(small_text_frame, small_detections, 1000)
    if small_ocr_results:
        print(f"  ✓ Small text detected: {[r.text for r in small_ocr_results]}")
    else:
        print(f"  ✗ Small text not detected")

def main():
    """Main test function."""
    print("=== Final OCR System Verification ===")
    print("Testing enhanced OCR engine with:")
    print("- Lower confidence threshold (0.30)")
    print("- Enhanced preprocessing (CLAHE, upscaling)")
    print("- Comprehensive text-likely object classes")
    print("- Detailed debug logging")
    
    test_enhanced_ocr_system()
    
    print(f"\n=== Testing Complete ===")
    print("Check debug_ocr_final/ directory for:")
    print("- Original test scenes")
    print("- OCR result overlays")
    print("- ROI extractions")

if __name__ == "__main__":
    main()
