#!/usr/bin/env python3
"""
test_ocr_enhanced.py — Enhanced OCR Testing with Improved Preprocessing
====================================================================

PURPOSE
-------
Test OCR system with enhanced preprocessing, lower thresholds, and comprehensive
debugging to identify and fix text recognition issues.

ENHANCEMENTS
-------------
- Aggressive text-likely object filtering
- Enhanced preprocessing with multiple upscaling options
- Lower confidence thresholds for debugging
- Detailed ROI analysis and saving
- Performance benchmarking
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path

# Import VisionAssist components
from core.ocr_engine import OCREngine, OCRResult, TEXT_LIKELY_CLASSES

# Enhanced text-likely classes (more comprehensive)
ENHANCED_TEXT_LIKELY_CLASSES = frozenset({
    # Printed text
    "book", "newspaper", "magazine", "letter",
    # Displays / screens
    "laptop", "tv", "monitor", "cell phone", "keyboard", "tablet",
    # Signage / labels
    "stop sign", "sign", "billboard", "traffic light", "street sign",
    # Containers with labels
    "bottle", "cup", "bowl", "wine glass", "can", "box",
    # Other text sources
    "remote", "clock", "microwave", "oven", "refrigerator", "calendar",
    "poster", "banner", "screen", "display", "label",
})

def create_test_scenes():
    """Create various test scenes with different text challenges."""
    scenes = []
    
    # Scene 1: High contrast EXIT sign
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (200, 150), (440, 250), (0, 0, 255), -1)
    cv2.putText(frame1, "EXIT", (250, 220), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
    scenes.append(("High Contrast EXIT", frame1))
    
    # Scene 2: Small text on laptop
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (150, 100), (490, 350), (50, 50, 50), -1)
    cv2.rectangle(frame2, (160, 110), (480, 340), (200, 200, 200), -1)
    cv2.putText(frame2, "HELLO WORLD", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    scenes.append(("Laptop Screen", frame2))
    
    # Scene 3: Bottle with small label
    frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame3, (280, 200), (360, 400), (0, 100, 255), -1)
    cv2.putText(frame3, "WATER", (285, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    scenes.append(("Bottle Label", frame3))
    
    # Scene 4: Low contrast sign
    frame4 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame4, (200, 150), (440, 250), (100, 100, 100), -1)
    cv2.putText(frame4, "PUSH", (250, 220), cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 200, 200), 8)
    scenes.append(("Low Contrast PUSH", frame4))
    
    return scenes

def test_enhanced_preprocessing():
    """Test different preprocessing techniques."""
    print("=== Testing Enhanced OCR Preprocessing ===")
    
    # Create test scene
    _, test_frame = create_test_scenes()[0]
    
    # Simulate detection
    detections = [{
        "class_name": "stop sign",
        "direction": "center", 
        "confidence": 0.95,
        "box": (200, 150, 440, 250)
    }]
    
    # Test different preprocessing approaches
    preprocessing_methods = [
        ("Original", lambda x: x),
        ("Grayscale", lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)),
        ("Grayscale + Sharpen", lambda x: cv2.addWeighted(
            cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 1.5,
            cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (0, 0), 3), -0.5, 0)),
        ("2x Upscale", lambda x: cv2.resize(x, (x.shape[1]*2, x.shape[0]*2), cv2.INTER_CUBIC)),
        ("3x Upscale", lambda x: cv2.resize(x, (x.shape[1]*3, x.shape[0]*3), cv2.INTER_CUBIC)),
        ("Adaptive Threshold", lambda x: cv2.adaptiveThreshold(
            cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
    ]
    
    debug_dir = Path("debug_preprocessing")
    debug_dir.mkdir(exist_ok=True)
    
    for method_name, preprocess_func in preprocessing_methods:
        print(f"\nTesting: {method_name}")
        
        # Extract ROI
        x1, y1, x2, y2 = detections[0]["box"]
        roi = test_frame[y1:y2, x1:x2]
        
        # Apply preprocessing
        processed = preprocess_func(roi)
        
        # Save for visual inspection
        if len(processed.shape) == 2:  # Grayscale
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            processed_bgr = processed
            
        cv2.imwrite(str(debug_dir / f"preprocess_{method_name.replace(' ', '_').lower()}.png"), processed_bgr)
        
        # Test OCR on this preprocessing
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            raw = reader.readtext(processed, detail=1)
            print(f"  Results: {[(t, float(c)) for _, t, c in raw]}")
        except Exception as e:
            print(f"  Error: {e}")

def test_comprehensive_ocr():
    """Test OCR with comprehensive settings and real scenarios."""
    print("\n=== Comprehensive OCR Testing ===")
    
    # Enhanced OCR engine with lower thresholds
    ocr_engine = OCREngine(
        ocr_interval=1,  # OCR every frame for testing
        text_cooldown_seconds=0.1,  # Very short cooldown for testing
        gpu=False,
        debug=True
    )
    
    # Override text-likely classes temporarily
    import core.ocr_engine
    original_classes = core.ocr_engine.TEXT_LIKELY_CLASSES
    core.ocr_engine.TEXT_LIKELY_CLASSES = ENHANCED_TEXT_LIKELY_CLASSES
    
    try:
        scenes = create_test_scenes()
        
        for scene_name, frame in scenes:
            print(f"\n--- Testing Scene: {scene_name} ---")
            
            # Create multiple detection scenarios
            test_detections = [
                {
                    "class_name": "stop sign" if "EXIT" in scene_name or "PUSH" in scene_name else "laptop",
                    "direction": "center",
                    "confidence": 0.95,
                    "box": (200, 150, 440, 250)
                }
            ]
            
            # Test multiple frames
            for i in range(3):
                print(f"Frame {i+1}:")
                results = ocr_engine.process_frame(frame, test_detections, i+1)
                
                if results:
                    for result in results:
                        print(f"  ✓ Found: '{result.text}' (conf: {result.confidence:.2f}, pos: {result.position})")
                else:
                    print(f"  ✗ No text detected")
                
                time.sleep(0.1)  # Small delay between frames
    
    finally:
        # Restore original classes
        core.ocr_engine.TEXT_LIKELY_CLASSES = original_classes

def test_roi_analysis():
    """Analyze ROI extraction and save debug images."""
    print("\n=== ROI Analysis ===")
    
    debug_dir = Path("debug_roi")
    debug_dir.mkdir(exist_ok=True)
    
    scenes = create_test_scenes()
    
    for scene_name, frame in scenes:
        print(f"\nAnalyzing: {scene_name}")
        
        # Test different padding values
        padding_values = [0, 5, 10, 15, 20]
        
        for padding in padding_values:
            x1, y1, x2, y2 = 200, 150, 440, 250
            
            # Apply padding
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(frame.shape[1], x2 + padding)
            y2_pad = min(frame.shape[0], y2 + padding)
            
            roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Save ROI with padding info
            roi_path = debug_dir / f"roi_{scene_name.replace(' ', '_')}_pad_{padding:02d}.png"
            cv2.imwrite(str(roi_path), roi)
            
            print(f"  Padding {padding:2d}: ROI size {roi.shape[:2]} -> {roi_path.name}")

def main():
    print("=== Enhanced OCR Testing Suite ===")
    
    # Test 1: Preprocessing methods
    test_enhanced_preprocessing()
    
    # Test 2: Comprehensive OCR
    test_comprehensive_ocr()
    
    # Test 3: ROI analysis
    test_roi_analysis()
    
    print("\n=== Testing Complete ===")
    print("Debug images saved in:")
    print("  - debug_preprocessing/")
    print("  - debug_roi/")

if __name__ == "__main__":
    main()
