#!/usr/bin/env python3
"""
test_ocr_live.py — Live OCR Debugging with Visual Feedback
======================================================

PURPOSE
-------
Comprehensive OCR testing with real camera input, visual debugging,
and detailed logging to identify and fix OCR issues.

FEATURES
--------
- Live camera feed with OCR overlays
- ROI visualization and saving
- Real-time text detection display
- Detailed debug logging
- Performance monitoring
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path

# Import VisionAssist components
from core.ocr_engine import OCREngine, OCRResult
from detector import ObjectDetector
from core.scene_builder import SceneBuilder, summarize_scene

# Create debug directory
DEBUG_DIR = Path("debug_ocr")
DEBUG_DIR.mkdir(exist_ok=True)

def draw_ocr_overlays(frame, ocr_results, detections):
    """Draw OCR results and detection boxes on frame."""
    overlay = frame.copy()
    
    # Draw YOLO detection boxes (blue)
    for det in detections:
        x1, y1, x2, y2 = det.get("box", (0, 0, 0, 0))
        label = det.get("class_name", "unknown")
        confidence = det.get("confidence", 0.0)
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(overlay, f"{label} {confidence:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw OCR results (green)
    for result in ocr_results:
        if hasattr(result, 'text'):
            text = result.text
            conf = result.confidence
            pos = result.position
        else:
            text = result.get('text', '')
            conf = result.get('confidence', 0.0)
            pos = result.get('position', 'center')
        
        # Position indicator
        h, w = overlay.shape[:2]
        if pos == "left":
            x, y = 50, h - 100
        elif pos == "right":
            x, y = w - 200, h - 100
        else:
            x, y = w//2 - 100, h - 100
        
        # Draw OCR result
        cv2.rectangle(overlay, (x, y), (x + 300, y + 60), (0, 255, 0), -1)
        cv2.putText(overlay, f"OCR: {text}", 
                   (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(overlay, f"Conf: {conf:.2f}", 
                   (x + 5, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return overlay

def save_debug_roi(frame, detections, frame_index):
    """Save ROI crops for debugging."""
    roi_count = 0
    for det in detections:
        label = det.get("class_name", "")
        if label not in ["stop sign", "sign", "book", "laptop", "monitor", "cell phone"]:
            continue  # Only save text-likely objects
        
        x1, y1, x2, y2 = det.get("box", (0, 0, 0, 0))
        
        # Add padding
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            roi_path = DEBUG_DIR / f"roi_{frame_index:04d}_{roi_count:02d}_{label.replace(' ', '_')}.png"
            cv2.imwrite(str(roi_path), roi)
            roi_count += 1
            print(f"[DEBUG] Saved ROI: {roi_path.name}")

def main():
    print("=== Live OCR Debug Test ===")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'd' to toggle debug mode")
    
    # Initialize components
    print("\n[INIT] Initializing components...")
    
    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return
    
    # Object detector
    detector = ObjectDetector(model_path="models/yolov8n.pt", confidence=0.5)
    print("[INIT] YOLO detector loaded")
    
    # OCR engine with debug mode
    ocr_engine = OCREngine(
        ocr_interval=15,  # OCR every 15 frames (~1 second at 15 FPS)
        text_cooldown_seconds=5.0,
        gpu=False,
        debug=True
    )
    print("[INIT] OCR engine configured")
    
    # Scene builder
    scene_builder = SceneBuilder(persistence_frames=3)
    print("[INIT] Scene builder ready")
    
    frame_count = 0
    debug_mode = True
    last_ocr_time = 0
    
    print("\n[START] Starting live OCR test...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Could not read frame")
                break
            
            frame_count += 1
            
            # Run YOLO detection
            detections = detector.detect(frame)
            
            # Convert detections to expected format
            formatted_detections = []
            for det in detections:
                formatted_detections.append({
                    "class_name": det.name,
                    "direction": det.direction,
                    "confidence": det.confidence,
                    "box": det.bbox
                })
            
            # Run OCR
            ocr_results = ocr_engine.process_frame(frame, formatted_detections, frame_count)
            
            # Get speakable results for speech
            speakable_ocr = ocr_engine.get_speakable_results()
            
            # Build scene
            scene = scene_builder.process_frame(formatted_detections, ocr_results=speakable_ocr)
            
            # Generate speech
            speech = summarize_scene(scene)
            
            # Save debug ROI on keyframes
            if frame_count % 15 == 0 and debug_mode:
                save_debug_roi(frame, formatted_detections, frame_count)
            
            # Display results
            overlay = draw_ocr_overlays(frame, speakable_ocr, formatted_detections)
            
            # Add info text
            cv2.putText(overlay, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"OCR Results: {len(speakable_ocr)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Speech: {speech[:50]}...", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("VisionAssist OCR Debug", overlay)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                frame_path = DEBUG_DIR / f"frame_{frame_count:04d}.png"
                cv2.imwrite(str(frame_path), frame)
                print(f"[DEBUG] Saved frame: {frame_path.name}")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[DEBUG] Debug mode: {'ON' if debug_mode else 'OFF'}")
            
            # Mark OCR as spoken (simulate TTS)
            if speakable_ocr and speech:
                ocr_engine.mark_spoken(speakable_ocr)
                print(f"[SPEECH] {speech}")
                last_ocr_time = time.time()
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[CLEANUP] Debug images saved in: {DEBUG_DIR.absolute()}")
        print("[DONE] Test completed")

if __name__ == "__main__":
    main()
