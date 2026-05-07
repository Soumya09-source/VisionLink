import cv2
import numpy as np
from core.ocr_engine import OCREngine, OCRResult
from core.scene_builder import SceneBuilder

print("--- OCR Debugging Verification ---")

# Create a dummy frame (black background)
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Draw an "EXIT" sign on the frame
# White text on red background
cv2.rectangle(frame, (200, 200), (400, 300), (0, 0, 255), -1)
cv2.putText(frame, "EXIT", (220, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

# Simulate YOLO detection
detections = [
    {"class_name": "stop sign", "direction": "center", "confidence": 0.95, "box": (200, 200, 400, 300)}
]

# Initialize components
ocr_engine = OCREngine(ocr_interval=1, text_cooldown_seconds=8.0, gpu=False, debug=True)
scene_builder = SceneBuilder(persistence_frames=3, cooldown_seconds=0.0)

print("\n[STEP 1] OCR Engine Processing")
# Simulate frame 1
ocr_engine.process_frame(frame, detections, 1)

print("\n[STEP 2] Get Speakable Results")
speakable_ocr = ocr_engine.get_speakable_results()
print(f"Speakable OCR: {[r.to_dict() for r in speakable_ocr]}")

print("\n[STEP 3] Scene Builder Integration")
scene = scene_builder.process_frame(detections, ocr_results=speakable_ocr)
print(f"Scene Snapshot: {scene}")

print("\n[STEP 4] Get Speech Update")
# Generate speech from scene using the correct API
from core.scene_builder import summarize_scene
scene = scene_builder.get_current_scene()
speech = summarize_scene(scene)
print(f"Speech: {speech}")

if speech:
    ocr_engine.mark_spoken(speakable_ocr)

print("\n--- Verification Complete ---")
