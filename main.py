"""
main.py — VisionAssist Entry Point
====================================

Full pipeline:
    Camera → YOLO → SceneBuilder → OCREngine → TTS

Run:
    python main.py
Press 'q' in the OpenCV window to quit.

DEBUG FLAGS (set to True to print verbose OCR trace):
    OCR_DEBUG = True    ← prints every stage of the OCR pipeline
"""

import cv2

from camera import CameraStream
from detector import ObjectDetector
from tts import TextToSpeech
from core.scene_builder import SceneBuilder
from core.ocr_engine import OCREngine

# ── Set True to see the full OCR trace in the terminal ──────────────────────
OCR_DEBUG: bool = True


def main():
    print("Initialising VisionAssist …")

    camera   = CameraStream(index=0)
    detector = ObjectDetector()
    tts      = TextToSpeech()

    # ---------------------------------------------------------------
    # SceneBuilder — temporal smoothing + speech cooldown
    # ---------------------------------------------------------------
    scene_builder = SceneBuilder(
        persistence_frames=8,
        cooldown_seconds=5.0,
    )

    # ---------------------------------------------------------------
    # OCREngine — text reading inside YOLO bounding boxes
    #
    #   ocr_interval          — run OCR every N frames (30 ≈ every 2 s)
    #   text_cooldown_seconds — don't re-announce same text for N s
    #   gpu                   — set True only if you have a CUDA GPU
    # ---------------------------------------------------------------
    ocr_engine = OCREngine(
        ocr_interval=30,
        text_cooldown_seconds=8.0,
        gpu=False,
        debug=OCR_DEBUG,
    )

    frame_index: int = 0
    print("Running — press 'q' to quit.")

    while True:
        frame = camera.get_frame()

        # Handle missing / empty frames without freezing the CPU
        if frame is None or frame.size == 0:
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            continue

        frame_index += 1

        # ---------------------------------------------------------
        # *** CRITICAL FIX ***
        # Make a clean copy of the frame BEFORE YOLO draws on it.
        # OCR must see the real-world image, not the annotated one.
        # detector.detect() draws bounding boxes and labels in-place,
        # so without this copy OCR would read "person - center 91%"
        # from the YOLO annotations.
        # ---------------------------------------------------------
        clean_frame = frame.copy()

        # 1. YOLO detection — annotates `frame` in-place
        detections = detector.detect(frame)

        # 2. OCR — runs on the CLEAN frame every `ocr_interval` frames
        ocr_engine.process_frame(clean_frame, detections, frame_index)

        # Get OCR results that have passed their cooldown and are ready to speak
        speakable_ocr = ocr_engine.get_speakable_results()

        # 3. Scene Builder — temporal smoothing + scene snapshot
        scene = scene_builder.process_frame(detections, ocr_results=speakable_ocr)

        # 4. Speech decision — only if the scene meaningfully changed
        speech = scene_builder.get_speech_update()

        if speech:
            print(f"[Scene] {speech}")
            tts.speak(speech)
            # Only commit OCR cooldowns AFTER speech is confirmed.
            # This prevents burning a text's cooldown when SceneBuilder
            # suppressed the announcement (scene not changed enough).
            ocr_engine.mark_spoken(speakable_ocr)

        # 5. Display annotated frame
        cv2.imshow("VisionAssist", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)   # macOS: helps GUI close cleanly
    print("Stopped.")


if __name__ == "__main__":
    main()