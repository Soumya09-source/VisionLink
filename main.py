"""
main.py — VisionAssist Entry Point
====================================

Full pipeline:
    Camera → YOLO → SceneBuilder → AlertSystem → SpatialAudio
    Voice → Vosk → CommandRouter → SpatialAudio
    Scene → VisualMemory → JSON store ↔ Voice Commands

Run:
    python main.py
Press 'q' in the OpenCV window to quit.

DEBUG FLAGS (set to True to print verbose OCR trace):
    OCR_DEBUG = True    ← prints every stage of the OCR pipeline
"""

import cv2

from camera import CameraStream
from detector import ObjectDetector
from feedback.tts import CrossPlatformTTS
from core.scene_builder import SceneBuilder, summarize_scene
from core.ocr_engine import OCREngine
from feedback.alert_system import AlertSystem
from input.voice_commands import VoiceCommandManager
from memory.visual_memory import VisualMemory

# ── Set True to see the full OCR trace in the terminal ──────────────────────
OCR_DEBUG: bool = True


def main():
    print("Initialising VisionAssist …")

    camera   = CameraStream(index=0)
    detector = ObjectDetector()
    tts = CrossPlatformTTS()

    # ---------------------------------------------------------------
    # SceneBuilder — temporal smoothing (cooldown handled by AlertSystem now)
    # ---------------------------------------------------------------
    scene_builder = SceneBuilder(
        persistence_frames=8,
        cooldown_seconds=0.0, # Cooldown moved to AlertSystem
    )

    # ---------------------------------------------------------------
    # AlertSystem — priority filtering and interruption logic
    # ---------------------------------------------------------------
    alert_system = AlertSystem(
        cooldown_seconds=5.0,
        critical_cooldown_seconds=3.0,
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

    alerts_paused = False
    last_spoken_text = ""
    last_spoken_dir = "center"

    # Persistent visual memory — survives process restarts
    visual_memory = VisualMemory(store_path="memory_store.json")

    def handle_command(cmd):
        nonlocal alerts_paused, last_spoken_text, last_spoken_dir
        print(f"[ACTION ROUTER] Executing: {cmd}")
        
        if cmd == "stop speaking":
            tts.interrupt()
            
        elif cmd == "pause alerts":
            alerts_paused = True
            tts.speak("Alerts paused.", "center", interrupt=True)
            
        elif cmd == "resume alerts":
            alerts_paused = False
            tts.speak("Alerts resumed.", "center", interrupt=True)
            
        elif cmd == "what do you see":
            scene = scene_builder.get_current_scene()
            speech = summarize_scene(scene)
            if speech:
                tts.speak(speech, "center", interrupt=True)
                last_spoken_text = speech
                last_spoken_dir = "center"
            else:
                tts.speak("I don't see anything.", "center", interrupt=True)

        elif cmd == "read text":
            scene = scene_builder.get_current_scene()
            texts = scene.get("text", [])
            if texts:
                speech = ", ".join(t.get("content", "") for t in texts if t.get("content"))
                speech = f"Text says: {speech}"
                tts.speak(speech, "center", interrupt=True)
                last_spoken_text = speech
                last_spoken_dir = "center"
            else:
                tts.speak("I don't see any text.", "center", interrupt=True)
                
        elif cmd == "repeat":
            if last_spoken_text:
                tts.speak(last_spoken_text, last_spoken_dir, interrupt=True)
            else:
                tts.speak("Nothing to repeat.", "center", interrupt=True)

        # ── Memory commands ──────────────────────────────────────────
        elif cmd == "remember this":
            scene = scene_builder.get_current_scene()
            result = VisualMemory.pick_priority_object(scene)
            if result:
                label, position = result
                # Pull any OCR text associated with this object from canonical text list
                ocr_context = None
                for t in scene.get("text", []):
                    if t.get("source_object", "") == label:
                        ocr_context = t.get("content")
                        break
                visual_memory.remember_object(label, position, ocr_context)
                tts.speak(f"{label} remembered.", "center", interrupt=True)
            else:
                tts.speak("Nothing visible to remember.", "center", interrupt=True)

        elif cmd == "what did you remember":
            memories = visual_memory.get_all_memories()
            if memories:
                labels = [m["label"] for m in memories]
                if len(labels) == 1:
                    summary = labels[0]
                else:
                    summary = ", ".join(labels[:-1]) + " and " + labels[-1]
                tts.speak(f"I remember {summary}.", "center", interrupt=True)
            else:
                tts.speak("I don't remember anything yet.", "center", interrupt=True)

        elif cmd.startswith("find "):
            target = cmd[len("find "):].strip()
            if not target:
                tts.speak("Please say what to find.", "center", interrupt=True)
            else:
                # First: check live scene (new canonical format)
                scene = scene_builder.get_current_scene()
                live_objects = scene.get("objects", [])
                matches = [o for o in live_objects if o.get("label") == target]
                if matches:
                    pos = matches[0]["position"]
                    print(f"[MEMORY] Match found: {target} on {pos}")
                    tts.speak(f"{target} is {pos}.", pos, interrupt=True)
                else:
                    # Fall back: check persistent memory
                    mem = visual_memory.find_memory(target)
                    if mem:
                        pos = mem["position"]
                        print(f"[MEMORY] Recalled from store: {target} was on {pos}")
                        tts.speak(f"I last saw {target} on the {pos}.", pos, interrupt=True)
                    else:
                        tts.speak(f"I don't know where {target} is.", "center", interrupt=True)

        elif cmd.startswith("forget "):
            target = cmd[len("forget "):].strip()
            if not target:
                tts.speak("Please say what to forget.", "center", interrupt=True)
            else:
                removed = visual_memory.forget_memory(target)
                if removed:
                    tts.speak(f"{target} forgotten.", "center", interrupt=True)
                else:
                    tts.speak(
                        f"I don't have {target} in memory.", "center", interrupt=True
                    )

    voice_manager = VoiceCommandManager(command_callback=handle_command)

    frame_index: int = 0
    print("Running — press 'q' to quit, press 'v' to issue a voice command.")

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

        # 3. Scene Builder — temporal smoothing + scene snapshot (canonical JSON)
        scene = scene_builder.process_frame(detections, ocr_results=speakable_ocr)

        # ── 3.1 Debug: Print canonical scene JSON ─────────────────────────────
        import json
        print(f"\n--- [SCENE FRAME {frame_index}] ---")
        print(json.dumps(scene, indent=2))
        print("------------------------------")

        # 4. Alert System — evaluate priorities, suppress noise, generate speech
        speech, is_critical = alert_system.process_scene(scene)

        if speech and not alerts_paused:
            # ── 4.1 Deduce dominant direction from JSON instead of string ─────
            # We look at the highest-priority objects in the scene
            objects = scene.get("objects", [])
            if objects:
                direction = objects[0]["position"]
            else:
                texts = scene.get("text", [])
                direction = texts[0]["position"] if texts else "center"

            if is_critical:
                print(f"[CRITICAL ALERT] {speech}")
            else:
                print(f"[Scene Summary] {speech}")
                
            tts.speak(speech, direction, interrupt=is_critical)
            last_spoken_text = speech
            last_spoken_dir = direction
            
            # Only commit OCR cooldowns AFTER speech is confirmed.
            # This prevents burning a text's cooldown when the system
            # suppressed the announcement (e.g. low priority).
            ocr_engine.mark_spoken(speakable_ocr)

        # 5. Display annotated frame
        cv2.imshow("VisionAssist", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("v"):
            voice_manager.listen_once(timeout=5.0)

    camera.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)   # macOS: helps GUI close cleanly
    print("Stopped.")


if __name__ == "__main__":
    main()