"""
vosk_test.py — Standalone Vosk speech recognition test
========================================================
Tests the full mic → Vosk pipeline OUTSIDE of main.py.
Speak clearly after "Listening..." appears. Press Ctrl+C to stop.

Usage:
    python3 vosk_test.py
"""

import sys
import json
import time
import os
import numpy as np

MODEL_PATH = "models/vosk-model-small-en-us-0.15"

# ── Step 1: Import checks ─────────────────────────────────────────────────────
print("=== STEP 1: Import Check ===")
try:
    import vosk
    vosk.SetLogLevel(-1)
    print("[OK] vosk imported")
except ImportError:
    print("[FAIL] vosk not installed. Run: pip install vosk")
    sys.exit(1)

try:
    import pyaudio
    print("[OK] pyaudio imported")
except ImportError:
    print("[FAIL] pyaudio not installed. Run: brew install portaudio && pip install pyaudio")
    sys.exit(1)

# ── Step 2: Model check ───────────────────────────────────────────────────────
print(f"\n=== STEP 2: Model Check ===")
if not os.path.exists(MODEL_PATH):
    print(f"[FAIL] Model not found at: {MODEL_PATH}")
    print("  Download: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
    sys.exit(1)
print(f"[OK] Model path exists: {MODEL_PATH}")

print("[...] Loading model (may take 1-2s)...")
model = vosk.Model(MODEL_PATH)
print("[OK] Model loaded successfully")

# ── Step 3: Recognizer setup ──────────────────────────────────────────────────
print(f"\n=== STEP 3: Recognizer Setup ===")
RATE = 16000

# Grammar-constrained: vastly more accurate than open-vocab for small command sets
GRAMMAR = json.dumps([
    "what do you see",
    "read text",
    "stop speaking",
    "pause alerts",
    "resume alerts",
    "repeat",
    "remember this",
    "what did you remember",
    "find bottle",   "find person",  "find chair",   "find laptop",
    "find cup",      "find book",    "find phone",   "find bag",
    "forget bottle", "forget person","forget chair", "forget laptop",
    "[unk]",
])

rec = vosk.KaldiRecognizer(model, RATE, GRAMMAR)
print(f"[OK] Grammar-constrained recognizer at {RATE} Hz")

# ── Step 4: Microphone setup ──────────────────────────────────────────────────
print(f"\n=== STEP 4: Microphone Setup ===")
pa = pyaudio.PyAudio()
CHUNK = 4000

# Auto-detect: prefer macbook/built-in over bluetooth or iPhone
best_idx = pa.get_default_input_device_info()["index"]
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        name = info["name"].lower()
        if any(kw in name for kw in ("macbook", "built-in", "internal")):
            best_idx = i
            break

dev_name = pa.get_device_info_by_index(best_idx)["name"]
print(f"[MIC] Using device [{best_idx}]: {dev_name}")

try:
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        input_device_index=best_idx,
        frames_per_buffer=CHUNK,
    )
    print(f"[MIC] Stream opened at {RATE} Hz, chunk={CHUNK}")

    # Quick amplitude check — 0.5 second sanity read
    print("[MIC] Checking amplitude (0.5s)...")
    amp_samples = []
    for _ in range(int(RATE / CHUNK * 0.5)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        amp = np.abs(np.frombuffer(data, dtype=np.int16)).max()
        amp_samples.append(amp)
    peak = max(amp_samples)
    print(f"[MIC] Peak amplitude: {peak}")
    if peak < 50:
        print("[WARN] ⚠ Very low amplitude — check:")
        print("       • macOS System Settings → Privacy → Microphone → allow Terminal")
        print("       • Input device is not muted")
    else:
        print("[OK] Microphone is live")

except Exception as e:
    print(f"[FAIL] Cannot open microphone: {e}")
    pa.terminate()
    sys.exit(1)

# ── Step 5: Live recognition loop ─────────────────────────────────────────────
print(f"\n=== STEP 5: Live Recognition ===")
print("Speak now! (press Ctrl+C to stop)\n")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "").strip()
            if text and text != "[unk]":
                ts = time.strftime("%H:%M:%S")
                print(f"\n[VOSK] ✓ Recognized: '{text}'  [{ts}]")
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "")
            if partial and partial != "[unk]":
                print(f"  [...] {partial}  ", end="\r")

except KeyboardInterrupt:
    print("\n\n[VOSK] Stopped by user.")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("[VOSK] Cleanup complete.")
