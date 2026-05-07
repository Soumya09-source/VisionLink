#!/usr/bin/env python3
"""
test_voice_simple.py - Simple voice recognition test without grammar constraints
==================================================================
Tests Vosk with open vocabulary to identify recognition issues.
"""

import sys
import json
import time
import os
import numpy as np

MODEL_PATH = "models/vosk-model-small-en-us-0.15"

print("=== Simple Voice Recognition Test ===")
print("Testing Vosk WITHOUT grammar constraints")

# Import checks
try:
    import vosk
    vosk.SetLogLevel(-1)
    print("[OK] vosk imported")
except ImportError:
    print("[FAIL] vosk not installed")
    sys.exit(1)

try:
    import pyaudio
    print("[OK] pyaudio imported")
except ImportError:
    print("[FAIL] pyaudio not installed")
    sys.exit(1)

# Model check
if not os.path.exists(MODEL_PATH):
    print(f"[FAIL] Model not found: {MODEL_PATH}")
    sys.exit(1)

print("[OK] Model path exists")

# Load model
print("[...] Loading model...")
model = vosk.Model(MODEL_PATH)
print("[OK] Model loaded")

# Create recognizer WITHOUT grammar (open vocabulary)
rec = vosk.KaldiRecognizer(model, 16000)
print("[OK] Open vocabulary recognizer created")

# Setup audio
pa = pyaudio.PyAudio()
CHUNK = 4000

# Find best input device
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

# Open stream
stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    input_device_index=best_idx,
    frames_per_buffer=CHUNK,
)

print("[MIC] Stream opened")
print("\n=== LISTENING (Open Vocabulary) ===")
print("Speak anything! Press Ctrl+C to stop.")
print("Expected: Should recognize general speech, not just commands\n")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "").strip()
            if text:
                ts = time.strftime("%H:%M:%S")
                print(f"[OPEN] Recognized: '{text}' [{ts}]")
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "")
            if partial:
                print(f"  [...] {partial}  ", end="\r")

except KeyboardInterrupt:
    print("\n\n[OPEN] Stopped by user.")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("[OPEN] Cleanup complete.")
