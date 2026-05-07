#!/usr/bin/env python3
"""
fix_voice_test.py - Quick voice recognition fix test
==================================================
Tests voice recognition with open vocabulary to verify audio works.
"""

import sys
import json
import time
import os
import numpy as np

MODEL_PATH = "models/vosk-model-small-en-us-0.15"

print("=== Voice Recognition Fix Test ===")

# Import and setup
try:
    import vosk
    import pyaudio
    vosk.SetLogLevel(-1)
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load model
if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    sys.exit(1)

print("Loading model...")
model = vosk.Model(MODEL_PATH)

# Test 1: Open vocabulary (should work if audio is good)
print("\n--- Test 1: Open Vocabulary ---")
rec_open = vosk.KaldiRecognizer(model, 16000)

# Test 2: Grammar constrained (current implementation)  
print("--- Test 2: Grammar Constrained ---")
grammar = json.dumps([
    "what do you see", "read text", "stop speaking", 
    "pause alerts", "resume alerts", "repeat",
    "remember this", "what did you remember",
    "find person", "find bottle", "find chair",
    "forget person", "forget bottle", "forget chair",
    "[unk]"
])
rec_grammar = vosk.KaldiRecognizer(model, 16000, grammar)

# Setup audio
pa = pyaudio.PyAudio()
CHUNK = 4000

# Use MacBook Air Microphone
best_idx = 0
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if "MacBook Air Microphone" in info["name"]:
        best_idx = i
        break

stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    input_device_index=best_idx,
    frames_per_buffer=CHUNK,
)

print(f"Using microphone: {pa.get_device_info_by_index(best_idx)['name']}")
print("\n=== SPEAK TEST PHRASES ===")
print("1. 'hello world' (test open vocabulary)")
print("2. 'what do you see' (test grammar)")
print("3. 'stop speaking' (test grammar)")
print("Testing for 15 seconds...\n")

start_time = time.time()
test_count = 0

try:
    while time.time() - start_time < 15:
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Test both recognizers
        if rec_open.AcceptWaveform(data):
            result = json.loads(rec_open.Result())
            text = result.get("text", "").strip()
            if text:
                print(f"[OPEN] '{text}'")
                test_count += 1
        
        if rec_grammar.AcceptWaveform(data):
            result = json.loads(rec_grammar.Result())
            text = result.get("text", "").strip()
            if text and text != "[unk]":
                print(f"[GRAMMAR] '{text}' ✓")
                test_count += 1
        
        # Show partial results
        partial_open = json.loads(rec_open.PartialResult()).get("partial", "")
        partial_grammar = json.loads(rec_grammar.PartialResult()).get("partial", "")
        
        if partial_open and len(partial_open) > 2:
            print(f"  [...] {partial_open} (open)", end="\r")
        elif partial_grammar and len(partial_grammar) > 2:
            print(f"  [...] {partial_grammar} (grammar)", end="\r")
            
except KeyboardInterrupt:
    print("\n\nTest stopped by user")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()

print(f"\n=== Test Results ===")
print(f"Tests completed: {test_count}")
if test_count > 0:
    print("✅ Voice recognition is working!")
else:
    print("❌ No speech recognized - check microphone")
    print("   • Speak louder and closer to microphone")
    print("   • Check macOS camera/microphone permissions")
