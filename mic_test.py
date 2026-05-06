"""
mic_test.py — Standalone microphone diagnostic tool
=====================================================
Run this OUTSIDE of main.py to isolate microphone issues.

Usage:
    python3 mic_test.py
"""

import sys
import time

# Step 1: Check PyAudio and sounddevice availability
print("=== STEP 1: Import Check ===")
try:
    import pyaudio
    print(f"[OK] PyAudio available: {pyaudio.__version__ if hasattr(pyaudio, '__version__') else 'installed'}")
except ImportError as e:
    print(f"[FAIL] PyAudio not found: {e}")
    sys.exit(1)

try:
    import sounddevice as sd
    print(f"[OK] sounddevice available for device listing")
    HAS_SD = True
except ImportError:
    print("[WARN] sounddevice not installed — skipping device list. Install with: pip install sounddevice")
    HAS_SD = False

# Step 2: List all audio devices
print("\n=== STEP 2: Audio Devices ===")
if HAS_SD:
    devices = sd.query_devices()
    print(devices)
    default_input = sd.query_devices(kind='input')
    print(f"\n[Default Input Device]\n  Name: {default_input['name']}")
    print(f"  Max Input Channels: {default_input['max_input_channels']}")
    print(f"  Default Sample Rate: {default_input['default_samplerate']}")
else:
    pa = pyaudio.PyAudio()
    print(f"PyAudio device count: {pa.get_device_count()}")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} | Channels={info['maxInputChannels']} | Rate={int(info['defaultSampleRate'])}")
    pa.terminate()

# Step 3: Try opening a 16kHz mono stream and reading audio for 3 seconds
print("\n=== STEP 3: Open 16kHz Mono Stream ===")
pa = pyaudio.PyAudio()
RATE = 16000
CHUNK = 4000
CHANNELS = 1

try:
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    print(f"[MIC] Stream opened: {RATE} Hz, {CHANNELS} ch, chunk={CHUNK}")
    print(f"[MIC] Reading 3 seconds of audio...")

    import numpy as np
    max_amplitude = 0
    for i in range(int(RATE / CHUNK * 3)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)
        amplitude = np.abs(audio).max()
        max_amplitude = max(max_amplitude, amplitude)
        bar = "#" * int(amplitude / 1000)
        print(f"  Frame {i+1:2d}: amplitude={amplitude:5d}  {bar}", end="\r")

    print(f"\n[MIC] Peak amplitude over 3s: {max_amplitude}")
    if max_amplitude < 100:
        print("[WARN] Very low amplitude — microphone might be muted or wrong device selected!")
    else:
        print("[OK] Microphone is picking up audio.")

    stream.stop_stream()
    stream.close()
except Exception as e:
    print(f"[FAIL] Could not open stream: {e}")
    print("  → Check macOS System Preferences → Security & Privacy → Microphone")
    print("  → Make sure Terminal/VS Code has microphone permission")

pa.terminate()
print("\n=== Mic test complete. ===")
