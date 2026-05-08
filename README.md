# VisionAssist

VisionAssist is a real-time offline-first assistive AI system designed for visually impaired users.  
It combines computer vision, OCR, voice interaction, spatial audio, and OpenClaw intelligence to understand surroundings and communicate useful information naturally.

---

# Features

- Real-time object detection using YOLOv8
- OCR text extraction using EasyOCR
- Structured scene understanding (JSON scene builder)
- Offline voice recognition using Vosk
- Offline speech synthesis
- Direction-aware/spatial audio feedback
- Memory system for remembering objects
- Voice command interaction
- OpenClaw AI integration
- NanoClaw preprocessing pipeline
- Cross-platform support (macOS + Windows)
- Offline-first architecture

---

# System Architecture

Camera
↓
Frame Capture Thread
↓
YOLO Object Detection
↓
OCR Extraction
↓
Scene Builder (Structured JSON)
↓
NanoClaw AI
↓
OpenClaw AI
↓
Response Parser
↓
Alert System
↓
Spatial Audio + TTS
↓
User

---

# Tech Stack

## Computer Vision
- OpenCV
- Ultralytics YOLOv8

## OCR
- EasyOCR

## Speech Recognition
- Vosk

## Speech Output
- pyttsx3
- Native platform audio

## Audio Processing
- NumPy
- sounddevice
- wave

## AI Integration
- OpenClaw

---

# Current Capabilities

- Detects real-world objects
- Identifies object direction:
  - left
  - center
  - right
- Reads visible text from objects/signs
- Generates structured scene JSON
- Accepts voice commands
- Provides contextual audio guidance
- Maintains scene memory
- Integrates OpenClaw for reasoning and intelligent response generation

---

# Example Scene JSON

```json
{
  "timestamp": 1715000000,
  "objects": [
    {
      "name": "person",
      "position": "left",
      "confidence": 0.91
    }
  ],
  "text": [
    {
      "content": "EXIT",
      "position": "center",
      "confidence": 0.92
    }
  ]
}
