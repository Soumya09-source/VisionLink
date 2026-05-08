# VisionAssist

## Problem

Visually impaired users often struggle to understand their surroundings in real time.

Existing assistive tools usually have one or more of these limitations:

- Require internet connectivity
- Provide delayed responses
- Lack contextual understanding
- Cannot describe object positions clearly
- Do not support natural voice interaction
- Fail to read environmental text effectively
- Lack intelligent scene reasoning

This creates difficulty in:
- indoor navigation
- object identification
- reading signs/text
- locating personal items
- understanding dynamic environments

---

# Solution

VisionAssist is a real-time offline-first assistive AI system designed to help visually impaired users perceive and understand their surroundings naturally.

The system combines:
- computer vision
- OCR
- voice interaction
- spatial audio
- structured scene understanding
- OpenClaw AI reasoning

to create an intelligent assistive perception pipeline.

VisionAssist can:
- detect objects
- identify object direction
- read text from signs/screens/books
- respond to voice commands
- generate contextual scene summaries
- provide directional audio guidance
- maintain structured scene memory

---

# Features

- Real-time YOLO object detection
- OCR text extraction using EasyOCR
- Offline voice recognition using Vosk
- Natural speech output
- Spatial/directional audio
- Structured scene JSON generation
- OpenClaw AI integration
- NanoClaw preprocessing pipeline
- Cross-platform support (macOS + Windows)
- Offline-first architecture

---

# System Workflow

Camera
↓
YOLO Detection
↓
OCR Extraction
↓
Scene Builder (JSON)
↓
NanoClaw AI
↓
OpenClaw AI
↓
Response Parser
↓
Spatial Audio + TTS
↓
User

---

# Tech Stack

## Computer Vision
- OpenCV
- YOLOv8 (Ultralytics)

## OCR
- EasyOCR

## Voice Recognition
- Vosk

## Speech Output
- pyttsx3

## Audio
- NumPy
- sounddevice

## AI Integration
- OpenClaw
- NanoClaw

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd VisionAssist
```

---

# 2. Create Virtual Environment

## macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Install OpenClaw

Follow official OpenClaw installation instructions.

Example:

```bash
curl -fsSL https://openclaw.ai/install.sh | bash
```

---

# 5. Download Required Models

## YOLO Model

Place:

yolov8n.pt

inside:

/models

## Vosk Model

Download a Vosk English model and place it inside:

/models/vosk

---

# 6. Enable Permissions

## macOS

Enable:
- Camera permission
- Microphone permission

Path:

System Settings → Privacy & Security

## Windows

Allow:
- Camera access
- Microphone access

---

# Running VisionAssist

```bash
python main.py
```

---

# Usage Instructions

## Voice Commands

Examples:

- "What do you see?"
- "Read text"
- "Find bottle"
- "Remember this"
- "Stop speaking"

---

# OCR Usage

VisionAssist can detect and read:
- EXIT signs
- books
- screens
- labels
- packages
- notices

Example output:

```text
"EXIT sign ahead."
```

---

# Spatial Audio Usage

Wear headphones for best experience.

Audio is direction-aware:
- left-side objects → stronger left audio
- right-side objects → stronger right audio
- center objects → balanced audio

---

# Scene Builder Example

```json
{
  "objects": [
    {
      "name": "person",
      "position": "left"
    }
  ],
  "text": [
    {
      "content": "EXIT",
      "position": "center"
    }
  ]
}
```

---

# Cross-Platform Support

Supported:
- macOS
- Windows

The system includes:
- platform-aware camera backends
- platform-safe audio handling
- cross-platform OCR pipeline

---

# Future Improvements

- Advanced navigation system
- Persistent visual memory
- Object tracking
- Hazard prioritization
- Edge deployment
- Smarter OpenClaw reasoning

---

# Vision

VisionAssist aims to become a:
- real-time
- context-aware
- offline-first
- assistive AI perception system

capable of:
- seeing
- reading
- reasoning
- guiding
- remembering
- communicating naturally

for visually impaired users.
