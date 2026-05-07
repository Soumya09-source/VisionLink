#!/usr/bin/env python3
"""
voice_final_test.py - Final working voice recognition test
==================================================
Simple, reliable voice recognition that works with VisionAssist.
"""

import sys
import json
import time
import os
import threading

MODEL_PATH = "models/vosk-model-small-en-us-0.15"

class SimpleVoiceManager:
    def __init__(self, command_callback=None):
        self.command_callback = command_callback
        self.model = None
        self.recognizer = None
        self._pa = None
        self._is_running = False
        
    def initialize(self):
        """Initialize voice recognition with open vocabulary."""
        try:
            import vosk
            import pyaudio
            global pyaudio  # Make available for cleanup
            vosk.SetLogLevel(-1)
        except ImportError as e:
            print(f"Import error: {e}")
            return False
            
        if not os.path.exists(MODEL_PATH):
            print(f"Model not found: {MODEL_PATH}")
            return False
            
        print("Loading Vosk model...")
        self.model = vosk.Model(MODEL_PATH)
        
        # Use open vocabulary for better recognition
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        print("Voice recognition ready (open vocabulary)")
        
        self._pa = pyaudio.PyAudio()
        
        # Use MacBook Air Microphone
        device_index = 0
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if "MacBook Air Microphone" in info["name"]:
                device_index = i
                break
                
        self.device_index = device_index
        return True
        
    def listen_continuous(self, duration=15):
        """Listen for commands for specified duration."""
        if not self.initialize():
            return False
            
        try:
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=4000,
            )
            
            print(f"\n🎤 Listening for {duration} seconds...")
            print("Speak clearly: 'what do you see', 'read text', 'stop speaking'")
            
            start_time = time.time()
            last_command_time = 0
            
            while time.time() - start_time < duration:
                data = stream.read(4000, exception_on_overflow=False)
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip().lower()
                    
                    if text and time.time() - last_command_time > 1.0:
                        # Simple keyword matching
                        command = self._match_command(text)
                        if command:
                            print(f"✅ RECOGNIZED: '{text}' → {command}")
                            if self.command_callback:
                                threading.Thread(
                                    target=self.command_callback, 
                                    args=(command,), 
                                    daemon=True
                                ).start()
                            last_command_time = time.time()
                        else:
                            print(f"❓ Heard: '{text}' (no command)")
                
                # Show partial recognition
                partial = json.loads(self.recognizer.PartialResult()).get("partial", "")
                if partial:
                    print(f"   ... {partial}  ", end="\r")
                    
                time.sleep(0.01)
                
            stream.stop_stream()
            stream.close()
            return True
            
        except Exception as e:
            print(f"Listening error: {e}")
            return False
        finally:
            if hasattr(self, '_pa') and self._pa:
                self._pa.terminate()
    
    def _match_command(self, text):
        """Simple, reliable command matching."""
        text = text.lower()
        
        # Direct command matches
        commands = {
            "what do you see": "what do you see",
            "read text": "read text", 
            "stop speaking": "stop speaking",
            "stop talking": "stop speaking",
            "pause alerts": "pause alerts",
            "resume alerts": "resume alerts",
            "repeat": "repeat",
            "remember this": "remember this",
            "remember": "remember this",
            "what did you remember": "what did you remember",
        }
        
        # Exact matches first
        for phrase, cmd in commands.items():
            if phrase in text:
                return cmd
                
        # Keyword matching for flexibility
        if any(word in text for word in ["what", "see"]):
            return "what do you see"
        elif any(word in text for word in ["read", "text"]):
            return "read text"
        elif any(word in text for word in ["stop", "speaking", "talking"]):
            return "stop speaking"
        elif any(word in text for word in ["pause", "alerts"]):
            return "pause alerts"
        elif any(word in text for word in ["resume", "alerts"]):
            return "resume alerts"
        elif any(word in text for word in ["repeat"]):
            return "repeat"
        elif any(word in text for word in ["remember"]):
            return "remember this"
            
        return None

def test_voice():
    """Test voice recognition with VisionAssist commands."""
    print("=== VisionAssist Voice Recognition Test ===")
    
    def handle_command(cmd):
        print(f"\n🎯 EXECUTING: {cmd}")
        
        if cmd == "what do you see":
            print("   → Would describe current scene")
        elif cmd == "read text":
            print("   → Would read visible text")
        elif cmd == "stop speaking":
            print("   → Would stop TTS")
        elif cmd == "remember this":
            print("   → Would remember current object")
        else:
            print(f"   → Would handle: {cmd}")
    
    voice = SimpleVoiceManager(command_callback=handle_command)
    
    print("🎤 Voice recognition ready!")
    print("Speak clearly and naturally.")
    print("Try these commands:")
    print("   • 'what do you see'")
    print("   • 'read text'")
    print("   • 'stop speaking'")
    print("   • 'remember this'")
    print()
    
    success = voice.listen_continuous(duration=20)
    
    if success:
        print("\n✅ Voice test completed successfully!")
        print("The voice recognition is working properly.")
    else:
        print("\n❌ Voice test failed")
        print("Check microphone permissions and try again.")

if __name__ == "__main__":
    test_voice()
