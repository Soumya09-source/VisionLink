import threading
import subprocess
import time

class TextToSpeech:
    """
    macOS-native TTS system using the built-in 'say' command.
    Replaces the unstable pyttsx3 backend. Uses a daemon thread to keep
    the main YOLO detection loop completely non-blocking, with an 'is_speaking'
    lock to prevent overlapping speech.
    """

    def __init__(self, voice: str = "Samantha", rate: int = 175):
        self.voice = voice
        self.rate = str(rate)
        
        self._is_speaking = False
        self._lock = threading.Lock()
        
        print(f"[TTS] Initialized native macOS speech (voice: {self.voice}, rate: {self.rate})")

    def speak(self, text: str) -> None:
        """
        Non-blocking speak. If already speaking, the new message is ignored
        to prevent overlapping/queue deadlocks.
        """
        if not text or not str(text).strip():
            return
            
        with self._lock:
            if self._is_speaking:
                return  # Skip overlapping speech
            self._is_speaking = True
            
        print(f"[TTS SPEAKING] {text}")
        
        # Fire and forget in a daemon thread so OpenCV loop doesn't block
        thread = threading.Thread(target=self._run_say_command, args=(str(text),), daemon=True)
        thread.start()

    def _run_say_command(self, text: str) -> None:
        """Internal worker to execute the macOS 'say' command."""
        try:
            # -v: voice, -r: words per minute
            subprocess.run(
                ["say", "-v", self.voice, "-r", self.rate, text],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[TTS Playback Error] {e}")
        finally:
            with self._lock:
                self._is_speaking = False

    def stop(self) -> None:
        """
        Compatibility method for main.py.
        Could kill 'say' processes here if hard-stop is needed, 
        but passive cleanup is safer.
        """
        pass

# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    tts = TextToSpeech()
    print("Testing macOS native TTS...")
    tts.speak("Testing VisionAssist. Person ahead, sign ahead says EXIT")
    
    # Wait for thread to finish so script doesn't exit immediately during test
    time.sleep(5)
    print("Test complete.")