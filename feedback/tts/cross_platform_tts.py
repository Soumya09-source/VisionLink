import time
import logging

from .queue_manager import SpeechQueueManager, SpeechItem
from .device_manager import AudioDeviceManager
from .voice_config import VoiceConfigManager
from .cooldown_manager import SpeechCooldownManager
from .speech_worker import SpeechWorker

class CrossPlatformTTS:
    """
    Main facade for the VisionAssist cross-platform speech system.
    """
    def __init__(self, voice: str = "Samantha", rate: int = 175, volume: float = 1.0):
        # Configure logging if not already configured
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
            
        self.logger = logging.getLogger("CrossPlatformTTS")
        
        self.queue_manager = SpeechQueueManager()
        self.device_manager = AudioDeviceManager()
        self.voice_config = VoiceConfigManager(target_voice=voice, rate=rate, volume=volume)
        self.cooldown_manager = SpeechCooldownManager()
        
        self.worker = SpeechWorker(
            queue_manager=self.queue_manager,
            device_manager=self.device_manager,
            voice_config=self.voice_config
        )
        self.worker.start()
        
        self.logger.info("[TTS] CrossPlatformTTS initialized.")

    def speak(
        self, 
        text: str, 
        direction: str = "center", 
        interrupt: bool = False, 
        priority: int = 10,
        ignore_cooldown: bool = False
    ) -> None:
        """
        Enqueues text to be spoken.
        - direction: "left", "right", or "center" for spatial panning.
        - interrupt: If True, stops current speech, clears queue, and speaks immediately.
        - priority: Lower number = spoken first.
        - ignore_cooldown: If True, bypasses the deduplication cache.
        """
        if not text or not str(text).strip():
            return
            
        # Cooldown check
        if not ignore_cooldown and not interrupt:
            if self.cooldown_manager.is_on_cooldown(text):
                self.logger.debug(f"[TTS] Skipped (on cooldown): {text}")
                return
                
        # Interruption logic
        if interrupt:
            self.interrupt()
            
        # Record this phrase so it goes on cooldown
        if not ignore_cooldown:
            self.cooldown_manager.record_speech(text)
            
        # Enqueue
        item = SpeechItem(
            priority=priority,
            text=str(text),
            direction=direction,
            interrupt=interrupt,
            timestamp=time.time()
        )
        self.queue_manager.put(item)

    def interrupt(self) -> None:
        """Stops currently playing audio and clears the queue."""
        self.queue_manager.clear()
        self.device_manager.interrupt()

    def stop(self) -> None:
        """Alias for interrupt() for compatibility."""
        self.interrupt()

    def clear_queue(self) -> None:
        """Clears pending items without stopping current speech."""
        self.queue_manager.clear()

    def pause(self) -> None:
        """Pauses processing of the speech queue."""
        self.queue_manager.pause()

    def resume(self) -> None:
        """Resumes processing of the speech queue."""
        self.queue_manager.resume()

    def set_voice(self, voice_name: str) -> None:
        """Updates the target voice. Takes effect on next initialization or worker loop."""
        self.voice_config.target_voice = voice_name
        if self.worker.engine:
            self.voice_config._set_best_voice(self.worker.engine)

    def set_rate(self, rate: int) -> None:
        self.voice_config.rate = rate
        if self.worker.engine:
            self.worker.engine.setProperty('rate', rate)

    def set_volume(self, volume: float) -> None:
        self.voice_config.volume = volume
        if self.worker.engine:
            self.worker.engine.setProperty('volume', volume)

    def cleanup(self) -> None:
        self.interrupt()
        self.device_manager.cleanup()
        
# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    tts = CrossPlatformTTS()
    tts.speak("Testing cross-platform TTS. Person ahead, sign ahead says EXIT", direction="center")
    time.sleep(4)
    tts.speak("Now panning left.", direction="left")
    time.sleep(3)
    tts.speak("Now panning right.", direction="right")
    time.sleep(3)
    print("Interrupt test in 2 seconds...")
    tts.speak("This is a long sentence that should be interrupted before it finishes speaking.", direction="center")
    time.sleep(1.5)
    tts.speak("Interrupted!", direction="center", interrupt=True)
    time.sleep(2)
    tts.cleanup()
    print("Test complete.")
