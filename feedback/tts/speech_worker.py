import threading
import queue
import pyttsx3
import logging
import tempfile
import os
import time

from .queue_manager import SpeechQueueManager
from .device_manager import AudioDeviceManager
from .voice_config import VoiceConfigManager

class SpeechWorker(threading.Thread):
    """
    Daemon thread that safely initializes pyttsx3, blocks on the SpeechQueue,
    generates WAV files, and plays them via AudioDeviceManager.
    Recovers automatically from engine crashes.
    """
    def __init__(
        self, 
        queue_manager: SpeechQueueManager, 
        device_manager: AudioDeviceManager,
        voice_config: VoiceConfigManager
    ):
        super().__init__(daemon=True)
        self.logger = logging.getLogger("SpeechWorker")
        self.queue_manager = queue_manager
        self.device_manager = device_manager
        self.voice_config = voice_config
        
        self.engine = None
        self._temp_dir = tempfile.mkdtemp()
        self._wav_path = os.path.join(self._temp_dir, "speech_temp.wav")

    def _init_engine(self):
        """Initializes pyttsx3 safely within the thread context."""
        try:
            if self.engine:
                del self.engine
            self.engine = pyttsx3.init()
            self.voice_config.configure_engine(self.engine)
            self.logger.info("[SpeechWorker] pyttsx3 engine initialized.")
        except Exception as e:
            self.logger.error(f"[SpeechWorker] Failed to init pyttsx3: {e}")
            self.engine = None

    def run(self):
        """Main thread loop."""
        # Initialize engine inside the thread!
        self._init_engine()
        
        while True:
            try:
                # 1. Block until an item is available
                # Use a timeout so we can periodically check for thread termination if we ever add it
                item = self.queue_manager.get(timeout=1.0)
            except queue.Empty:
                continue
                
            try:
                if self.queue_manager.is_paused:
                    self.queue_manager.task_done()
                    continue
                    
                # 2. If engine crashed, try to recover
                if not self.engine:
                    self.logger.warning("[SpeechWorker] Engine missing. Attempting recovery...")
                    self._init_engine()
                    if not self.engine:
                        self.logger.error("[SpeechWorker] Recovery failed. Dropping speech item.")
                        self.queue_manager.task_done()
                        continue

                # 3. Generate WAV using pyttsx3
                try:
                    self.engine.save_to_file(item.text, self._wav_path)
                    self.engine.runAndWait()
                except Exception as e:
                    self.logger.error(f"[SpeechWorker] pyttsx3 generation failed: {e}")
                    # Reinitialize engine for next time
                    self._init_engine()
                    self.queue_manager.task_done()
                    continue
                    
                # 4. Play the WAV file with panning (blocking call to device_manager)
                # It will stop early if device_manager.interrupt() is called by main thread
                if os.path.exists(self._wav_path):
                    self.device_manager.play_wav(self._wav_path, item.direction)
                
            except Exception as e:
                self.logger.error(f"[SpeechWorker] Unhandled error processing speech: {e}")
            finally:
                self.queue_manager.task_done()
