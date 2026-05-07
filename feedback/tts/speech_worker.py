import threading
import queue
import pyttsx3
import logging
import tempfile
import os
import time
import re

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
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for natural speech with proper pacing and punctuation."""
        if not text:
            return text
            
        # Convert to lowercase for consistent processing
        processed = text.lower()
        
        # Add proper punctuation for natural pauses
        # Replace common patterns with punctuation
        replacements = {
            r'\b(left|right|center|ahead|behind)\b': r'\1,',
            r'\b(person|bottle|chair|table|phone|book|bag|laptop|cup|clock|remote|cat|dog)\b': r'\1,',
            r'\b(detected|found|seen|located)\b': r'\1,',
            r'\b(says|reads|shows)\b': r'\1,',
            r'\b(exit|stop|warning|alert|danger)\b': r'\1!',
            r'\b(and|or|but)\b': r'\1 ',
        }
        
        for pattern, replacement in replacements.items():
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        
        # Ensure sentences end with proper punctuation
        if not processed.endswith(('.', ',', '!', '?')):
            processed += '.'
            
        # Clean up multiple spaces and punctuation
        processed = re.sub(r'\s+', ' ', processed)
        processed = re.sub(r'[,.]{2,}', ',', processed)
        processed = re.sub(r'[!?]{2,}', '!', processed)
        
        # Capitalize first letter for better TTS
        if processed:
            processed = processed[0].upper() + processed[1:]
            
        self.logger.debug(f"[SpeechWorker] Preprocessed: '{text}' → '{processed}'")
        return processed

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

                # 3. Generate speech using pyttsx3 with robust error handling
                try:
                    # Preprocess text for natural speech
                    processed_text = self._preprocess_text(item.text)
                    
                    # Try direct speech synthesis first (most reliable)
                    try:
                        # Stop any ongoing speech
                        self.engine.stop()
                        
                        # Connect engine to our device manager for direct playback
                        self.engine.connect('started-word', lambda name, loc, length: None)
                        self.engine.connect('finished-utterance', lambda name, completed: None)
                        
                        # Speak directly (most reliable method)
                        self.engine.say(processed_text)
                        self.engine.runAndWait()
                        
                        self.logger.info(f"[SpeechWorker] Direct synthesis successful: '{processed_text[:50]}...'")
                        # task_done() will be called in finally block
                        continue
                        
                    except Exception as direct_error:
                        self.logger.warning(f"[SpeechWorker] Direct synthesis failed, trying file method: {direct_error}")
                    
                    # Fallback to file generation with multiple attempts
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            # Use unique filename to avoid corruption issues
                            unique_wav_path = os.path.join(self._temp_dir, f"speech_{int(time.time() * 1000)}_{attempt}.wav")
                            
                            # Clean up any existing file
                            if os.path.exists(unique_wav_path):
                                os.remove(unique_wav_path)
                            
                            # Reset engine state
                            self.engine.stop()
                            
                            # Generate speech file
                            self.engine.save_to_file(processed_text, unique_wav_path)
                            self.engine.runAndWait()
                            
                            # Verify file was created and is valid
                            if not os.path.exists(unique_wav_path):
                                raise Exception("WAV file was not created")
                            
                            # Check file size (should be > 0 for valid speech)
                            file_size = os.path.getsize(unique_wav_path)
                            if file_size == 0:
                                raise Exception("WAV file is empty")
                            
                            # Check for valid WAV header
                            with open(unique_wav_path, 'rb') as f:
                                header = f.read(4)
                                if header != b'RIFF':
                                    raise Exception("Invalid WAV file header")
                            
                            self.logger.info(f"[SpeechWorker] File generation successful (attempt {attempt+1}): {file_size} bytes")
                            
                            # Update wav path for playback
                            self._wav_path = unique_wav_path
                            break  # Success, exit retry loop
                            
                        except Exception as attempt_error:
                            self.logger.warning(f"[SpeechWorker] File generation attempt {attempt+1} failed: {attempt_error}")
                            if attempt == max_attempts - 1:
                                raise attempt_error  # Re-raise last attempt error
                            time.sleep(0.1)  # Brief delay between attempts
                    
                except Exception as e:
                    self.logger.error(f"[SpeechWorker] All speech generation methods failed: {e}")
                    # Reinitialize engine for next time
                    self._init_engine()
                    self.queue_manager.task_done()
                    continue
                    
                # 4. Play the WAV file with panning (blocking call to device_manager)
                # It will stop early if device_manager.interrupt() is called by main thread
                if os.path.exists(self._wav_path):
                    self.device_manager.play_wav(self._wav_path, item.direction)
                
                # Clean up the WAV file after playback
                if os.path.exists(self._wav_path):
                    try:
                        os.remove(self._wav_path)
                        self.logger.debug(f"[SpeechWorker] Cleaned up WAV file: {self._wav_path}")
                    except Exception as e:
                        self.logger.warning(f"[SpeechWorker] Failed to clean up WAV file: {e}")
                
            except Exception as e:
                self.logger.error(f"[SpeechWorker] Unhandled error processing speech: {e}")
            finally:
                self.queue_manager.task_done()
