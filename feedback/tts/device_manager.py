import wave
import numpy as np
import threading
import logging

try:
    import pyaudio
except ImportError:
    pyaudio = None
    logging.warning("[DeviceManager] pyaudio not installed. Speech playback will be disabled.")

class AudioDeviceManager:
    """
    Handles cross-platform playback of WAV files using PyAudio.
    Supports stereo panning and instant interruption.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AudioDeviceManager")
        self._pyaudio_instance = None
        self._stop_event = threading.Event()
        
        if pyaudio:
            try:
                self._pyaudio_instance = pyaudio.PyAudio()
                self.logger.info("[DeviceManager] PyAudio initialized successfully.")
            except Exception as e:
                self.logger.error(f"[DeviceManager] PyAudio init failed: {e}")
                self._pyaudio_instance = None

    def play_wav(self, file_path: str, direction: str = "center") -> None:
        """
        Reads a WAV file, applies spatial panning, and plays it chunk-by-chunk.
        Blocks until finished or interrupted.
        """
        if not self._pyaudio_instance:
            self.logger.error("[DeviceManager] PyAudio not available. Cannot play audio.")
            return

        self._stop_event.clear()

        try:
            with wave.open(file_path, 'rb') as wf:
                params = wf.getparams()
                n_channels = params.nchannels
                sampwidth = params.sampwidth
                framerate = params.framerate
                n_frames = params.nframes

                if n_frames <= 0:
                    return

                raw_data = wf.readframes(n_frames)

            # Apply panning
            panned_data = self._apply_panning(raw_data, direction, n_channels)
            
            # Open stream
            stream = self._pyaudio_instance.open(
                format=self._pyaudio_instance.get_format_from_width(sampwidth),
                channels=2, # We force 2 channels in panning
                rate=framerate,
                output=True
            )

            # Playback chunk-by-chunk for interruptibility
            chunk_size = 1024
            
            # Calculate bytes per frame: 2 channels * sampwidth
            bytes_per_frame = 2 * sampwidth
            chunk_bytes = chunk_size * bytes_per_frame

            for i in range(0, len(panned_data), chunk_bytes):
                if self._stop_event.is_set():
                    self.logger.debug("[DeviceManager] Playback interrupted.")
                    break
                    
                chunk = panned_data[i:i + chunk_bytes]
                stream.write(chunk)

            stream.stop_stream()
            stream.close()

        except FileNotFoundError:
            self.logger.error(f"[DeviceManager] WAV file not found: {file_path}")
        except Exception as e:
            self.logger.error(f"[DeviceManager] Playback error: {e}")

    def _apply_panning(self, raw_data: bytes, direction: str, original_channels: int) -> bytes:
        """Applies L/R volume multipliers based on direction. Forces output to stereo."""
        if direction == "left":
            vol_l, vol_r = 1.0, 0.0
        elif direction == "right":
            vol_l, vol_r = 0.0, 1.0
        else:
            vol_l, vol_r = 0.8, 0.8  # center

        audio = np.frombuffer(raw_data, dtype=np.int16)

        if original_channels == 2:
            audio = audio.reshape(-1, 2)
        elif audio.ndim == 1:
            # Mono to stereo
            audio = np.stack([audio, audio], axis=1)

        audio_float = audio.astype(np.float32)
        audio_float[:, 0] *= vol_l
        audio_float[:, 1] *= vol_r

        np.clip(audio_float, -32768, 32767, out=audio_float)
        return audio_float.astype(np.int16).tobytes()

    def interrupt(self) -> None:
        """Signals the playback loop to stop immediately."""
        self._stop_event.set()

    def cleanup(self) -> None:
        """Terminates PyAudio instance."""
        self.interrupt()
        if self._pyaudio_instance:
            self._pyaudio_instance.terminate()
            self._pyaudio_instance = None
