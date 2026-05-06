"""
feedback/spatial_audio.py — VisionAssist Spatial Audio Layer
=============================================================

PURPOSE
-------
Adds stereo panning to speech feedback so that objects detected on
the left sound louder in the left ear, and objects on the right
sound louder in the right ear. Uses macOS native tools (say, afplay)
and numpy for zero external dependencies.
"""

import threading
import subprocess
import tempfile
import time
import wave
import os
import numpy as np

class SpatialAudio:
    def __init__(self, voice: str = "Samantha", rate: int = 175):
        self.voice = voice
        self.rate = str(rate)
        
        self._lock = threading.Lock()
        self._is_playing = False
        self._play_id = 0
        self._afplay_proc = None
        
        # We need a stable temp directory to write WAV files
        self._temp_dir = tempfile.mkdtemp()
        self._raw_wav = os.path.join(self._temp_dir, "raw.wav")
        self._panned_wav = os.path.join(self._temp_dir, "panned.wav")
        
        print(f"[SPATIAL AUDIO] Initialized macOS stereo engine (voice: {self.voice}, rate: {self.rate})")

    def play(self, text: str, direction: str, interrupt: bool = False) -> None:
        """
        Non-blocking spatial playback.
        direction should be "left", "center", or "right".
        """
        if not text or not str(text).strip():
            return
            
        if interrupt:
            self.interrupt()
            
        with self._lock:
            if self._is_playing and not interrupt:
                return  # Skip overlapping speech
            
            self._play_id += 1
            current_id = self._play_id
            self._is_playing = True
            
        # Fire and forget in a daemon thread so OpenCV loop doesn't block
        thread = threading.Thread(
            target=self._generate_and_play, 
            args=(str(text), direction, current_id), 
            daemon=True
        )
        thread.start()

    def _generate_and_play(self, text: str, direction: str, play_id: int) -> None:
        try:
            # 1. Generate speech to raw WAV
            # Use LEI16@22050 format for easy numpy processing
            subprocess.run(
                [
                    "say", 
                    "-v", self.voice, 
                    "-r", self.rate, 
                    "-o", self._raw_wav, 
                    "--data-format=LEI16@22050", 
                    text
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 2. Apply stereo panning
            self._apply_panning(direction)
            
            # 3. Play the panned audio using afplay
            with self._lock:
                if self._play_id != play_id:
                    return # A new play was requested, abort this one
                self._afplay_proc = subprocess.Popen(
                    ["afplay", self._panned_wav],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            # Wait for playback to finish
            self._afplay_proc.wait()
            
        except Exception as e:
            print(f"[SPATIAL AUDIO ERROR] {e}")
        finally:
            with self._lock:
                if self._play_id == play_id:
                    self._is_playing = False
                    self._afplay_proc = None

    def _apply_panning(self, direction: str) -> None:
        """Reads raw_wav, applies L/R volume multipliers, and saves to panned_wav."""
        # Define volume multipliers (Left, Right)
        if direction == "left":
            vol_l, vol_r = 1.0, 0.0
            print(f"[SPATIAL AUDIO] Applying LEFT pan")
        elif direction == "right":
            vol_l, vol_r = 0.0, 1.0
            print(f"[SPATIAL AUDIO] Applying RIGHT pan")
        else:
            # Center
            vol_l, vol_r = 0.8, 0.8
            print(f"[SPATIAL AUDIO] Applying CENTER pan")
            
        with wave.open(self._raw_wav, 'rb') as wf_in:
            params = wf_in.getparams()
            n_channels = params.nchannels
            sampwidth = params.sampwidth
            framerate = params.framerate
            n_frames = params.nframes
            
            raw_data = wf_in.readframes(n_frames)
            
        # Convert to numpy array of int16
        audio = np.frombuffer(raw_data, dtype=np.int16)
        
        if n_channels == 2:
            audio = audio.reshape(-1, 2)
            
        print(f"[SPATIAL AUDIO] Before conversion:")
        print(f"[SPATIAL AUDIO] audio.shape: {audio.shape}")
        print(f"[SPATIAL AUDIO] audio.dtype: {audio.dtype}")
        print(f"[SPATIAL AUDIO] channel count: {n_channels}")
        
        if audio.ndim == 1:
            print(f"[SPATIAL AUDIO] Mono detected → converting to stereo")
            audio = np.stack([audio, audio], axis=1)
            
        print(f"[SPATIAL AUDIO] Shape after conversion: {audio.shape}")
        
        # Apply floating point multipliers
        audio_float = audio.astype(np.float32)
        
        audio_float[:, 0] *= vol_l
        audio_float[:, 1] *= vol_r
        
        # Clip back to int16 bounds and cast
        np.clip(audio_float, -32768, 32767, out=audio_float)
        panned_array = audio_float.astype(np.int16)
        
        # Save to panned_wav
        with wave.open(self._panned_wav, 'wb') as wf_out:
            wf_out.setnchannels(2)  # Force 2 channels
            wf_out.setsampwidth(sampwidth)
            wf_out.setframerate(framerate)
            wf_out.writeframes(panned_array.tobytes())

    def interrupt(self) -> None:
        """Kills any active afplay process and 'say' process to interrupt."""
        try:
            # Kill any system 'say' processes just in case it's mid-generation
            subprocess.run(["killall", "say"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Kill afplay globally as backup for immediate interruption
            subprocess.run(["killall", "afplay"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            with self._lock:
                # Increment play_id to immediately invalidate any ongoing threads
                self._play_id += 1
                if self._afplay_proc:
                    self._afplay_proc.terminate()
                    
        except Exception as e:
            print(f"[SPATIAL AUDIO Interrupt Error] {e}")
        finally:
            with self._lock:
                self._is_playing = False
                self._afplay_proc = None

    def stop(self) -> None:
        self.interrupt()
