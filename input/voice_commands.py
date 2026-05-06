"""
input/voice_commands.py — VisionAssist Voice Command Layer
===========================================================

PURPOSE
-------
Listens for voice commands using an offline Vosk speech recognizer.
Triggered by an external call (e.g. keypress), it opens a listening
window, transcribes audio, and executes a callback if a recognized
command is found.

KEY FIXES (v2)
--------------
1. Grammar-constrained recognizer: vastly improves accuracy for our
   small fixed command set vs. open-vocabulary mode.
2. Explicit device selection: always opens the MacBook Air Microphone
   (or whichever device index is configured) rather than relying on
   PyAudio's default, which may pick Bluetooth at the wrong rate.
3. Native 16000 Hz stream: avoids the silent sample-rate mismatch
   that caused Vosk to never fire AcceptWaveform.
"""

import threading
import json
import time

# Deferred imports so the script can load quickly
vosk = None
pyaudio = None

# ── Configuration ─────────────────────────────────────────────────────────────
# Set to None to use PyAudio's default device, or an integer index
# (from mic_test.py output) to force a specific microphone.
# Example: PREFERRED_INPUT_DEVICE = 3  ← MacBook Air Microphone
PREFERRED_INPUT_DEVICE = None   # None = auto-detect best device
SAMPLE_RATE = 16000             # Must match KaldiRecognizer rate

# All commands the recognizer is allowed to return.
# Providing a grammar gives Vosk a huge accuracy boost on small command sets.
_GRAMMAR_COMMANDS = [
    "what do you see",
    "read text",
    "stop speaking",
    "pause alerts",
    "resume alerts",
    "repeat",
    "remember this",
    "what did you remember",
    # "find <label>" and "forget <label>" with common COCO labels
    "find bottle",   "find person",  "find chair",   "find laptop",
    "find cup",      "find book",    "find phone",   "find bag",
    "find clock",    "find remote",  "find cat",     "find dog",
    "forget bottle", "forget person","forget chair", "forget laptop",
    "forget cup",    "forget book",  "forget phone", "forget bag",
    "[unk]",  # required catch-all token for grammar mode
]


class VoiceCommandManager:
    """
    Manages offline speech-to-text and command matching.
    """

    # Fixed commands matched by substring
    ALLOWED_COMMANDS = {
        "what do you see",
        "read text",
        "stop speaking",
        "pause alerts",
        "resume alerts",
        "repeat",
        "remember this",
        "what did you remember",
    }

    # Variable-prefix commands: matched by prefix, remainder = target label
    PREFIX_COMMANDS = (
        "find ",
        "forget ",
    )

    def __init__(
        self,
        model_path: str = "models/vosk-model-small-en-us-0.15",
        command_callback=None,
        input_device: int | None = PREFERRED_INPUT_DEVICE,
    ):
        self.model_path = model_path
        self.command_callback = command_callback
        self.input_device = input_device

        self.model = None
        self.recognizer = None
        self._pa = None

        self._is_listening = False
        self._lock = threading.Lock()

    def _initialize(self):
        """Lazy load Vosk and PyAudio only when first needed."""
        if self.model is not None:
            return

        print("[VOICE] Initializing offline speech recognition (this may take a few seconds)...")
        import vosk as v
        import pyaudio as pa

        global vosk, pyaudio
        vosk = v
        pyaudio = pa

        # Suppress verbose vosk logs
        vosk.SetLogLevel(-1)

        try:
            self.model = vosk.Model(self.model_path)

            # Grammar-constrained recognizer: far more accurate than open vocab
            grammar_json = json.dumps(_GRAMMAR_COMMANDS)
            self.recognizer = vosk.KaldiRecognizer(self.model, SAMPLE_RATE, grammar_json)

            self._pa = pyaudio.PyAudio()

            # Auto-detect: prefer built-in mic if device not forced
            if self.input_device is None:
                self.input_device = self._find_best_input_device()

            dev_name = self._pa.get_device_info_by_index(self.input_device)["name"]
            print(f"[VOICE] Initialization complete.")
            print(f"[VOICE] Using device [{self.input_device}]: {dev_name}")
            print(f"[VOICE] Sample rate: {SAMPLE_RATE} Hz | Grammar mode: ON")
        except Exception as e:
            print(f"[VOICE ERROR] Failed to initialize: {e}")

    def _find_best_input_device(self) -> int:
        """
        Auto-select the best input device.
        Prefers built-in / internal microphones over Bluetooth devices
        because BT mics often report mismatched sample rates that silently
        break Vosk recognition.
        """
        best_idx = self._pa.get_default_input_device_info()["index"]
        best_name = ""

        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if info["maxInputChannels"] < 1:
                continue
            name = info["name"].lower()
            # Prefer macbook / built-in / internal over bluetooth or iPhone
            # Order matters: macbook > internal/built-in > iphone > other
            if any(kw in name for kw in ("macbook", "built-in", "internal")):
                best_idx = i
                best_name = info["name"]
                break   # take first preferred match

        if best_name:
            print(f"[VOICE] Auto-selected input device [{best_idx}]: {best_name}")
        return best_idx

    def listen_once(self, timeout: float = 5.0):
        """
        Opens a listening window for `timeout` seconds.
        Runs asynchronously so it doesn't block the caller.
        """
        with self._lock:
            if self._is_listening:
                print("[VOICE] Already listening...")
                return
            self._is_listening = True

        thread = threading.Thread(target=self._listen_worker, args=(timeout,), daemon=True)
        thread.start()

    def _listen_worker(self, timeout: float):
        try:
            self._initialize()
            if self.model is None or self._pa is None:
                return

            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=4000,
            )

            print(f"[VOICE] Listening for {timeout}s... Speak now!")
            stream.start_stream()

            start_time = time.time()
            final_text = ""

            while time.time() - start_time < timeout:
                data = stream.read(4000, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    res = json.loads(self.recognizer.Result())
                    text = res.get("text", "").strip()
                    if text and text != "[unk]":
                        final_text = text
                        break   # complete sentence found, stop early
                else:
                    partial = json.loads(self.recognizer.PartialResult()).get("partial", "")
                    if partial and partial != "[unk]":
                        print(f"[VOICE] (...{partial})", end="\r")

            # Flush final result if we ran out of time
            if not final_text:
                res = json.loads(self.recognizer.FinalResult())
                final_text = res.get("text", "").strip()

            stream.stop_stream()
            stream.close()

            if final_text and final_text != "[unk]":
                self._match_and_route(final_text)
            else:
                print("[VOICE] No speech detected.")

        except Exception as e:
            print(f"[VOICE Listening Error] {e}")
        finally:
            with self._lock:
                self._is_listening = False

    def _match_and_route(self, text: str):
        """Match the transcribed text against allowed commands."""
        text = text.lower().strip()
        print(f"[VOICE] Recognized: {text}")

        # 1. Check variable-prefix commands first (e.g. "find bottle", "forget chair")
        matched_cmd = None
        for prefix in self.PREFIX_COMMANDS:
            if text.startswith(prefix):
                # Pass the full text so the router can extract the label
                matched_cmd = text   # e.g. "find bottle"
                break

        # 2. Fall back to fixed-command substring matching
        if matched_cmd is None:
            for cmd in self.ALLOWED_COMMANDS:
                if cmd in text:
                    matched_cmd = cmd
                    break

        if matched_cmd:
            print(f"[VOICE] Executing command: {matched_cmd}")
            if self.command_callback:
                # Execute callback on a new thread so we don't block the worker's cleanup
                threading.Thread(target=self.command_callback, args=(matched_cmd,), daemon=True).start()
        else:
            print("[VOICE] Command not recognized (ignoring).")
