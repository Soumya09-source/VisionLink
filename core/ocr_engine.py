"""
core/ocr_engine.py — VisionAssist OCR Engine
=============================================

PURPOSE
-------
Extracts readable text from the environment by running OCR *only inside
YOLO bounding boxes* (ROI-based OCR).  This is dramatically faster than
scanning the full frame and reduces false-positive garbage text.

DESIGN DECISIONS
----------------
1. Keyframe OCR — only runs every `ocr_interval` frames, so it never
   competes with the main YOLO detection loop for CPU time.

2. ROI cropping — we only OCR regions where YOLO detected a "text-likely"
   object (sign, book, laptop, screen, etc.).  Random background is skipped.

3. Lazy EasyOCR init — the reader is created on the FIRST call to
   `process_frame()`, not at `__init__` time.  This lets the camera and
   YOLO start immediately while EasyOCR loads in the background.

4. Result caching — OCR results are kept between keyframes so the scene
   doesn't flicker when we skip OCR frames.

5. Anti-spam cooldown — each piece of detected text gets its own cooldown
   timer so it is only spoken once per `text_cooldown_seconds`.

FLOW
----
main.py
  ↓  every frame
ocr_engine.process_frame(frame, detections, frame_index)
  ↓  every `ocr_interval` frames, for text-likely ROIs only
EasyOCR
  ↓
OCRResult list  (text, confidence, position, source_object)
  ↓
cached until next keyframe

scene_builder.py reads ocr_engine.get_active_results()
and merges them into the scene dict.
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# YOLO class names that commonly contain readable text.
# Only bounding boxes whose class_name is in this set get sent to OCR.
# ---------------------------------------------------------------------------
TEXT_LIKELY_CLASSES: frozenset[str] = frozenset({
    # Printed text
    "book", "newspaper",
    # Displays / screens
    "laptop", "tv", "monitor", "cell phone", "keyboard",
    # Signage / labels
    "stop sign", "sign", "billboard",
    # Containers with labels
    "bottle", "cup", "bowl", "wine glass",
    # Other
    "remote", "clock", "microwave", "oven", "refrigerator",
})

# How many pixels to pad around a bounding box before OCR.
# A small pad gives EasyOCR more context at the crop border.
ROI_PADDING: int = 8

# Minimum OCR confidence to accept a result (0.0 – 1.0).
MIN_OCR_CONFIDENCE: float = 0.50

# Minimum text length to bother announcing (filters out single-char and two-char noise).
MIN_TEXT_LENGTH: int = 3

# Maximum characters per OCR result to include in the spoken sentence.
# Very long strings (e.g. entire paragraphs) are truncated.
MAX_TEXT_DISPLAY_LEN: int = 40


@dataclass
class OCRResult:
    """
    One piece of text successfully extracted from the scene.

    Attributes
    ----------
    text            : The cleaned string detected by EasyOCR.
    confidence      : EasyOCR confidence score, 0.0–1.0.
    position        : Spatial zone — "left" | "center" | "right".
    source_object   : YOLO class name of the ROI this text was found in.
    timestamp       : Unix time when this result was produced.
    """
    text: str
    confidence: float
    position: str
    source_object: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "text":          self.text,
            "confidence":    round(self.confidence, 3),
            "position":      self.position,
            "source_object": self.source_object,
            "timestamp":     self.timestamp,
        }


class OCREngine:
    """
    Lightweight, real-time-safe OCR engine for VisionAssist.

    Parameters
    ----------
    ocr_interval : int
        Run OCR every this many frames.  At 15 FPS, interval=30 → OCR runs
        every 2 seconds, which is fast enough for static signage and slow
        enough not to tax the CPU.

    text_cooldown_seconds : float
        A piece of text will not be returned for speech again until this many
        seconds have elapsed since it was last announced.

    gpu : bool
        Pass True if your machine has a CUDA GPU.  On a Mac without CUDA
        this must be False (EasyOCR will use the CPU or Apple MPS).
    """

    def __init__(
        self,
        ocr_interval: int = 30,
        text_cooldown_seconds: float = 8.0,
        gpu: bool = False,
        debug: bool = False,
    ):
        self.ocr_interval = ocr_interval
        self.text_cooldown_seconds = text_cooldown_seconds
        self.gpu = gpu
        self.debug = debug   # verbose trace printing

        # EasyOCR reader — created lazily on the first call
        self._reader = None
        self._reader_ready: bool = False

        # Cache: last set of OCR results (persisted across non-keyframes)
        self._cached_results: list[OCRResult] = []

        # Anti-spam: text → last time it was returned for speaking
        self._last_spoken: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        detections: list[dict],
        frame_index: int,
    ) -> list[OCRResult]:
        """
        Main entry point — call once per camera frame.

        On keyframes (frame_index % ocr_interval == 0) this runs EasyOCR
        on relevant bounding box crops.  On all other frames it returns the
        cached results from the last keyframe.

        Parameters
        ----------
        frame       : Full BGR frame from the camera.
        detections  : YOLO detections list (each dict has class_name,
                      direction, confidence, box).
        frame_index : Monotonic frame counter from main.py.

        Returns
        -------
        List of fresh OCRResult objects ready to be merged into the scene.
        """
        # Lazy-load EasyOCR the first time we're called
        if not self._reader_ready:
            self._init_reader()

        if not self._reader_ready:
            return []   # still loading or failed

        # Only run OCR on keyframes
        if frame_index % self.ocr_interval != 0:
            if self.debug and self._cached_results:
                print(f"[OCR] frame {frame_index}: skipping (non-keyframe), cache={[r.text for r in self._cached_results]}")
            return self._cached_results

        # ----------------------------------------------------------------
        # Keyframe OCR pass
        # ----------------------------------------------------------------
        new_results: list[OCRResult] = []

        if self.debug:
            text_likely = [d.get('class_name','') for d in detections if d.get('class_name','') in TEXT_LIKELY_CLASSES]
            print(f"[OCR] ── KEYFRAME {frame_index} ── text-likely objects detected: {text_likely or 'none'}")

        for det in detections:
            label = det.get("class_name", "")
            if label not in TEXT_LIKELY_CLASSES:
                continue   # skip objects unlikely to have text

            roi = self._extract_roi(frame, det)
            if roi is None:
                if self.debug:
                    print(f"[OCR]   {label}: ROI too small, skipped")
                continue

            if self.debug:
                print(f"[OCR]   {label}: ROI shape={roi.shape}")

            # Preprocess the crop for better OCR accuracy
            processed_roi = self._preprocess(roi)

            # Run EasyOCR on the tiny crop — much faster than full frame
            try:
                raw = self._reader.readtext(processed_roi, detail=1)
                if self.debug:
                    print(f"[OCR]   {label}: EasyOCR raw → {[(t, round(c,2)) for _,t,c in raw]}")
            except Exception as e:
                print(f"[OCR Error] {e}")
                continue

            for (_bbox, text, conf) in raw:
                text_clean = _clean_text(text)
                if self.debug:
                    print(f"[OCR]     raw={text!r:20s} clean={text_clean!r:20s} conf={conf:.2f}", end="")
                if conf < MIN_OCR_CONFIDENCE:
                    if self.debug: print(" ✗ conf too low")
                    continue
                if len(text_clean) < MIN_TEXT_LENGTH:
                    if self.debug: print(" ✗ too short")
                    continue
                if self.debug: print(" ✓ ACCEPTED")

                # Truncate very long strings
                display_text = text_clean[:MAX_TEXT_DISPLAY_LEN]

                new_results.append(OCRResult(
                    text=display_text,
                    confidence=float(conf),
                    position=det.get("direction", "center"),
                    source_object=label,
                ))

        # Deduplicate: if the same text appears in multiple ROIs, keep best
        new_results = _deduplicate_results(new_results)
        if self.debug:
            print(f"[OCR] Final results this keyframe: {[r.text for r in new_results] or 'none'}")

        # Update the cache
        self._cached_results = new_results
        return new_results

    def get_active_results(self) -> list[OCRResult]:
        """Return the current cached OCR results (no new OCR pass)."""
        return self._cached_results

    def get_speakable_results(self) -> list[OCRResult]:
        """
        Return cached OCR results that are NOT in cooldown.

        IMPORTANT: does NOT update cooldown timestamps here.
        Cooldown is only committed by mark_spoken(), which main.py calls
        AFTER confirming the speech was actually emitted by SceneBuilder.
        This prevents burning a text's cooldown slot when SceneBuilder
        suppresses the announcement (e.g. scene hasn't changed enough).
        """
        now = time.time()
        speakable: list[OCRResult] = []

        for result in self._cached_results:
            key = result.text.lower()
            last_time = self._last_spoken.get(key, 0.0)
            if now - last_time >= self.text_cooldown_seconds:
                speakable.append(result)

        return speakable

    def mark_spoken(self, results: list[OCRResult]) -> None:
        """
        Commit cooldown timestamps for results that were actually spoken.
        Call this from main.py only when tts.speak() is invoked.
        """
        now = time.time()
        for r in results:
            self._last_spoken[r.text.lower()] = now
        if self.debug and results:
            print(f"[OCR] Marked as spoken: {[r.text for r in results]}")

    def clear_cache(self) -> None:
        """Force clear the OCR result cache (e.g. after a large scene shift)."""
        self._cached_results = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_reader(self) -> None:
        """
        Lazily initialise EasyOCR.
        Runs once.  If it fails, _reader_ready stays False and OCR is
        silently skipped so the rest of the system keeps running.
        """
        try:
            print("[OCR] Loading EasyOCR model (one-time, ~3–5 s) …")
            import easyocr
            self._reader = easyocr.Reader(
                ["en"],
                gpu=self.gpu,
                verbose=False,
            )
            self._reader_ready = True
            print("[OCR] EasyOCR ready ✓")
        except Exception as e:
            print(f"[OCR Init Error] EasyOCR failed to load: {e}")
            self._reader_ready = False

    def _extract_roi(self, frame: np.ndarray, det: dict) -> Optional[np.ndarray]:
        """
        Crop a padded region from the frame for the given detection.

        Returns None if the resulting crop would be too small to OCR.
        """
        x1, y1, x2, y2 = det.get("box", (0, 0, 0, 0))
        h, w = frame.shape[:2]

        # Apply padding, clamped to frame bounds
        x1 = max(0, x1 - ROI_PADDING)
        y1 = max(0, y1 - ROI_PADDING)
        x2 = min(w, x2 + ROI_PADDING)
        y2 = min(h, y2 + ROI_PADDING)

        roi = frame[y1:y2, x1:x2]

        # Skip crops that are too tiny (< 20×10 px) — EasyOCR would fail
        if roi.shape[0] < 10 or roi.shape[1] < 20:
            return None

        return roi

    @staticmethod
    def _preprocess(roi: np.ndarray) -> np.ndarray:
        """
        Lightweight image preprocessing to improve OCR accuracy.

        Steps:
          1. Upscale small crops (EasyOCR performs better on larger images)
          2. Convert to grayscale
          3. Apply mild sharpening

        We deliberately avoid heavy transforms (thresholding, morphology)
        because EasyOCR's built-in preprocessing is already robust.
        """
        h, w = roi.shape[:2]

        # Upscale if the crop is small — 2× helps EasyOCR read small fonts
        if h < 80 or w < 80:
            roi = cv2.resize(roi, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        # Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Mild unsharp mask for edge sharpening
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        return sharpened


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Remove noise characters from OCR output.
    Keeps: letters, digits, spaces, basic punctuation.
    Strips: leading/trailing whitespace.
    """
    # Keep alphanumerics, spaces, and common punctuation
    cleaned = "".join(
        ch for ch in text
        if ch.isalnum() or ch in " .,!?:'-/"
    )
    return cleaned.strip()


def _deduplicate_results(results: list[OCRResult]) -> list[OCRResult]:
    """
    If the same text was found in multiple ROIs, keep only the highest-
    confidence version. Case-insensitive comparison.
    """
    best: dict[str, OCRResult] = {}
    for r in results:
        key = r.text.lower()
        if key not in best or r.confidence > best[key].confidence:
            best[key] = r
    return list(best.values())


def generate_ocr_phrases(ocr_results) -> list[str]:
    """
    Convert OCRResult objects (or duck-typed dicts) into natural spoken phrases.

    Examples
    --------
    text="EXIT", source_object="stop sign", position="center"
        → "sign ahead says EXIT"

    text="OPEN", source_object="bottle", position="left"
        → "label on the left says OPEN"

    text="HELLO", source_object="laptop", position="center"
        → "screen ahead shows HELLO"
    """
    from core.scene_builder import DIRECTION_PHRASES   # late import avoids circular ref

    # Map YOLO class → what kind of text container it is
    source_labels = {
        "book":       ("book",    "says"),
        "laptop":     ("screen",  "shows"),
        "tv":         ("screen",  "shows"),
        "monitor":    ("screen",  "shows"),
        "cell phone": ("screen",  "shows"),
        "stop sign":  ("sign",    "says"),
        "bottle":     ("label",   "says"),
        "cup":        ("label",   "says"),
        "keyboard":   ("keyboard","shows"),
        "clock":      ("clock",   "shows"),
    }

    phrases: list[str] = []
    for r in ocr_results:
        # Support both OCRResult objects and plain dicts
        text     = r.text          if hasattr(r, 'text')          else r.get('text', '')
        position = r.position      if hasattr(r, 'position')      else r.get('position', 'center')
        src_obj  = r.source_object if hasattr(r, 'source_object') else r.get('source_object', '')

        dir_phrase = DIRECTION_PHRASES.get(position, f"on the {position}")
        container, verb = source_labels.get(src_obj, ("sign", "says"))

        # Natural form: "sign ahead says EXIT" — container + position + verb + text
        phrases.append(f"{container} {dir_phrase} {verb} {text}")

    return phrases
