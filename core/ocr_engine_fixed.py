"""
core/ocr_engine_fixed.py — Enhanced VisionAssist OCR Engine
========================================================

PURPOSE
-------
Extracts readable text from the environment by running OCR *only inside
YOLO bounding boxes* (ROI-based OCR). Enhanced with better preprocessing
and lower confidence thresholds for debugging.

ENHANCEMENTS
-----------
- Lower confidence threshold (0.30) for debugging
- Enhanced preprocessing with CLAHE and aggressive upscaling
- More comprehensive text-likely object classes
- Better ROI extraction with proper padding
- Detailed debug logging
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Enhanced YOLO class names that commonly contain readable text.
# ---------------------------------------------------------------------------
TEXT_LIKELY_CLASSES: frozenset[str] = frozenset({
    # Printed text
    "book", "newspaper", "magazine", "letter", "paper",
    # Displays / screens
    "laptop", "tv", "monitor", "cell phone", "keyboard", "tablet", "screen", "display",
    # Signage / labels
    "stop sign", "sign", "billboard", "traffic light", "street sign", "poster", "banner",
    # Containers with labels
    "bottle", "cup", "bowl", "wine glass", "can", "box", "package", "label",
    # Other text sources
    "remote", "clock", "microwave", "oven", "refrigerator", "calendar",
})

# How many pixels to pad around a bounding box before OCR.
ROI_PADDING: int = 12

# Minimum OCR confidence to accept a result (0.0 – 1.0).
MIN_OCR_CONFIDENCE: float = 0.30

# Minimum text length to bother announcing (filters out single-char and two-char noise).
MIN_TEXT_LENGTH: int = 3

# Maximum characters per OCR result to include in the spoken sentence.
MAX_TEXT_DISPLAY_LEN: int = 40

@dataclass
class OCRResult:
    """
    One piece of text successfully extracted from the scene.
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
    Enhanced OCR engine with better preprocessing and debugging.
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
        self.debug = debug

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
        """
        # Lazy-load EasyOCR first time we're called
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

            # Preprocess crop for better OCR accuracy
            processed_roi = self._preprocess(roi)

            # Run EasyOCR on processed crop
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
        """Force clear OCR result cache (e.g. after a large scene shift)."""
        self._cached_results = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_reader(self) -> None:
        """
        Lazily initialise EasyOCR.
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
        Crop a padded region from frame for the given detection.
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
        Enhanced image preprocessing to improve OCR accuracy.
        """
        h, w = roi.shape[:2]

        # More aggressive upscaling for very small crops
        if h < 60 or w < 60:
            # 3× upscaling for tiny text
            roi = cv2.resize(roi, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        elif h < 100 or w < 100:
            # 2× upscaling for small text
            roi = cv2.resize(roi, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        # Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Enhanced sharpening with stronger kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)

        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)

        return enhanced

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
