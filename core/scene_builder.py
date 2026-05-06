"""
core/scene_builder.py — VisionAssist Scene Builder Layer
=========================================================

PURPOSE
-------
Sits between YOLO detections and the TTS engine.
Turns noisy, frame-by-frame detections into a stable, structured scene
and generates natural speech summaries only when something meaningful changes.

ARCHITECTURE
------------
Camera
  ↓
YOLO Detection  (list of dicts: class_name, direction, confidence, box)
  ↓
SceneBuilder.process_frame()        <-- YOU ARE HERE
  ↓ (stable scene dict)
SceneBuilder.get_speech_update()
  ↓ (str or None)
TTS

DESIGN PRINCIPLES
-----------------
- No AI reasoning — pure rule-based logic
- Real-time safe: all operations are O(n) with n = number of detections
- Non-blocking: no sleep(), no threads, no I/O
- Beginner-friendly: heavy inline comments explaining every decision
"""

import time
from collections import defaultdict


# ---------------------------------------------------------------------------
# Priority table — lower number = spoken first (person is most important)
# ---------------------------------------------------------------------------
OBJECT_PRIORITIES: dict[str, int] = {
    "person":   1,
    "car":      2, "bus":    2, "truck":   2,
    "bicycle":  3, "motorcycle": 3,
    "chair":    4, "couch":  4, "bed":     4, "table":   4,
    "dog":      5, "cat":    5,
    "bottle":   6, "cup":    6, "bowl":    6,
    "laptop":   7, "cell phone": 7, "remote": 7,
    "book":     8, "clock":  8,
}

# Direction → human-readable phrase
DIRECTION_PHRASES: dict[str, str] = {
    "left":   "on the left",
    "center": "ahead",
    "right":  "on the right",
}

# How many consecutive missed frames before an object is removed from the scene.
# At 10–15 FPS, 8 frames ≈ 0.5–0.8 seconds of persistence.
DEFAULT_PERSISTENCE_FRAMES: int = 8

# Minimum number of top-priority objects to include in each sentence.
MAX_OBJECTS_IN_SENTENCE: int = 3

# When a scene changes slightly, we need ≥ this fraction of objects to
# differ before we consider it a "meaningful" change worth announcing.
# 0.0 = announce every single change (noisy)
# 1.0 = only announce completely new scenes (too quiet)
SCENE_CHANGE_THRESHOLD: float = 0.0   # start simple: announce any change


class _TrackedObject:
    """
    Internal record for one tracked (class, direction) pair.

    Why track per (class, direction) and not just per class?
    Because "bottle on the left" and "bottle on the right" are two distinct
    scene facts that should be tracked independently.
    """

    __slots__ = ("label", "direction", "last_seen_frame", "confidence")

    def __init__(self, label: str, direction: str, confidence: float, frame: int):
        self.label = label
        self.direction = direction
        self.confidence = confidence
        self.last_seen_frame = frame  # the frame index when last detected


class SceneBuilder:
    """
    Converts raw per-frame YOLO detections into a stable, structured scene.

    Usage example
    -------------
    builder = SceneBuilder()

    # Inside your main loop:
    scene = builder.process_frame(detections)          # always call this
    speech = builder.get_speech_update()               # returns str or None
    if speech:
        tts.speak(speech)

    Parameters
    ----------
    persistence_frames : int
        How many consecutive frames an object can be "missing" from YOLO
        before it is removed from the scene.  Tune up for smoother output,
        tune down to react faster to objects leaving the scene.

    cooldown_seconds : float
        Minimum wall-clock seconds between two spoken updates.
        Prevents rapid-fire speech when the scene oscillates.
    """

    def __init__(
        self,
        persistence_frames: int = DEFAULT_PERSISTENCE_FRAMES,
        cooldown_seconds: float = 5.0,
    ):
        self.persistence_frames = persistence_frames
        self.cooldown_seconds = cooldown_seconds

        # The live scene: key = (label, direction), value = _TrackedObject
        self._tracked: dict[tuple[str, str], _TrackedObject] = {}

        # The "last announced" scene snapshot so we can detect changes.
        # Stored as a tuple so OCR texts are included in the comparison.
        # _last_announced_scene  : dict[label, frozenset[directions]]
        # _last_announced_ocr    : frozenset[str]  — lower-cased OCR texts
        self._last_announced_scene: dict[str, frozenset[str]] = {}
        self._last_announced_ocr: frozenset[str] = frozenset()

        # Wall-clock time of the last speech output (None = never spoken)
        self._last_spoken_time: float | None = None

        # Monotonically increasing frame counter
        self._frame_index: int = 0

        # The most recent structured scene snapshot (updated every frame)
        self._current_scene: dict = {}

        # Latest OCR results merged into the scene (set externally)
        self._active_ocr: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, detections: list[dict], ocr_results: list = None) -> dict:
        """
        Ingest one frame's worth of YOLO detections and update the scene.

        Parameters
        ----------
        detections : list[dict]
            Each dict must contain at least:
                "class_name"  : str   — e.g. "person"
                "direction"   : str   — "left" | "center" | "right"
                "confidence"  : float — 0.0–1.0

        Parameters
        ----------
        ocr_results : list[OCRResult], optional
            Fresh OCR results from OCREngine.get_speakable_results().
            Pass an empty list or None if OCR is not enabled.

        Returns
        -------
        dict
            A structured scene snapshot:
            {
                "objects": {
                    "person":  ["center"],
                    "bottle":  ["left", "right"]
                },
                "ocr": [
                    {"text": "EXIT", "confidence": 0.92,
                     "position": "center", "source_object": "sign"}
                ],
                "timestamp": 1715000000.0
            }
        """
        self._frame_index += 1

        # --- STEP 1: Mark which (label, direction) pairs were seen this frame ---
        seen_keys: set[tuple[str, str]] = set()

        for det in detections:
            label = det.get("class_name") or det.get("label", "unknown")
            direction = det.get("direction") or det.get("position", "center")
            confidence = float(det.get("confidence", 1.0))

            key = (label, direction)
            seen_keys.add(key)

            if key in self._tracked:
                # Already tracked — just refresh the frame counter
                self._tracked[key].last_seen_frame = self._frame_index
                self._tracked[key].confidence = confidence
            else:
                # New detection — add it to the scene
                self._tracked[key] = _TrackedObject(
                    label=label,
                    direction=direction,
                    confidence=confidence,
                    frame=self._frame_index,
                )

        # --- STEP 2: Expire objects that have been missing too long ---
        # We use a threshold: if the object hasn't been seen for
        # `persistence_frames` frames, drop it from the tracked set.
        expired_keys = [
            key
            for key, obj in self._tracked.items()
            if (self._frame_index - obj.last_seen_frame) > self.persistence_frames
        ]
        for key in expired_keys:
            del self._tracked[key]

        # --- STEP 3: Build structured scene snapshot ---
        # Group all surviving tracked objects by label → set of directions
        grouped: dict[str, set[str]] = defaultdict(set)
        for (label, direction) in self._tracked:
            grouped[label].add(direction)

        # Store OCR results if provided
        if ocr_results is not None:
            self._active_ocr = ocr_results

        self._current_scene = {
            "objects":   {label: sorted(dirs) for label, dirs in grouped.items()},
            "ocr":       [r.to_dict() if hasattr(r, 'to_dict') else r
                          for r in self._active_ocr],
            "timestamp": time.time(),
        }

        return self._current_scene

    def get_speech_update(self) -> str | None:
        """
        Decide whether to announce the current scene.

        Returns a natural-language string if:
          1. The scene has changed meaningfully since the last announcement, AND
          2. The speech cooldown has elapsed.

        Returns None otherwise (caller should skip TTS this frame).
        """
        # Check cooldown
        now = time.monotonic()
        if self._last_spoken_time is not None:
            elapsed = now - self._last_spoken_time
            if elapsed < self.cooldown_seconds:
                return None   # too soon — stay quiet

        # Check for meaningful scene change
        current_snapshot = self._build_snapshot()
        if not self._scene_changed(current_snapshot):
            return None   # scene is the same — no need to repeat

        # Generate a natural summary (objects + OCR)
        summary = self._generate_summary(
            self._current_scene["objects"],
            self._current_scene.get("ocr", []),
        )

        if not summary:
            return None

        # Commit: update state so we don't re-announce the same scene
        self._last_announced_scene = current_snapshot["objects"]
        self._last_announced_ocr   = current_snapshot["ocr_texts"]
        self._last_spoken_time = now

        return summary

    def get_current_scene(self) -> dict:
        """Return the latest scene snapshot (useful for debugging/logging)."""
        return self._current_scene

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_snapshot(self) -> dict:
        """
        Build a hashable snapshot of the current scene for change detection.

        Returns a dict with two keys:
          - "objects"   : dict[label, frozenset[directions]]  — YOLO objects
          - "ocr_texts" : frozenset[str]  — lower-cased OCR texts

        Including OCR texts here means that when new text appears on a laptop
        screen (while the laptop object itself stays in the scene), the change
        detector still fires and TTS announces the new text.
        """
        objects = self._current_scene.get("objects", {})
        ocr_items = self._current_scene.get("ocr", [])

        obj_snapshot = {label: frozenset(dirs) for label, dirs in objects.items()}
        ocr_snapshot = frozenset(
            (item.get("text", "") if isinstance(item, dict) else item.text).lower()
            for item in ocr_items
        )
        return {"objects": obj_snapshot, "ocr_texts": ocr_snapshot}

    def _scene_changed(self, current_snapshot: dict) -> bool:
        """
        Return True if the scene is meaningfully different from the
        last announced scene.

        Checks both YOLO objects AND OCR texts so that new text appearing
        on an already-tracked object (e.g. a laptop screen refreshing) will
        still trigger a new announcement.
        """
        obj_snap = current_snapshot["objects"]
        ocr_snap = current_snapshot["ocr_texts"]

        if (not self._last_announced_scene and not obj_snap
                and not self._last_announced_ocr and not ocr_snap):
            return False   # both completely empty

        # --- OCR change check (new or gone text) ---
        if ocr_snap != self._last_announced_ocr:
            return True

        # --- Object change checks ---
        # New object types appeared
        new_labels = set(obj_snap) - set(self._last_announced_scene)
        if new_labels:
            return True

        # Object types that completely disappeared
        gone_labels = set(self._last_announced_scene) - set(obj_snap)
        if gone_labels:
            return True

        # Same labels, but direction(s) changed for any one of them
        for label, dirs in obj_snap.items():
            if dirs != self._last_announced_scene.get(label):
                return True

        return False   # nothing changed

    @staticmethod
    def _generate_summary(objects: dict[str, list[str]], ocr: list[dict] = None) -> str:
        """
        Turn the grouped objects dict into a natural English sentence.

        Examples
        --------
        {"person": ["center"]}
            → "Person ahead"

        {"person": ["center"], "bottle": ["left", "right"]}
            → "Person ahead, bottles on the left and right"

        {"chair": ["left"], "laptop": ["center"], "cup": ["right"]}
            → "Chair on the left, laptop ahead, cup on the right"
        """
        if not objects:
            return ""

        # Sort by priority so important objects come first in the sentence
        sorted_labels = sorted(
            objects.keys(),
            key=lambda lbl: OBJECT_PRIORITIES.get(lbl, 99),
        )

        # Cap sentence length to avoid overwhelming the user
        top_labels = sorted_labels[:MAX_OBJECTS_IN_SENTENCE]

        phrases: list[str] = []
        for label in top_labels:
            dirs: list[str] = objects[label]  # already sorted by process_frame

            # --- Direction phrase ---
            if len(dirs) == 1:
                dir_phrase = DIRECTION_PHRASES.get(dirs[0], f"on the {dirs[0]}")
            else:
                # Multiple directions: "on the left and right"
                human_dirs = [
                    DIRECTION_PHRASES.get(d, f"on the {d}") for d in dirs
                ]
                # Join last two with "and", rest with ", "
                dir_phrase = ", ".join(human_dirs[:-1]) + " and " + human_dirs[-1]

            # --- Pluralization ---
            # Pluralize if the object appears in more than one direction zone,
            # implying there are multiple instances.
            is_plural = len(dirs) > 1
            noun = _pluralize(label) if is_plural else label

            phrases.append(f"{noun} {dir_phrase}")

        # --- Append OCR phrases ---
        if ocr:
            from core.ocr_engine import generate_ocr_phrases, OCRResult  # late import
            # ocr items may be dicts (from snapshot) or OCRResult objects
            # convert dicts back to a simple namespace for generate_ocr_phrases
            class _R:
                pass
            ocr_objs = []
            for item in ocr:
                if isinstance(item, dict):
                    o = _R()
                    o.text = item.get("text", "")
                    o.position = item.get("position", "center")
                    o.source_object = item.get("source_object", "sign")
                    o.confidence = item.get("confidence", 1.0)
                    ocr_objs.append(o)
                else:
                    ocr_objs.append(item)
            ocr_phrases = generate_ocr_phrases(ocr_objs)
            phrases.extend(ocr_phrases)

        if not phrases:
            return ""

        # Assemble
        sentence = ", ".join(phrases)
        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:] if sentence else ""
        return sentence


# ---------------------------------------------------------------------------
# Standalone helper — simple English pluralization
# ---------------------------------------------------------------------------

def _pluralize(word: str) -> str:
    """
    Very lightweight English pluralization — covers 99% of COCO class names.
    No external libraries needed.
    """
    irregulars = {
        "person": "people",
        "mouse":  "mice",
        "knife":  "knives",
        "leaf":   "leaves",
    }
    if word in irregulars:
        return irregulars[word]

    # Words ending in -s, -x, -z, -ch, -sh → add "es"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"

    # Words ending in consonant + y → replace y with ies
    vowels = set("aeiou")
    if word.endswith("y") and len(word) > 1 and word[-2] not in vowels:
        return word[:-1] + "ies"

    # Default: just add "s"
    return word + "s"
