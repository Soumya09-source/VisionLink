"""
memory/visual_memory.py — VisionAssist Visual Memory Layer
============================================================

PURPOSE
-------
Provides persistent object memory across sessions. When a user says
"remember this", the highest-priority detected object is stored with
its position and timestamp. Later, commands like "find bottle" or
"what did you remember" query this store.

STORAGE
-------
Uses a single human-readable JSON file (memory_store.json by default).
Lightweight, offline, and requires no external dependencies.

FORMAT
------
{
  "memories": [
    {
      "label": "bottle",
      "position": "left",
      "timestamp": 1715000000.0,
      "ocr_context": null
    }
  ]
}
"""

import json
import os
import time
import threading
from typing import Optional


# Priority table — lower number = more important.
# Mirrors the priority table in scene_builder to pick the "most important"
# object when the user says "remember this" without specifying a label.
_OBJECT_PRIORITIES: dict[str, int] = {
    "person": 1,
    "car": 2, "bus": 2, "truck": 2,
    "bicycle": 3, "motorcycle": 3,
    "chair": 4, "couch": 4, "bed": 4, "table": 4,
    "dog": 5, "cat": 5,
    "bottle": 6, "cup": 6, "bowl": 6,
    "laptop": 7, "cell phone": 7, "remote": 7,
    "book": 8, "clock": 8,
}


class VisualMemory:
    """
    Lightweight, JSON-backed persistent visual memory store.

    Usage
    -----
    memory = VisualMemory()

    # Save an object
    memory.remember_object("bottle", "left")

    # Recall all memories
    for m in memory.get_all_memories():
        print(m["label"], m["position"])

    # Find a specific object
    mem = memory.find_memory("bottle")

    # Forget an object
    memory.forget_memory("bottle")
    """

    def __init__(self, store_path: str = "memory_store.json"):
        self.store_path = store_path
        self._lock = threading.Lock()          # thread-safe file writes
        self._cache: list[dict] = []           # in-memory cache for fast reads
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember_object(
        self,
        label: str,
        position: str,
        ocr_context: Optional[str] = None,
    ) -> dict:
        """
        Save or update an object in memory.

        If an object with the same label already exists it is *updated*
        (position and timestamp refreshed) rather than duplicated.

        Returns the saved memory record.
        """
        label = label.lower().strip()
        position = position.lower().strip()

        record = {
            "label": label,
            "position": position,
            "timestamp": time.time(),
            "ocr_context": ocr_context,
        }

        with self._lock:
            # Update existing entry if label already remembered
            for i, m in enumerate(self._cache):
                if m["label"] == label:
                    self._cache[i] = record
                    self._save_locked()
                    print(f"[MEMORY] Updated: {label} @ {position}")
                    return record

            # Otherwise append
            self._cache.append(record)
            self._save_locked()

        print(f"[MEMORY] Saved: {label} @ {position}")
        return record

    def get_all_memories(self) -> list[dict]:
        """Return all stored memories, sorted newest-first."""
        with self._lock:
            return sorted(self._cache, key=lambda m: m["timestamp"], reverse=True)

    def find_memory(self, label: str) -> Optional[dict]:
        """
        Look up a specific object by label (case-insensitive, exact match).
        Returns the record dict or None if not found.
        """
        label = label.lower().strip()
        with self._lock:
            for m in self._cache:
                if m["label"] == label:
                    print(f"[MEMORY] Retrieved: {label}")
                    return m
        print(f"[MEMORY] Not found: {label}")
        return None

    def forget_memory(self, label: str) -> bool:
        """
        Remove an object from memory by label.
        Returns True if something was removed, False if not found.
        """
        label = label.lower().strip()
        with self._lock:
            before = len(self._cache)
            self._cache = [m for m in self._cache if m["label"] != label]
            removed = len(self._cache) < before
            if removed:
                self._save_locked()
                print(f"[MEMORY] Forgotten: {label}")
            else:
                print(f"[MEMORY] Cannot forget — not in memory: {label}")
        return removed

    def clear_all(self) -> None:
        """Wipe the entire memory store."""
        with self._lock:
            self._cache = []
            self._save_locked()
        print("[MEMORY] All memories cleared.")

    # ------------------------------------------------------------------
    # Helper: pick best object from a live scene
    # ------------------------------------------------------------------

    @staticmethod
    def pick_priority_object(scene: dict) -> Optional[tuple[str, str]]:
        """
        Given a canonical scene snapshot from SceneBuilder, return (label, position)
        for the highest-priority object currently visible.
        Returns None if the scene has no objects.

        Supports both new canonical format (objects: list[dict]) and the
        legacy format (objects: dict[str, list[str]]) for robustness.
        """
        objects = scene.get("objects", [])
        if not objects:
            return None

        # New canonical format: objects is list[dict]
        if isinstance(objects, list):
            if not objects:
                return None
            # already sorted by priority from SceneBuilder
            best = sorted(objects, key=lambda o: _OBJECT_PRIORITIES.get(o.get("label", ""), 99))[0]
            return best["label"], best["position"]

        # Legacy format: objects is dict[str, list[str]]
        sorted_labels = sorted(objects.keys(), key=lambda lbl: _OBJECT_PRIORITIES.get(lbl, 99))
        best_label = sorted_labels[0]
        positions = objects[best_label]
        best_position = positions[0] if positions else "center"
        return best_label, best_position

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load memories from disk into cache. Safe to call even if file is missing."""
        if not os.path.exists(self.store_path):
            self._cache = []
            print(f"[MEMORY] No existing store at '{self.store_path}' — starting fresh.")
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._cache = data.get("memories", [])
            print(f"[MEMORY] Loaded {len(self._cache)} memories from '{self.store_path}'.")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[MEMORY ERROR] Could not load store: {e}. Starting fresh.")
            self._cache = []

    def _save_locked(self) -> None:
        """
        Write the in-memory cache to disk.
        MUST be called while holding self._lock.
        """
        try:
            data = {"memories": self._cache}
            # Write to a temp file first then rename — prevents corruption if
            # the process is killed mid-write.
            tmp_path = self.store_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.store_path)
        except OSError as e:
            print(f"[MEMORY ERROR] Failed to save store: {e}")
