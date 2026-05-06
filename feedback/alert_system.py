"""
feedback/alert_system.py — VisionAssist Alert & Priority Layer
================================================================

PURPOSE
-------
Consumes the canonical scene JSON from SceneBuilder and decides what to
announce.  It suppresses low-value noise, highlights high-priority objects,
and triggers immediate interruptions for critical alerts (STOP signs, vehicles).

INPUT: canonical scene dict (scene["objects"] is now list[dict]).
OUTPUT: (speech_string | None, is_critical bool)
"""

import time

# --- Priority Tables ---
CRITICAL_OCR_TEXT  = {"stop", "exit", "danger", "warning", "caution"}
CRITICAL_OBJECTS   = {"car", "bus", "truck"}
HIGH_PRIORITY_OBJECTS = {"person", "bicycle", "motorcycle"}

# Direction → human-readable phrase
DIRECTION_PHRASES = {
    "left":   "on the left",
    "center": "ahead",
    "right":  "on the right",
}


class AlertSystem:
    def __init__(self, cooldown_seconds=5.0, critical_cooldown_seconds=3.0):
        self.cooldown_seconds          = cooldown_seconds
        self.critical_cooldown_seconds = critical_cooldown_seconds

        self._last_spoken_time      = 0.0
        self._last_critical_time    = 0.0
        self._last_announced_snapshot = None

    def process_scene(self, scene: dict) -> tuple[str | None, bool]:
        """
        Evaluate canonical scene JSON and decide what to say.

        Returns
        -------
        (speech_string, is_critical)  or  (None, False)
        """
        now     = time.monotonic()
        objects = scene.get("objects", [])   # list[dict] in new schema
        texts   = scene.get("text", [])      # list[dict] in new schema

        # ── 1. Critical alerts ────────────────────────────────────────────────
        critical_alerts: list[str] = []

        for t in texts:
            content  = (t.get("content", "") if isinstance(t, dict) else "").lower()
            position = t.get("position", "center") if isinstance(t, dict) else "center"
            if any(crit in content for crit in CRITICAL_OCR_TEXT):
                critical_alerts.append(
                    f"{content.upper()} sign {DIRECTION_PHRASES.get(position, 'ahead')}"
                )

        for obj in objects:
            label    = obj.get("label", "") if isinstance(obj, dict) else ""
            position = obj.get("position", "center") if isinstance(obj, dict) else "center"
            if label in CRITICAL_OBJECTS:
                critical_alerts.append(f"{label} {DIRECTION_PHRASES.get(position, 'ahead')}")

        if critical_alerts:
            if now - self._last_critical_time > self.critical_cooldown_seconds:
                self._last_critical_time = now
                alert_text = "! ".join(critical_alerts) + "!"
                alert_text = alert_text[0].upper() + alert_text[1:]
                return alert_text, True

        # ── 2. Normal / high priority alerts ──────────────────────────────────
        current_snapshot = self._build_snapshot(scene)

        if self._last_announced_snapshot and not self._scene_changed(
            self._last_announced_snapshot, current_snapshot
        ):
            return None, False

        if now - self._last_spoken_time < self.cooldown_seconds:
            return None, False

        speech = self.generate_summary(objects, texts)
        if not speech:
            return None, False

        self._last_spoken_time        = now
        self._last_announced_snapshot = current_snapshot
        return speech, False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def generate_summary(self, objects: list[dict], texts: list[dict]) -> str:
        """
        Generate a natural-language summary from canonical scene lists.
        Called both by process_scene() and by voice-command handlers.
        """
        if not objects and not texts:
            return ""

        from collections import defaultdict

        # Score/sort objects by priority
        scored = sorted(objects, key=lambda o: _priority_rank(o.get("label", "")))
        top = scored[:3]

        label_positions: dict[str, list[str]] = defaultdict(list)
        for obj in top:
            label_positions[obj["label"]].append(obj["position"])

        phrases: list[str] = []
        for label in [o["label"] for o in top if o["label"] in label_positions]:
            if label not in label_positions:
                continue
            positions = sorted(set(label_positions.pop(label)))   # consume so no duplicates
            if len(positions) == 1:
                dir_phrase = DIRECTION_PHRASES.get(positions[0], f"on the {positions[0]}")
            else:
                human = [DIRECTION_PHRASES.get(p, f"on the {p}") for p in positions]
                dir_phrase = ", ".join(human[:-1]) + " and " + human[-1]

            is_plural = len(positions) > 1
            noun = _pluralize(label) if is_plural else label
            phrases.append(f"{noun} {dir_phrase}")

        for t in texts:
            content  = t.get("content", "").strip() if isinstance(t, dict) else ""
            position = t.get("position", "center") if isinstance(t, dict) else "center"
            src      = t.get("source_object", "sign") if isinstance(t, dict) else "sign"
            if content:
                dir_phrase = DIRECTION_PHRASES.get(position, f"on the {position}")
                phrases.append(f"{src} {dir_phrase} says {content}")

        if not phrases:
            return ""
        sentence = ", ".join(phrases)
        return sentence[0].upper() + sentence[1:]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_snapshot(self, scene: dict) -> dict:
        objects = scene.get("objects", [])
        texts   = scene.get("text", [])

        obj_snap = frozenset(
            (o.get("label", ""), o.get("position", ""))
            for o in objects if isinstance(o, dict)
        )
        txt_snap = frozenset(
            t.get("content", "").lower()
            for t in texts if isinstance(t, dict)
        )
        return {"objects": obj_snap, "texts": txt_snap}

    def _scene_changed(self, old: dict, new: dict) -> bool:
        return old["objects"] != new["objects"] or old["texts"] != new["texts"]

    # kept for backward compat if anything still calls it
    def pluralize(self, word: str) -> str:
        return _pluralize(word)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _priority_rank(label: str) -> int:
    table = {
        "person": 1, "car": 2, "bus": 2, "truck": 2,
        "bicycle": 3, "motorcycle": 3,
        "chair": 4, "couch": 4, "bed": 4, "table": 4,
        "dog": 5, "cat": 5, "bottle": 6, "cup": 6, "bowl": 6,
        "laptop": 7, "cell phone": 7, "remote": 7, "book": 8, "clock": 8,
    }
    return table.get(label, 99)


def _pluralize(word: str) -> str:
    irregulars = {"person": "people", "mouse": "mice", "knife": "knives"}
    if word in irregulars:
        return irregulars[word]
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"
