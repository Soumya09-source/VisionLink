"""
core/openclaw_bridge.py — VisionLink ↔ OpenClaw Bridge
Minimal, reliable, demo-ready.
"""
import requests
import hashlib
import time

OPENCLAW_URL = "http://127.0.0.1:18789/v1/chat/completions"

class OpenClawBridge:
    def __init__(self):
        self._last_hash = ""
        self._last_call_time = 0
        self._min_interval = 6.5# seconds between OpenClaw calls

    def _scene_hash(self, scene: dict) -> str:
        objects = tuple(sorted(
            (o["label"], o["position"]) for o in scene.get("objects", [])
        ))
        texts = tuple(t["content"] for t in scene.get("text", []))
        return hashlib.md5(str((objects, texts)).encode()).hexdigest()

    def narrate(self, scene: dict, is_critical: bool) -> str | None:
        """
        Send scene to OpenClaw. Returns natural narration string or None.
        Falls back to None (caller uses original speech) if anything fails.
        """
        now = time.time()

        # Spam prevention: skip if scene unchanged and not critical
        scene_hash = self._scene_hash(scene)
        if not is_critical:
            if scene_hash == self._last_hash:
                return None
            if (now - self._last_call_time) < self._min_interval:
                return None

        self._last_hash = scene_hash
        self._last_call_time = now

        try:
            objects = [
                f"{o['label']} ({o['position']})"
                for o in scene.get("objects", [])
            ]
            texts = [t["content"] for t in scene.get("text", [])]

            prompt = (
                "You are a real-time visual assistant for a blind person. "
                "Describe the scene in ONE short natural sentence (max 12 words). "
                "Be specific and helpful. Do NOT list items robotically.\n\n"
                f"Objects detected: {', '.join(objects) or 'none'}\n"
                f"Text detected: {', '.join(texts) or 'none'}\n"
                f"Critical alert: {is_critical}\n\n"
                "Respond with ONLY the sentence to speak aloud. "
                "If the scene is trivial or unchanged, respond with SKIP."
            )

            response = requests.post(
                OPENCLAW_URL,
                headers={
                 "Authorization": "Bearer 73178da61098c134815289cb7f837a768dfcd2b395a6dbed",
                 "Content-Type": "application/json"
},
                json={
                    "model": "openclaw/default",
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=2.5
            )

            data = response.json()
            reply = data["choices"][0]["message"]["content"].strip()
            print(f"[OpenClaw] → {reply}")

            if reply.upper().startswith("SKIP"):
                return None
            return reply

        except Exception as e:
            print(f"[OpenClaw] Fallback (offline/timeout): {e}")
            return None  # caller uses original AlertSystem speech