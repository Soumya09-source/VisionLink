import time
import logging
from typing import Dict

class SpeechCooldownManager:
    """
    Manages deduplication and cooldowns for speech phrases.
    Prevents the assistant from repeating the exact same phrase too frequently.
    """
    
    def __init__(self, default_cooldown: float = 5.0):
        self.logger = logging.getLogger("SpeechCooldownManager")
        self.default_cooldown = default_cooldown
        self._last_spoken: Dict[str, float] = {}

    def is_on_cooldown(self, text: str, custom_cooldown: float = None) -> bool:
        """
        Checks if a specific text is currently on cooldown.
        Returns True if on cooldown (should not speak), False otherwise.
        """
        now = time.time()
        cooldown = custom_cooldown if custom_cooldown is not None else self.default_cooldown
        
        # Normalize text to lower case for consistent matching
        normalized_text = text.strip().lower()
        
        last_time = self._last_spoken.get(normalized_text, 0.0)
        
        if now - last_time < cooldown:
            return True
            
        return False

    def record_speech(self, text: str) -> None:
        """
        Records that a phrase was spoken right now, starting its cooldown.
        """
        now = time.time()
        normalized_text = text.strip().lower()
        self._last_spoken[normalized_text] = now

    def clear_cache(self) -> None:
        """Clears the deduplication cache."""
        self._last_spoken.clear()
        self.logger.debug("[CooldownManager] Deduplication cache cleared.")
