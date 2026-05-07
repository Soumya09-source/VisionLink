import platform
import pyttsx3
import logging

class PlatformDetector:
    @staticmethod
    def get_os() -> str:
        os_name = platform.system().lower()
        if os_name == "darwin":
            return "macos"
        elif os_name == "windows":
            return "windows"
        else:
            return "linux"

class VoiceConfigManager:
    """Manages cross-platform voices, rate, and volume for pyttsx3."""
    
    def __init__(self, target_voice: str = "Samantha", rate: int = 175, volume: float = 1.0):
        self.os_name = PlatformDetector.get_os()
        self.target_voice = target_voice
        self.rate = rate
        self.volume = volume
        self.logger = logging.getLogger("VoiceConfigManager")

    def configure_engine(self, engine: pyttsx3.Engine) -> None:
        """Applies platform-specific configurations to the pyttsx3 engine."""
        try:
            # 1. Set Rate
            engine.setProperty('rate', self.rate)
            # 2. Set Volume
            engine.setProperty('volume', self.volume)
            # 3. Set Voice
            self._set_best_voice(engine)
        except Exception as e:
            self.logger.error(f"[VoiceConfig Error] Failed to configure engine: {e}")

    def _set_best_voice(self, engine: pyttsx3.Engine) -> None:
        voices = engine.getProperty('voices')
        if not voices:
            self.logger.warning("[VoiceConfig] No voices found by pyttsx3.")
            return

        # Try to find the exact target voice (case-insensitive)
        for voice in voices:
            if self.target_voice.lower() in voice.name.lower():
                engine.setProperty('voice', voice.id)
                self.logger.info(f"[VoiceConfig] Found target voice: {voice.name}")
                return

        # Fallback 1: English Female
        for voice in voices:
            name_lower = voice.name.lower()
            if "zira" in name_lower or "samantha" in name_lower or ("en" in voice.id.lower() and "female" in name_lower):
                engine.setProperty('voice', voice.id)
                self.logger.info(f"[VoiceConfig] Using fallback English female voice: {voice.name}")
                return

        # Fallback 2: Any English Voice
        for voice in voices:
            if "en" in voice.id.lower() or "english" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                self.logger.info(f"[VoiceConfig] Using fallback English voice: {voice.name}")
                return

        # Fallback 3: System Default (do nothing)
        self.logger.warning("[VoiceConfig] No suitable English voice found, using system default.")
