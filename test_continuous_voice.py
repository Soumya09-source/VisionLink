#!/usr/bin/env python3
"""
test_continuous_voice.py — Test continuous voice command listening
==================================================================
Tests the new continuous listening feature for VoiceCommandManager.
"""

import time
import signal
import sys
from input.voice_commands import VoiceCommandManager

def handle_command(cmd):
    """Handle recognized voice commands."""
    print(f"\n[EXECUTED] Command: '{cmd}'")
    
    if cmd == "what do you see":
        print("  → Would describe current scene")
    elif cmd == "read text":
        print("  → Would read visible text")
    elif cmd == "stop speaking":
        print("  → Would stop TTS")
    elif cmd.startswith("find "):
        target = cmd[5:].strip()
        print(f"  → Would find {target}")
    elif cmd.startswith("forget "):
        target = cmd[7:].strip()
        print(f"  → Would forget {target}")
    else:
        print(f"  → Would handle: {cmd}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nStopping continuous listening...")
    voice_manager.stop_continuous()
    sys.exit(0)

if __name__ == "__main__":
    print("=== Continuous Voice Command Test ===")
    print("Say commands like:")
    print("  - 'what do you see'")
    print("  - 'read text'")
    print("  - 'stop speaking'")
    print("  - 'find bottle'")
    print("  - 'forget chair'")
    print("\nPress Ctrl+C to stop.\n")
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create voice manager
    voice_manager = VoiceCommandManager(command_callback=handle_command)
    
    # Start continuous listening
    voice_manager.start_continuous(listen_window=3.0, gap=1.0)
    
    print("Listening continuously... Speak now!\n")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
