#!/usr/bin/env python3
"""
test_voice_integration.py — Test voice command integration
=========================================================
Tests the voice command manager exactly as used in main.py
to verify the issue and demonstrate the fix.
"""

import time
from input.voice_commands import VoiceCommandManager

def dummy_command(cmd):
    print(f"[CALLBACK] Received command: '{cmd}'")

def main():
    print("=== Voice Integration Test ===")
    print("This simulates how main.py uses voice commands.")
    print("Current behavior: Only listens when listen_once() is called.")
    print()
    
    voice_manager = VoiceCommandManager(command_callback=dummy_command)
    
    print("1. Testing manual trigger (current main.py behavior):")
    print("   Say 'what do you see' now...")
    voice_manager.listen_once(timeout=5.0)
    
    time.sleep(2)
    
    print("\n2. Testing continuous listening (proposed fix):")
    print("   Say commands continuously for 10 seconds...")
    start_time = time.time()
    while time.time() - start_time < 10:
        voice_manager.listen_once(timeout=2.0)
        time.sleep(0.5)  # Small gap between listens
    
    print("\n=== Test complete ===")

if __name__ == "__main__":
    main()
