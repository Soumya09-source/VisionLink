#!/usr/bin/env python3
"""
test_tts_stable.py - Test the stable TTS system
==================================================
Tests the existing CrossPlatformTTS system for smoothness,
interruption handling, and natural speech quality.
"""

import time
import threading
from feedback.tts import CrossPlatformTTS

def test_tts_functionality():
    """Test all TTS functionality comprehensively."""
    print("=== VisionAssist TTS System Test ===")
    print("Testing speech quality, interruption, and queue management\n")
    
    # Initialize TTS system
    tts = CrossPlatformTTS(rate=180, volume=0.9)
    
    print("✅ TTS System Initialized")
    print("Testing various speech scenarios...\n")
    
    # Test 1: Basic speech
    print("--- Test 1: Basic Speech ---")
    tts.speak("VisionAssist is ready.", direction="center")
    time.sleep(2)
    
    # Test 2: Spatial panning
    print("--- Test 2: Spatial Panning ---")
    tts.speak("Person on your left.", direction="left")
    time.sleep(2)
    tts.speak("Bottle on your right.", direction="right")
    time.sleep(2)
    
    # Test 3: Priority interrupt
    print("--- Test 3: Priority Interrupt ---")
    print("Starting long speech...")
    
    # Start long speech in background
    def long_speech():
        tts.speak("This is a very long sentence that should be interrupted by a critical alert. The system should immediately stop this speech and play the interrupting message without any delay or stuttering.", direction="center")
    
    speech_thread = threading.Thread(target=long_speech)
    speech_thread.start()
    
    # Let it start speaking
    time.sleep(1.5)
    
    print("Interrupting with critical alert...")
    tts.speak("Critical alert! Person detected!", direction="center", interrupt=True, priority=1)
    
    speech_thread.join(timeout=3)
    time.sleep(2)
    
    # Test 4: Queue management
    print("--- Test 4: Queue Management ---")
    print("Queueing multiple sentences rapidly...")
    
    # These should be queued and spoken smoothly
    tts.speak("First sentence.", direction="center")
    tts.speak("Second sentence.", direction="left")  
    tts.speak("Third sentence.", direction="right")
    
    # Wait for queue to process
    time.sleep(6)
    
    # Test 5: Duplicate suppression
    print("--- Test 5: Duplicate Suppression ---")
    print("Saying same sentence twice (should be suppressed)...")
    
    tts.speak("This is a test sentence.", direction="center")
    time.sleep(1)
    tts.speak("This is a test sentence.", direction="center")  # Should be suppressed
    
    time.sleep(3)
    
    # Test 6: Natural pacing
    print("--- Test 6: Natural Pacing ---")
    natural_text = "Person on your left, bottle ahead. Sign ahead says exit."
    tts.speak(natural_text, direction="center")
    time.sleep(4)
    
    # Test 7: Voice configuration
    print("--- Test 7: Voice Configuration ---")
    print("Testing different voice settings...")
    
    tts.set_rate(150)  # Slower
    tts.speak("This is slower speech.", direction="center")
    time.sleep(2)
    
    tts.set_rate(200)  # Normal
    tts.speak("This is normal speed.", direction="center")
    time.sleep(2)
    
    tts.set_rate(250)  # Faster
    tts.speak("This is faster speech.", direction="center")
    time.sleep(2)
    
    # Reset to normal
    tts.set_rate(180)
    
    print("\n=== Test Complete ===")
    print("✅ TTS system test completed!")
    print("Check for:")
    print("  • Smooth speech without stuttering")
    print("  • Proper spatial panning")
    print("  • Immediate interruption handling") 
    print("  • No duplicate sentences")
    print("  • Natural pacing and pauses")
    
    # Cleanup
    time.sleep(2)
    tts.cleanup()
    print("✅ TTS system cleaned up successfully")

def test_integration_with_visionassist():
    """Test TTS integration with simulated VisionAssist scenarios."""
    print("\n=== Integration Test with VisionAssist ===")
    
    tts = CrossPlatformTTS()
    
    # Simulate VisionAssist scenarios
    scenarios = [
        ("Person detected", "Person on your left.", "left"),
        ("Text found", "Sign ahead says exit.", "center"),
        ("Critical alert", "Vehicle approaching!", "right"),
        ("Memory command", "Person remembered.", "center"),
        ("Navigation", "Turn right ahead.", "right"),
    ]
    
    for scenario_name, text, direction in scenarios:
        print(f"\n--- {scenario_name} ---")
        tts.speak(text, direction=direction)
        time.sleep(2.5)
    
    tts.cleanup()
    print("\n✅ Integration test completed!")

def main():
    """Run comprehensive TTS tests."""
    print("VisionAssist Stable TTS Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Basic functionality
        test_tts_functionality()
        
        print("\n" + "=" * 50)
        input("Press Enter to continue with integration test...")
        
        # Test 2: Integration scenarios
        test_integration_with_visionassist()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 TTS testing completed!")

if __name__ == "__main__":
    main()
