import pyttsx3
import sys

def test_pyttsx3_standalone():
    """
    Minimal standalone test for pyttsx3 on the main thread.
    If this fails, pyttsx3 may be fundamentally incompatible with your current macOS Python environment.
    """
    print("Initializing pyttsx3...")
    try:
        engine = pyttsx3.init()
        
        # macOS specific: sometimes default voice is broken, test changing it
        voices = engine.getProperty('voices')
        if voices:
            print(f"Using voice: {voices[0].name}")
            engine.setProperty('voice', voices[0].id)
            
        engine.setProperty('rate', 175)
        
        test_phrase = "Testing pyttsx3 on macOS main thread. If you can hear this, the engine works."
        print(f"Speaking: {test_phrase}")
        
        engine.say(test_phrase)
        engine.runAndWait()
        print("Finished speaking.")
        
    except Exception as e:
        print(f"Error during pyttsx3 execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_pyttsx3_standalone()
