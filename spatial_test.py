import sys
import os

# Add parent directory to path so we can import feedback.spatial_audio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feedback.spatial_audio import SpatialAudio
import time

def run_test():
    print("Initializing SpatialAudio...")
    audio = SpatialAudio()
    
    print("\n--- Testing LEFT Pan ---")
    audio.play("This is a test of the left channel.", "left")
    time.sleep(4)
    
    print("\n--- Testing RIGHT Pan ---")
    audio.play("This is a test of the right channel.", "right")
    time.sleep(4)
    
    print("\n--- Testing CENTER Pan ---")
    audio.play("This is a test of the center channel.", "center")
    time.sleep(4)
    
    print("\nTest completed.")

if __name__ == "__main__":
    run_test()
