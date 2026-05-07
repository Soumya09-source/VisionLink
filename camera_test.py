#!/usr/bin/env python3
"""
camera_test.py — Comprehensive Cross-Platform Camera Testing
================================================================
Tests all camera backends and indices to find working configurations.
Provides detailed diagnostics for troubleshooting camera issues.

Usage:
    python3 camera_test.py
"""

import cv2
import platform
import time
from typing import List, Tuple, Optional


class CameraDiagnostics:
    """Comprehensive camera testing and diagnostics."""
    
    # Platform-specific backend preferences
    BACKEND_PREFERENCES = {
        'Darwin': [cv2.CAP_AVFOUNDATION, cv2.CAP_FFMPEG, None],
        'Windows': [cv2.CAP_DSHOW, cv2.CAP_FFMPEG, None],
        'Linux': [cv2.CAP_V4L2, cv2.CAP_FFMPEG, None],
    }
    
    @staticmethod
    def get_backend_name(backend: Optional[int]) -> str:
        """Get human-readable backend name."""
        backend_names = {
            cv2.CAP_AVFOUNDATION: "AVFoundation",
            cv2.CAP_DSHOW: "DirectShow", 
            cv2.CAP_V4L2: "V4L2",
            cv2.CAP_FFMPEG: "FFMPEG",
            None: "Default"
        }
        return backend_names.get(backend, "Unknown")
    
    @staticmethod
    def test_all_cameras(max_indices: int = 5) -> List[Tuple[int, Optional[int], bool, str]]:
        """
        Test all camera combinations and return results.
        
        Returns:
            List of (index, backend, success, details) tuples
        """
        system = platform.system()
        backends = CameraDiagnostics.BACKEND_PREFERENCES.get(system, [None])
        results = []
        
        print(f"=== Camera Diagnostics for {system} ===")
        print(f"Testing backends: {[CameraDiagnostics.get_backend_name(b) for b in backends]}")
        print()
        
        for backend in backends:
            backend_name = CameraDiagnostics.get_backend_name(backend)
            print(f"--- Testing {backend_name} Backend ---")
            
            for index in range(max_indices):
                print(f"Camera {index}: ", end="")
                
                try:
                    if backend is not None:
                        cap = cv2.VideoCapture(index, backend)
                    else:
                        cap = cv2.VideoCapture(index)
                    
                    if cap.isOpened():
                        # Test actual frame capture
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            width, height = frame.shape[1], frame.shape[0]
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            print(f"✓ SUCCESS ({width}x{height} @ {fps:.1f}fps)")
                            results.append((index, backend, True, f"{width}x{height} @ {fps:.1f}fps"))
                        else:
                            print("✗ OPENED but no frame")
                            results.append((index, backend, False, "Opened but no frame"))
                        cap.release()
                    else:
                        print("✗ FAILED to open")
                        results.append((index, backend, False, "Failed to open"))
                        
                except Exception as e:
                    print(f"✗ EXCEPTION: {e}")
                    results.append((index, backend, False, f"Exception: {e}"))
            
            print()
        
        return results
    
    @staticmethod
    def find_working_cameras(max_indices: int = 5) -> List[Tuple[int, Optional[int], str]]:
        """Find all working camera configurations."""
        results = CameraDiagnostics.test_all_cameras(max_indices)
        working = [(idx, backend, details) for idx, backend, success, details in results if success]
        return working
    
    @staticmethod
    def test_best_camera(max_indices: int = 5):
        """Test and display the best camera configuration."""
        working = CameraDiagnostics.find_working_cameras(max_indices)
        
        if not working:
            print("❌ No working cameras found!")
            print("\nTroubleshooting:")
            print("• macOS: System Settings → Privacy & Security → Camera → allow Terminal/VS Code")
            print("• Windows: Close other apps using the camera (Zoom, Teams, etc.)")
            print("• Check camera is connected and not covered")
            print("• Try unplugging and reconnecting the camera")
            return None
        
        print(f"✅ Found {len(working)} working camera(s):")
        for i, (idx, backend, details) in enumerate(working):
            backend_name = CameraDiagnostics.get_backend_name(backend)
            print(f"  {i+1}. Index {idx} with {backend_name} ({details})")
        
        # Return the first working camera
        best_idx, best_backend, best_details = working[0]
        return best_idx, best_backend


def test_camera_stream():
    """Test continuous camera streaming with performance metrics."""
    print("\n=== Camera Stream Test ===")
    
    # Find best camera
    best_config = CameraDiagnostics.test_best_camera()
    if best_config is None:
        return
    
    index, backend = best_config
    backend_name = CameraDiagnostics.get_backend_name(backend)
    print(f"\nUsing camera {index} with {backend_name} backend")
    
    try:
        if backend is not None:
            cap = cv2.VideoCapture(index, backend)
        else:
            cap = cv2.VideoCapture(index)
        
        if not cap.isOpened():
            print("Failed to open camera for streaming test")
            return
        
        # Configure for performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Streaming test started. Press 'q' to quit.")
        print("Press 'f' for FPS info, 'i' for camera info")
        
        frame_count = 0
        start_time = time.time()
        last_fps_time = start_time
        
        while True:
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame_count += 1
                cv2.imshow("Camera Stream Test", frame)
                
                # Show FPS every 2 seconds
                current_time = time.time()
                if current_time - last_fps_time >= 2.0:
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed
                    print(f"FPS: {fps:.1f} | Frames: {frame_count} | Time: {elapsed:.1f}s")
                    last_fps_time = current_time
            else:
                print("Failed to read frame")
                break
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Current FPS: {fps:.1f} | Total frames: {frame_count}")
            elif key == ord('i'):
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                buffer = int(cap.get(cv2.CAP_PROP_BUFFERSIZE))
                print(f"Resolution: {width}x{height}")
                print(f"Camera FPS: {fps}")
                print(f"Buffer size: {buffer}")
        
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nStream Test Results:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        
    except Exception as e:
        print(f"Stream test error: {e}")


def main():
    """Main test routine."""
    print("VisionAssist Camera Diagnostics")
    print("=" * 40)
    
    # Test all cameras
    CameraDiagnostics.test_all_cameras()
    
    # Test streaming
    test_camera_stream()


if __name__ == "__main__":
    main()
