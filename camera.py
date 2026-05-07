import cv2
import threading
import platform
import time
from typing import Optional, Tuple, List


class CameraInitializer:
    """Cross-platform camera initialization with backend detection and fallback."""
    
    # Platform-specific backend preferences
    BACKEND_PREFERENCES = {
        'Darwin': [cv2.CAP_AVFOUNDATION, cv2.CAP_FFMPEG, None],  # macOS
        'Windows': [cv2.CAP_DSHOW, cv2.CAP_FFMPEG, None],       # Windows
        'Linux': [cv2.CAP_V4L2, cv2.CAP_FFMPEG, None],          # Linux
    }
    
    @staticmethod
    def get_platform() -> str:
        """Get the current platform name."""
        return platform.system()
    
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
    
    @classmethod
    def initialize_camera(cls, max_attempts: int = 3) -> Tuple[cv2.VideoCapture, int, Optional[int]]:
        """
        Initialize camera with cross-platform backend detection and index fallback.
        
        Returns:
            Tuple of (VideoCapture object, working index, backend used)
        
        Raises:
            RuntimeError: If no camera can be initialized after all attempts
        """
        system = cls.get_platform()
        backends = cls.BACKEND_PREFERENCES.get(system, [None])
        
        print(f"[CAMERA] Platform: {system}")
        print(f"[CAMERA] Trying backends in order: {[cls.get_backend_name(b) for b in backends]}")
        
        for backend in backends:
            backend_name = cls.get_backend_name(backend)
            print(f"[CAMERA] Testing backend: {backend_name}")
            
            # Try different camera indices
            for index in range(max_attempts):
                print(f"[CAMERA] Trying camera index {index} with {backend_name}...")
                
                try:
                    if backend is not None:
                        cap = cv2.VideoCapture(index, backend)
                    else:
                        cap = cv2.VideoCapture(index)
                    
                    if cap.isOpened():
                        # Test if we can actually read a frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"[CAMERA] ✓ Success: index {index}, backend {backend_name}")
                            return cap, index, backend
                        else:
                            print(f"[CAMERA] ✗ Can read but no frame from index {index}")
                            cap.release()
                    else:
                        print(f"[CAMERA] ✗ Failed to open index {index}")
                        if backend is not None:
                            cap.release()
                        
                except Exception as e:
                    print(f"[CAMERA] ✗ Exception with index {index}, {backend_name}: {e}")
                    try:
                        cap.release()
                    except:
                        pass
        
        # If we get here, all attempts failed
        raise RuntimeError(
            f"[CAMERA] Failed to initialize any camera on {system}. "
            f"Tried {max_attempts} indices with backends: {[cls.get_backend_name(b) for b in backends]}. "
            f"Check camera permissions and ensure no other app is using the camera."
        )


class CameraStream:
    """
    Continuously captures frames from a webcam in a background thread.
    Cross-platform with robust initialization and error handling.
    """

    def __init__(self, index: Optional[int] = None, max_camera_attempts: int = 3):
        """
        Initialize camera stream with automatic backend detection.
        
        Args:
            index: Specific camera index to use (None for auto-detection)
            max_camera_attempts: Maximum camera indices to try when auto-detecting
        """
        print("[CAMERA] Initializing CameraStream...")
        
        self._cap = None
        self._camera_index = 0
        self._backend = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._error_count = 0
        self._max_errors = 10  # Max consecutive errors before giving up
        
        try:
            if index is not None:
                # Use specific index with auto backend detection
                system = platform.system()
                backends = CameraInitializer.BACKEND_PREFERENCES.get(system, [None])
                
                for backend in backends:
                    backend_name = CameraInitializer.get_backend_name(backend)
                    print(f"[CAMERA] Using specific index {index} with backend {backend_name}")
                    
                    try:
                        if backend is not None:
                            self._cap = cv2.VideoCapture(index, backend)
                        else:
                            self._cap = cv2.VideoCapture(index)
                        
                        if self._cap.isOpened():
                            ret, test_frame = self._cap.read()
                            if ret and test_frame is not None:
                                self._camera_index = index
                                self._backend = backend
                                print(f"[CAMERA] ✓ Success: index {index}, backend {backend_name}")
                                break
                            else:
                                print(f"[CAMERA] ✗ Index {index} opened but no frame")
                                self._cap.release()
                                self._cap = None
                        else:
                            print(f"[CAMERA] ✗ Failed to open index {index} with {backend_name}")
                            if self._cap is not None:
                                self._cap.release()
                                self._cap = None
                    except Exception as e:
                        print(f"[CAMERA] ✗ Exception with index {index}, {backend_name}: {e}")
                        if self._cap is not None:
                            try:
                                self._cap.release()
                            except:
                                pass
                        self._cap = None
                
                if self._cap is None:
                    raise RuntimeError(f"Failed to open camera at specific index {index}")
            else:
                # Auto-detect working camera
                self._cap, self._camera_index, self._backend = CameraInitializer.initialize_camera(max_camera_attempts)
            
            # Set camera properties for better performance
            self._configure_camera()
            
            # Start capture thread
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            
            print(f"[CAMERA] CameraStream initialized successfully")
            print(f"[CAMERA] Index: {self._camera_index}, Backend: {CameraInitializer.get_backend_name(self._backend)}")
            
        except Exception as e:
            print(f"[CAMERA] Initialization failed: {e}")
            self.cleanup()
            raise
    
    def _configure_camera(self):
        """Configure camera properties for optimal performance."""
        if self._cap is None:
            return
        
        try:
            # Set reasonable defaults
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
            
            # Verify settings were applied
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            
            print(f"[CAMERA] Camera configured: {width}x{height} @ {fps:.1f}fps")
            
        except Exception as e:
            print(f"[CAMERA] Warning: Could not configure camera properties: {e}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Read frames continuously until stop() is called."""
        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    print("[CAMERA] Camera disconnected, attempting to reconnect...")
                    self._error_count += 1
                    if self._error_count >= self._max_errors:
                        print("[CAMERA] Too many errors, stopping capture loop")
                        break
                    
                    time.sleep(0.5)
                    continue
                
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._error_count += 1
                    print(f"[CAMERA] Failed to read frame (error {self._error_count}/{self._max_errors})")
                    
                    if self._error_count >= self._max_errors:
                        print("[CAMERA] Too many consecutive read failures, stopping")
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Reset error count on successful read
                self._error_count = 0
                
                with self._lock:
                    self._frame = frame.copy()
                    
            except Exception as e:
                self._error_count += 1
                print(f"[CAMERA] Exception in capture loop: {e} (error {self._error_count}/{self._max_errors})")
                
                if self._error_count >= self._max_errors:
                    print("[CAMERA] Too many exceptions, stopping capture loop")
                    break
                    
                time.sleep(0.2)
        
        print("[CAMERA] Capture loop ended")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self):
        """Return the most recent frame, or None if none captured yet."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def get_camera_info(self) -> dict:
        """Get information about the current camera configuration."""
        info = {
            'index': self._camera_index,
            'backend': CameraInitializer.get_backend_name(self._backend),
            'platform': platform.system(),
            'is_running': self._running,
            'error_count': self._error_count,
        }
        
        if self._cap is not None and self._cap.isOpened():
            try:
                info.update({
                    'width': int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fps': self._cap.get(cv2.CAP_PROP_FPS),
                    'buffer_size': int(self._cap.get(cv2.CAP_PROP_BUFFERSIZE)),
                })
            except Exception as e:
                info['property_error'] = str(e)
        
        return info

    def is_healthy(self) -> bool:
        """Check if the camera stream is healthy."""
        if self._cap is None or not self._cap.isOpened():
            return False
        return self._error_count < self._max_errors and self._running

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        print("[CAMERA] Stopping CameraStream...")
        
        # Signal thread to stop
        self._running = False
        
        # Wait for thread to finish
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                print("[CAMERA] Warning: Thread did not stop cleanly")
        
        # Clean up resources
        self.cleanup()
        print("[CAMERA] CameraStream stopped")

    def cleanup(self):
        """Release camera resources."""
        if self._cap is not None:
            try:
                self._cap.release()
                print("[CAMERA] Camera released")
            except Exception as e:
                print(f"[CAMERA] Error releasing camera: {e}")
            finally:
                self._cap = None
        
        with self._lock:
            self._frame = None


# ----------------------------------------------------------------------
# Quick test — press 'q' to quit
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Camera Test ===")
    print("Press 'q' to quit, 'i' for camera info")
    
    try:
        # Test auto-detection
        stream = CameraStream()  # Auto-detect camera
        
        print("\nCamera initialized successfully!")
        print("Camera Info:", stream.get_camera_info())
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            frame = stream.get_frame()
            
            if frame is not None:
                frame_count += 1
                cv2.imshow("Camera Feed", frame)
                
                # Show FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"[CAMERA] FPS: {fps:.1f}, Frames: {frame_count}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("i"):
                info = stream.get_camera_info()
                print("\n=== Camera Information ===")
                for k, v in info.items():
                    print(f"{k}: {v}")
                print("========================\n")
            elif key == ord("h"):
                print(f"Camera healthy: {stream.is_healthy()}")
            
    except RuntimeError as e:
        print(f"[CAMERA ERROR] {e}")
        print("\nTroubleshooting tips:")
        print("• macOS: System Settings → Privacy & Security → Camera → allow Terminal/VS Code")
        print("• Windows: Ensure no other app is using the camera")
        print("• Check camera is connected and not covered")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        try:
            if 'stream' in locals():
                stream.stop()
        except:
            pass
        cv2.destroyAllWindows()
        print("Camera test stopped.")