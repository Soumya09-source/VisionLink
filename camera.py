import cv2
import threading


class CameraStream:
    """Continuously captures frames from a webcam in a background thread."""

    def __init__(self, index: int = 0):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {index}.")

        self._frame = None
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Read frames continuously until stop() is called."""
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                print("Warning: failed to read frame from camera.")
                continue
            with self._lock:
                self._frame = frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self):
        """Return the most recent frame, or None if none captured yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()


# ----------------------------------------------------------------------
# Quick test — press 'q' to quit
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting camera stream … press 'q' to quit.")

    try:
        stream = CameraStream(index=0)
    except RuntimeError as e:
        print(e)
        raise SystemExit(1)

    while True:
        frame = stream.get_frame()

        if frame is not None:
            cv2.imshow("Camera Feed", frame)

        # waitKey must be called even when frame is None so the window stays responsive
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.stop()
    cv2.destroyAllWindows()
    print("Camera stream stopped.")