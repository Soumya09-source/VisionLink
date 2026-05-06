import cv2
from ultralytics import YOLO


class ObjectDetector:
    """Real-time object detector powered by YOLOv8n."""

    # Visual style
    BOX_COLOR   = (0, 255, 0)   # green
    TEXT_COLOR  = (0, 0, 0)     # black (on label background)
    FONT        = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE  = 0.55
    THICKNESS   = 2

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.4):
        """
        Args:
            model_name:  YOLOv8 weights file (downloaded automatically on first run).
            confidence:  Minimum confidence threshold to keep a detection.
        """
        self.confidence = confidence
        self.model = YOLO(model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_position_label(x1: int, x2: int, frame_width: int) -> str:
        """Converts a bounding box into a horizontal direction label."""
        center_x = (x1 + x2) / 2.0
        ratio = center_x / frame_width
        
        if ratio < 0.33:
            return "left"
        elif ratio < 0.66:
            return "center"
        else:
            return "right"

    def detect(self, frame):
        """
        Run inference on *frame*, draw annotations in-place, and return detections.

        Args:
            frame: BGR image as a NumPy array (as returned by cv2.VideoCapture.read).

        Returns:
            List of dicts, each containing:
                {
                    "class_name": str,
                    "confidence": float,   # 0.0 – 1.0
                    "box": (x1, y1, x2, y2)  # pixel coords (ints)
                }
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        frame_width = frame.shape[1]

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence:
                continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            class_name = self.model.names[int(box.cls[0])]
            direction = self.get_position_label(x1, x2, frame_width)

            detections.append({
                "class_name": class_name,
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "direction": direction,
            })

            self._draw(frame, class_name, direction, conf, x1, y1, x2, y2)

        return detections

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _draw(self, frame, class_name: str, direction: str, conf: float,
              x1: int, y1: int, x2: int, y2: int) -> None:
        """Draw a bounding box and label on *frame* (in-place)."""
        # Optional: Color coding by direction
        if direction == "left":
            color = (0, 165, 255)  # Orange in BGR
        elif direction == "right":
            color = (255, 0, 0)    # Blue in BGR
        else:
            color = (0, 255, 0)    # Green in BGR

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.THICKNESS)

        # Combine name + direction into one label
        label = f"{class_name} - {direction} {conf:.0%}"
        
        (tw, th), baseline = cv2.getTextSize(label, self.FONT, self.FONT_SCALE, self.THICKNESS)
        
        # Edge case: If the box is too close to the top, draw label inside the box
        if y1 - th - baseline - 4 < 0:
            bg_y1 = y1
            bg_y2 = y1 + th + baseline + 4
            text_y = y1 + th + 2
        else:
            bg_y1 = y1 - th - baseline - 4
            bg_y2 = y1
            text_y = y1 - baseline - 2
            
        # Label background
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 4, bg_y2), color, cv2.FILLED)

        # Label text
        cv2.putText(frame, label, (x1 + 2, text_y),
                    self.FONT, self.FONT_SCALE, self.TEXT_COLOR, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------
# Quick test — press 'q' to quit
# ----------------------------------------------------------------------

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Error: could not open webcam at index 0.")

    detector = ObjectDetector()
    print("Detector ready. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Warning: dropped frame.")
            continue

        detections = detector.detect(frame)

        # Print detections to console (optional)
        for d in detections:
            print(f"  {d['class_name']:20s} {d['confidence']:.0%}  box={d['box']}")

        cv2.imshow("YOLOv8 Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")