import cv2
import time

def test_camera():
    cap = cv2.VideoCapture(0)
    
    # Allow camera sensor to warm up
    time.sleep(2.0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        
        # 1. Always validate frame
        if not ret or frame is None or frame.size == 0:
            print("Warning: Empty or invalid frame received.")
            # Even if frame is invalid, we MUST pump the event loop
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue
            
        # 2. Display frame
        cv2.imshow("Camera Test", frame)
        
        # 3. Always call waitKey (pumps macOS GUI events)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # On macOS, calling waitKey(1) after destroyAllWindows helps fully close the window
    cv2.waitKey(1)

if __name__ == "__main__":
    test_camera()
