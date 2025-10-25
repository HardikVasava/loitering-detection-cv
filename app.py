import cv2
import time
import torch
from collections import deque
from ultralytics import YOLO

# ---------------- CONFIGURATION ---------------- #
VIDEO_PATH = "data\Crowd.mp4"
MODEL_PATH = "yolov8s.pt"
LOITERING_THRESHOLD = 10
ZONE = (100, 100, 800, 800)
USE_CUDA = torch.cuda.is_available()
# ------------------------------------------------ #

# ------------------ MODEL INITIALIZATION ------------------ #
device = "cuda" if USE_CUDA else "cpu"
print(f"[INFO] Using device: {device.upper()}")
model = YOLO(MODEL_PATH)
model.to(device)
# ------------------------------------------------------------ #

# --------------- LOITERING DETECTION FUNCTION --------------- #
person_positions = {}

def detect_loitering(person_id, position):
    current_time = time.time()

    if person_id not in person_positions:
        person_positions[person_id] = [deque(maxlen=30), current_time]

    history, last_time = person_positions[person_id]
    history.append(position)

    x, y = position
    if ZONE[0] < x < ZONE[2] and ZONE[1] < y < ZONE[3]:
        if current_time - last_time > LOITERING_THRESHOLD:
            print(f"[⚠️ ALERT] Person {person_id} is loitering in the restricted area!")
            person_positions[person_id][1] = current_time
# ------------------------------------------------------------ #

# ------------------ MAIN SURVEILLANCE LOOP ------------------ #
def run_surveillance(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video stream ended.")
            break

        results = model.track(frame, tracker="bytetrack.yaml", persist=True, device=device)

        # Draw restricted zone on the frame
        cv2.rectangle(frame, (ZONE[0], ZONE[1]), (ZONE[2], ZONE[3]), (0, 0, 255), 2)
        cv2.putText(frame, "Restricted Zone", (ZONE[0], ZONE[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, person_id, cls_id in zip(boxes, ids, classes):
                if cls_id != 0:
                    continue
                x, y, w, h = box
                detect_loitering(person_id, (x, y))

        annotated_frame = results[0].plot()
        cv2.imshow("Human Surveillance (CUDA 12.8)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# ------------------------------------------------------------ #

# ------------------ PROGRAM EXECUTION ------------------ #
if __name__ == "__main__":
    print("[INFO] Starting Human Surveillance System (CUDA 12.8 enabled)...")
    run_surveillance(VIDEO_PATH)
    print("[INFO] Surveillance finished.")
# ----------------------------------------------------------- #
