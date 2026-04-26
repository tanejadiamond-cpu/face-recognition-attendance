import cv2
import pandas as pd
from datetime import datetime
import os
import pickle
from ultralytics import YOLO

# Load LBPH face recognizer
model = cv2.face.LBPHFaceRecognizer_create()  # create empty model
model.read("face_model.yml")                  # load trainned data

with open("labels.pkl", "rb") as f:
    label_map = pickle.load(f)

# Load YOLOv8n-face model
# Downloads automatically on first run; or place yolov8n-face.pt in the project folder
yolo_model = YOLO(r"yolov8n.pt")

attendance_file = "attendance.csv"

def ensure_attendance_file():
    if not os.path.exists(attendance_file) or os.path.getsize(attendance_file) == 0:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(attendance_file, index=False)

def mark_attendance(name):
    ensure_attendance_file()

    df = pd.read_csv(attendance_file)
    today = datetime.now().strftime("%Y-%m-%d")

    if not ((df["Name"] == name) & (df["Date"] == today)).any():
        now = datetime.now()
        new_row = {"Name": name, "Date": today, "Time": now.strftime("%H:%M:%S")}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print("Attendance marked for", name)

cap = cv2.VideoCapture(0)

print("Starting Face Attendance System... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = yolo_model(frame, verbose=False, conf=0.5)

    for result in results:
        for box in result.boxes:
            # Bounding box in pixel coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Clamp coordinates to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face_img = gray[y1:y2, x1:x2]

            # Skip tiny / empty crops
            if face_img.size == 0:
                continue

            # --- LBPH recognition ---
            label, confidence = model.predict(face_img)

            name = "Unknown"
            if confidence < 60: 
                name = label_map[label]
                mark_attendance(name)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} ({confidence:.1f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

'''
Load trained face recognition model
Load label → name mapping
Start webcam
Detect faces in real-time
Recognize face using LBPH
If recognized → mark attendance (once per day)
Show live video with names
'''
'''
“My pipeline uses detection first (YOLO) to locate faces, and then LBPH as a classifier to recognize identities.”

My current implementation uses LBPH and a pretrained YOLO model yolov8n-face which is of 300 epochs .
I am using a classifier model. After detecting faces, I use an LBPH-based classifier to assign each face to a known identity.”
'''


