import cv2
import os
import numpy as np
import pickle
from ultralytics import YOLO

yolo_model = YOLO(r"yolov8n.pt")  # lightweight model Used to detect faces (or persons) inside images

dataset_path = "dataset"
faces = []              # list
labels = []             # list
label_map = {}          # number of people
current_label = 0       # initialise with zero

for person in os.listdir(dataset_path):         # returns all the file inside this dataset...
    person_folder = os.path.join(dataset_path, person)   # dataset k andr person naam ka folder

    if not os.path.isdir(person_folder):  # os.path.isdir() → checks: is it a folder (person)?  or a file (ignore)?
        continue

    label_map[current_label] = person

    for image_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, image_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        results = yolo_model(img)  # Now results contains detections from YOLO

        for result in results:
            boxes = result.boxes  # boxes is collection of objects.....Each box = one detected object (face/person/etc.)

            if boxes is None:    # if no person detected then continue.
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                face = img[y1:y2, x1:x2] # (x1,y1 --> top left x2,y2 --> bottom right)

                if face.size == 0:
                    continue

                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                gray_face = cv2.resize(gray_face, (200, 200))

                faces.append(gray_face)
                labels.append(current_label)

    current_label += 1

model = cv2.face.LBPHFaceRecognizer_create()   
model.train(faces, np.array(labels))           

model.save("face_model.yml")

with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Training complete!")
print("Total people:", len(label_map))
print("Total faces:", len(faces))


# epoche time is 300

cv2.face.LBPHFaceRecognizer_create()