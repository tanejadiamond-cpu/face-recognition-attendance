import cv2
import os

name = input("Enter your name: ")
folder = f"dataset/{name}"

if not os.path.exists(folder):
    os.makedirs(folder)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read() # tupple unpacking --> returns two outputs ret for true/false and frame for img reading
    if not ret:
        break

    cv2.imshow("Capture Faces - Press C to capture, Q to quit", frame)
    key = cv2.waitKey(1)

    if key == ord('c'):
        img_path = f"{folder}/{count}.jpg"
        cv2.imwrite(img_path, frame)
        print("Saved:", img_path)
        count = count + 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()