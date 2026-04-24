# 🎯 Face Recognition Attendance System (YOLOv8)

An AI-powered attendance system that uses **face recognition** and **YOLOv8** for real-time detection, identification, and automated attendance logging.

---

## 🚀 Features

* 🎥 Real-time face detection using YOLOv8
* 🧠 Face recognition using trained model (LBPH / embeddings-based)
* 📝 Automatic attendance marking with timestamp
* ⚡ Fast and efficient processing using OpenCV
* 📂 Organized and modular code structure

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** OpenCV, NumPy, Pandas
* **Model:** YOLOv8 (Ultralytics)
* **Other:** Pickle (for label encoding)

---

## 📂 Project Structure

```
face-recognition-attendance/
│ yolov8n.pt
├── train.py
├── recognize.py
├── attendance.py
│── dataset/            # (not included)
│── attendance.csv      # auto-generated
│── requirements.txt
│── README.md
│── .gitignore
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Train the model

```
python src/train.py
```

### Step 2: Start recognition system

```
python src/recognize.py
```

### Step 3: Attendance logging

* Attendance is automatically stored in `attendance.csv`

---

## 🧠 Working Pipeline

```
Face Detection (YOLOv8)
        ↓
Face Extraction
        ↓
Feature Recognition (LBPH / trained model)
        ↓
Match with database
        ↓
Mark Attendance (CSV with timestamp)
```

---

## 📸 Output

* Real-time webcam detection
* Recognized faces labeled with names
* Attendance recorded with date & time

*(Add screenshots here for better presentation)*

---

## ⚠️ Notes

* Dataset is not included due to size/privacy
* You can create your own dataset using webcam
* Ensure proper lighting for better accuracy

---

## 📦 Requirements

Example `requirements.txt`:

```
opencv-python
numpy
pandas
ultralytics
pickle-mixin
```

---

## 💡 Future Improvements

* Add GUI (Tkinter / Web dashboard)
* Improve accuracy with deep learning embeddings (FaceNet / Dlib)
* Cloud-based attendance storage
* Multi-face tracking optimization

---

## 👨‍💻 Author

Developed by *Diamond*
https://github.com/tanejadiamond-cpu

---
