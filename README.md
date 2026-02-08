# 🔍 YOLOv8 Object Detection

A **production‑ready Streamlit application** for real‑time object detection, tracking, and live counting using **YOLOv8**.
This project supports **Live Webcam** and **Video Upload** modes with **persistent tracking IDs** and **per‑frame object counts** for all **80 COCO classes**.

---

## 🚀 Features

### ✅ Real‑Time Object Detection

- Powered by **YOLOv8 (Ultralytics)**
- Supports all **80 COCO classes** (person, car, dog, bottle, etc.)

### 🎯 Persistent Object Tracking

- Unique IDs like:

  ```
  person-001
  car-002
  dog-001
  ```

- IDs persist across frames
- Objects can temporarily disappear and reappear without losing identity

### 🔢 Live Object Counting (Per Frame)

- Displays **current active objects only**
- Example:

  ```
  Person : 3
  Car    : 1
  Dog    : 2
  ```

- Updates instantly when objects enter or leave the frame

### 📹 Dual Input Modes

- **Live Webcam Detection**
- **Upload & Process Video Files**

### 🧠 Smart Matching Logic

- IoU‑based bounding box matching
- Frame‑gap tolerance for occlusions
- Prevents duplicate IDs for the same object

### 🎨 Clean Visual Overlay

- Bounding boxes with class + ID
- Live object count legend on video feed
- Color‑coded objects (persons, vehicles, animals, others)

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** – UI & dashboard
- **OpenCV** – Video processing
- **YOLOv8 (Ultralytics)** – Object detection
- **NumPy** – Math & geometry

---

## 📂 Project Structure

```bash
.
├── main.py              # Streamlit app (detection + tracking + UI)
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── yolov8n.pt           # YOLOv8 model (auto‑downloaded if missing)
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/nidhinkumar07/autonomous_recon.git
cd autonomous_recon
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, use:

```bash
pip install streamlit opencv-python ultralytics numpy pillow
```

---

## ▶️ Run the Application

```bash
streamlit run main.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 🧪 How It Works (High Level)

1. **YOLOv8** detects objects per frame
2. **ObjectTracker** assigns or matches IDs using IoU
3. Active objects are tracked frame‑by‑frame
4. Live counts are computed **only from visible objects**
5. Streamlit overlays boxes, IDs, and counts in real time

---

## 🧩 Object Tracking Logic

- Each detection is matched to an existing object using **IoU**
- If no match is found → a new ID is created
- Objects are removed only after being missing for multiple frames
- Counts reflect **current frame only**, not historical totals

---

## 📊 Output Examples

- `3 Persons Detected`
- `Live Objects Panel`:

  ```
  Person : 3
  Car    : 1
  Bottle : 2
  ```

---

## 🔄 Reset Behavior

- Tracker resets automatically when:
  - Starting Webcam Detection
  - Starting Video Processing

(IDs remain consistent _within_ each session)

---

## 🧠 Future Enhancements

- Zone‑based entry / exit counting
- Heatmaps & dwell‑time analysis
- DeepSORT / ByteTrack integration
- Export analytics as CSV / JSON
- Multi‑camera support

---
