# 🔍 YOLOv8 Detection Dashboard

A professional, real-time object detection and tracking web application built with **Streamlit** and **YOLOv8**. Supports live webcam feeds and uploaded video files, with persistent multi-object tracking, class filtering, and live statistics.

---

## 📸 Features

- 🎥 **Live Webcam Detection** — Real-time object detection from your webcam
- 📹 **Video File Upload** — Process uploaded video files (MP4, AVI, MOV, MKV, and more)
- 🏷️ **Multi-Object Tracking** — Each detected object gets a unique persistent ID across frames
- 📊 **Live Statistics** — Real-time FPS, detection count, and per-class tracking summaries
- 🎛️ **Sidebar Controls** — Filter by class (Persons, Vehicles, Animals, All), adjust confidence threshold, frame skip, and box thickness
- 🎨 **Color-coded Bounding Boxes** — Green for persons, blue for vehicles, red for animals, yellow for others
- 🖼️ **On-frame Overlay** — Live count overlay rendered directly onto each video frame
- 💅 **Custom CSS Styling** — External `styles.css` for a clean professional SaaS look

---

## 📁 Project Structure

```
├── main.py              # Entry point — page config, title, wires all modules
├── tracker.py           # ObjectTracker class + COCO_CLASSES list
├── detection.py         # YOLO model loading, inference, filtering, drawing
├── ui_components.py     # CSS loader and tracking summary display
├── session_state.py     # All Streamlit session state initialization
├── sidebar.py           # Sidebar settings panel
├── tabs.py              # Tab 1 (Webcam) and Tab 2 (Video Upload) logic
├── styles.css           # External CSS for UI styling
└── README.md            # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/yolov8-detection-dashboard.git
cd yolov8-detection-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run main.py
```

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```
streamlit
opencv-python
numpy
Pillow
ultralytics
```

> **Note:** If YOLOv8 (`ultralytics`) is not installed or the model file is unavailable, the app automatically falls back to **realistic dummy detections** for testing purposes.

---

## 🧩 Module Breakdown

### `main.py`

The entry point. Sets the Streamlit page config (must be the very first Streamlit command), suppresses warnings, loads CSS, renders the title, initializes session state, and calls the sidebar and tab renderers.

### `tracker.py`

Contains:

- `COCO_CLASSES` — list of 80 COCO dataset class names
- `ObjectTracker` — IoU + center-distance based multi-object tracker with:
  - Persistent object IDs across frames
  - Missing-frame tolerance (configurable)
  - Per-class active counts
  - Full tracking history and summary generation

### `detection.py`

Contains:

- `load_yolo_model()` — loads `yolov8n.pt` via Ultralytics, falls back to `None` if unavailable
- `detect_with_yolo(model, image)` — runs inference or generates realistic dummy detections
- `filter_detections(detections)` — filters by class preference and confidence threshold
- `draw_detections(frame, tracked_data, frame_number, source)` — draws bounding boxes, labels, IDs, and the live count overlay onto the frame

### `ui_components.py`

Contains:

- `load_css_from_file(file_path)` — reads and injects `styles.css`; falls back gracefully if not found
- `display_summary()` — renders the comprehensive per-class tracking summary with expandable object details

### `session_state.py`

Contains `init_session_state()` which initializes all `st.session_state` variables including detection preferences, visualization settings, tracker instance, and processing flags.

### `sidebar.py`

Contains `render_sidebar()` which renders the full settings sidebar including detection class checkboxes, visualization toggles, confidence/frame-skip/box-thickness sliders, and the color legend.

### `tabs.py`

Contains `render_tabs()` which renders:

- **Tab 1 — Live Webcam:** start/stop webcam, frame capture loop, detection, tracking, stats display
- **Tab 2 — Upload Video:** file uploader, video preview, start/stop processing, per-frame progress bar, stats display

---

## ⚙️ Configuration Options (via Sidebar)

| Setting              | Description                            | Default |
| -------------------- | -------------------------------------- | ------- |
| Persons              | Detect people                          | ✅ On   |
| Vehicles             | Detect cars, trucks, buses, etc.       | ✅ On   |
| Animals              | Detect dogs, cats, birds, etc.         | ❌ Off  |
| All Classes          | Detect everything                      | ❌ Off  |
| Show Labels          | Show class name on boxes               | ✅ On   |
| Show Confidence      | Show confidence score on boxes         | ✅ On   |
| Show Object IDs      | Show unique tracking ID on boxes       | ✅ On   |
| Confidence Threshold | Minimum confidence to show a detection | 0.5     |
| Frame Skip           | Process every Nth frame                | 2       |
| Box Thickness        | Bounding box line thickness            | 2       |

---

## 🎨 Detection Colors

| Category    | Color     |
| ----------- | --------- |
| 👤 Persons  | 🟢 Green  |
| 🚗 Vehicles | 🔵 Blue   |
| 🐾 Animals  | 🔴 Red    |
| 📦 Other    | 🟡 Yellow |

---

## 🛠️ Supported Video Formats

`MP4`, `AVI`, `MOV`, `MKV`, `FLV`, `MPG`, `MPEG`, `WMV`

---

## 📝 Notes

- The `styles.css` file must be present in the same directory as `main.py` for full styling. A minimal fallback is shown if it is missing.
- Object IDs are **globally persistent** — they do not reset between webcam and video sessions, ensuring unique IDs throughout the app's lifetime.
- The tracker preserves full bounding box and confidence history for every object, even after it becomes inactive.

---

## 📄 License

MIT License. Feel free to use, modify, and distribute.
