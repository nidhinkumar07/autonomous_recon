import cv2
import numpy as np
import streamlit as st
from tracker import COCO_CLASSES

# ── Pre-built frozensets for O(1) class matching ───────────────────────────
_VEHICLE_SET = frozenset(['car', 'truck', 'bus', 'motorcycle', 'bicycle'])
_ANIMAL_SET  = frozenset(['dog', 'cat', 'bird', 'horse', 'sheep', 'cow'])

# ── Colour cache — resolved once, reused every frame ──────────────────────
_CLASS_COLORS: dict = {}


def _get_class_color(class_name: str):
    """Return (box_color, text_color) for a class name. Result is memoised."""
    cn = class_name.lower()
    if cn in _CLASS_COLORS:
        return _CLASS_COLORS[cn]
    if 'person' in cn:
        result = ((0, 255, 0),   (0, 255, 0))
    elif any(v in cn for v in _VEHICLE_SET):
        result = ((255, 0, 0),   (255, 200, 0))
    elif any(a in cn for a in _ANIMAL_SET):
        result = ((0, 0, 255),   (255, 255, 0))
    else:
        result = ((255, 255, 0), (0, 0, 0))
    _CLASS_COLORS[cn] = result
    return result


# ── Model loading ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_name: str = "yolov8n.pt"):
    """
    Load a YOLO model once per process and cache it via @st.cache_resource.
    Runs a warm-up inference pass to eliminate the first-frame JIT delay.

    Search order for the .pt file:
      1. Absolute path (if model_name is already absolute)
      2. Same directory as this script (detection.py)
      3. Current working directory (where `streamlit run` was called from)
      4. Let ultralytics auto-download it from the internet

    Returns (model, error_message).  error_message is None on success.
    """
    import os

    # ── 1. Resolve the actual file path ───────────────────────────────────
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),  # project folder
        os.getcwd(),                                  # cwd where streamlit was run
    ]
    resolved_path = model_name  # default: let ultralytics handle it (triggers auto-download)
    for d in search_dirs:
        candidate = os.path.join(d, model_name)
        if os.path.isfile(candidate):
            resolved_path = candidate
            break

    # ── 2. Check ultralytics is importable ────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, (
            "ultralytics is not installed.\n"
            "Run:  pip install ultralytics"
        )

    # ── 3. Load model ─────────────────────────────────────────────────────
    try:
        model = YOLO(resolved_path)
        # Warm-up pass: eliminates first-frame JIT delay
        dummy = np.zeros((384, 640, 3), dtype=np.uint8)
        model(dummy, verbose=False)
        return model, None
    except Exception as e:
        return None, (
            f"Could not load model '{model_name}'.\n"
            f"Error: {e}\n\n"
            f"Searched in:\n" +
            "\n".join(f"  • {d}" for d in search_dirs) +
            f"\n\nEither place {model_name} in your project folder, "
            f"or let ultralytics download it automatically by ensuring "
            f"you have an internet connection."
        )


# ── Inference resolution ───────────────────────────────────────────────────
# Frames are resized to this before inference (YOLOv8 native stride),
# then bounding boxes are scaled back to the original frame size.
_INFERENCE_WIDTH  = 640
_INFERENCE_HEIGHT = 384


def detect_with_yolo(model, image: np.ndarray) -> list:
    """
    Run real YOLO inference on `image` (H×W×3 RGB uint8 numpy array).

    - No dummy / simulation mode. If the model is None (not installed or
      file not found), returns [] so nothing is drawn on screen.
    - Resizes to _INFERENCE_WIDTH × _INFERENCE_HEIGHT before inference for
      speed, then scales bounding boxes back to original resolution.
    - Batch-converts all boxes/confs/classes to numpy in one call.
    """
    if model is None:
        return []

    height, width = image.shape[:2]

    # Resize for inference speed; scale factors restore boxes to original coords
    scale_x = width  / _INFERENCE_WIDTH
    scale_y = height / _INFERENCE_HEIGHT
    small   = cv2.resize(image, (_INFERENCE_WIDTH, _INFERENCE_HEIGHT),
                         interpolation=cv2.INTER_LINEAR)

    try:
        results = model(small, verbose=False)
    except Exception:
        return []

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        # Batch numpy conversion — much faster than per-box .cpu().numpy() calls
        xyxy   = boxes.xyxy.cpu().numpy()
        confs  = boxes.conf.cpu().numpy()
        clsids = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clsids):
            cname = (COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES)
                     else f"class_{cls_id}")
            detections.append({
                "class":      cname,
                "bbox":       [float(x1 * scale_x), float(y1 * scale_y),
                               float(x2 * scale_x), float(y2 * scale_y)],
                "confidence": float(conf),
            })
    return detections


def filter_detections(detections: list) -> list:
    """
    Keep only detections that pass the confidence threshold AND match
    the user's class-filter toggles (Persons / Vehicles / Animals / All).

    Uses exact set membership (cn in _VEHICLE_SET) — not substring matching —
    so 'bus' never leaks into 'minibus', and 'person' never matches 'spokesperson'.
    """
    if not detections:
        return []

    threshold     = st.session_state.confidence_threshold
    show_all      = st.session_state.show_everything
    show_persons  = st.session_state.show_persons
    show_vehicles = st.session_state.show_vehicles
    show_animals  = st.session_state.show_animals

    filtered = []
    for det in detections:
        if det['confidence'] < threshold:
            continue

        cn = det['class'].lower()
        is_person  = (cn == 'person')
        is_vehicle = (cn in _VEHICLE_SET)
        is_animal  = (cn in _ANIMAL_SET)

        if (show_all
                or (is_person  and show_persons)
                or (is_vehicle and show_vehicles)
                or (is_animal  and show_animals)):
            filtered.append(det)

    return filtered


def draw_detections(frame: np.ndarray, tracked_data: dict,
                    frame_number: int = 0, source: str = "webcam",
                    draw_cfg: dict = None, object_data_map: dict = None):
    """
    Draw bounding boxes and the heads-up counts overlay onto the frame.

    draw_cfg: optional dict with keys show_labels, show_ids, show_confidence,
              box_thickness.  When None, values are read from st.session_state
              (main-thread usage).  Pass a plain dict when calling from a
              background thread that has no Streamlit ScriptRunContext.

    object_data_map: optional pre-fetched tracker.object_data dict.  When None,
                     read from st.session_state.object_tracker.object_data.
    """
    frame_copy = frame.copy()
    height, width = frame_copy.shape[:2]

    tracked_objects = tracked_data.get("active_objects", {})
    current_counts  = tracked_data.get("current_counts", {})
    total_objects   = tracked_data.get("total_objects", 0)

    # Read display flags — from caller-supplied dict (thread-safe) or session_state
    if draw_cfg is not None:
        show_labels     = draw_cfg.get('show_labels',     True)
        show_ids        = draw_cfg.get('show_ids',        True)
        show_confidence = draw_cfg.get('show_confidence', True)
        box_thickness   = draw_cfg.get('box_thickness',   2)
    else:
        show_labels     = st.session_state.show_labels
        show_ids        = st.session_state.show_ids
        show_confidence = st.session_state.show_confidence
        box_thickness   = st.session_state.box_thickness

    if object_data_map is None:
        object_data_map = st.session_state.object_tracker.object_data

    # Frame border + source watermark
    src_text = "Webcam" if source == "webcam" else "Video"
    cv2.rectangle(frame_copy, (0, 0), (width - 1, height - 1), (100, 100, 100), 2)
    cv2.putText(frame_copy, f"Source: {src_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_copy, f"Frame: {frame_number}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    detection_count = 0

    for obj_id, bbox in tracked_objects.items():
        od         = object_data_map.get(obj_id, {})
        class_name = od.get('class', 'unknown')
        avg_conf   = od.get('avg_confidence', 0.0)

        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, width  - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width  - 1))
        y2 = max(0, min(y2, height - 1))
        if x1 >= x2 or y1 >= y2:
            continue

        color, _ = _get_class_color(class_name)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, box_thickness)

        if show_labels:
            label = class_name
            if show_ids:
                label = f"ID:{obj_id.split('-')[-1]} {label}"
            if show_confidence:
                label += f" {avg_conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_copy, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        detection_count += 1

    # Counts overlay — blend only the ROI slice (not the whole frame)
    ov_x = max(0, width - 250)
    ov_y = 100
    ov_w = min(240, width - ov_x)
    ov_h = min(400, 100 + len(current_counts) * 30, height - ov_y)

    if ov_w > 0 and ov_h > 0:
        roi = frame_copy[ov_y:ov_y + ov_h, ov_x:ov_x + ov_w]
        cv2.addWeighted(np.zeros_like(roi), 0.7, roi, 0.3, 0, roi)
        frame_copy[ov_y:ov_y + ov_h, ov_x:ov_x + ov_w] = roi

    cv2.putText(frame_copy, "CURRENT COUNTS",
                (ov_x + 10, ov_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_copy, f"Total: {total_objects}",
                (ov_x + 10, ov_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    y_off = 90
    for i, (cn, cnt) in enumerate(sorted(current_counts.items())):
        if i >= 10:
            cv2.putText(frame_copy, f"...+{len(current_counts) - 10} more",
                        (ov_x + 10, ov_y + y_off + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            break
        dot, _ = _get_class_color(cn)
        cv2.circle(frame_copy, (ov_x + 10, ov_y + y_off), 5, dot, -1)
        lbl = f"{cn.capitalize()}: {cnt}" if cnt == 1 else f"{cn.capitalize()}s: {cnt}"
        cv2.putText(frame_copy, lbl, (ov_x + 25, ov_y + y_off + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_off += 30

    cv2.putText(frame_copy, f"Detections: {detection_count}", (width - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame_copy, f"Tracked: {len(tracked_objects)}", (width - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame_copy, detection_count, current_counts