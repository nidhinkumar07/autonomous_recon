import streamlit as st
from tracker import ObjectTracker
from detection import load_yolo_model

# Available YOLO model variants for the model selector
YOLO_MODELS = {
    "YOLOv8n — Fastest":  "yolov8n.pt",
    "YOLOv8s — Fast":     "yolov8s.pt",
    "YOLOv8m — Balanced": "yolov8m.pt",
    "YOLOv8l — Accurate": "yolov8l.pt",
    "YOLOv8x — Best":     "yolov8x.pt",
}


def init_session_state():
    """Initialise all session state variables exactly once per browser session."""

    defaults = {
        'show_persons':            True,
        'show_vehicles':           True,
        'show_animals':            False,
        'show_everything':         False,
        'confidence_threshold':    0.5,
        'frame_skip':              2,
        'box_thickness':           2,
        'show_labels':             True,
        'show_confidence':         True,
        'show_ids':                True,
        'selected_model_label':    "YOLOv8n — Fastest",
        'webcam_width':            640,
        'webcam_height':           480,
        'webcam_active':           False,
        'video_processing':        False,
        'detection_counter':       0,
        'active_tab':              "webcam",
        'video_path':              None,
        'uploaded_file':           None,
        'webcam_stats':            {},
        'video_stats':             {},
        'webcam_start_time':       0.0,
        'webcam_frame_count':      0,
        'webcam_detected_objects': 0,
        'webcam_processing_time':  0.0,
        'video_start_time':        0.0,
        'video_frame_count':       0,
        'video_detected_objects':  0,
        'video_processing_time':   0.0,
        'model_ready':             False,
        # Flag set to True when ultralytics is confirmed missing
        'model_missing':           False,
        'model_error':             None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if 'object_tracker' not in st.session_state:
        st.session_state.object_tracker = ObjectTracker()

    # Pre-load real YOLO model at startup.
    # @st.cache_resource means this runs only once per process — not on every rerun.
    if 'current_model' not in st.session_state:
        model_file = YOLO_MODELS[st.session_state.selected_model_label]
        model, err = load_yolo_model(model_file)
        st.session_state.current_model = model
        if model is not None:
            st.session_state.model_ready   = True
            st.session_state.model_missing = False
            st.session_state.model_error   = None
        else:
            st.session_state.model_ready   = False
            st.session_state.model_missing = True
            st.session_state.model_error   = err  # real error message shown in UI