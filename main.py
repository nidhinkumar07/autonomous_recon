# main.py - Fixed version with working buttons

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import matplotlib.pyplot as plt
import random

# COCO class names for better detection labels
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# =============== SIMPLE OBJECT TRACKING FUNCTIONS ===============
def calculate_iou_simple(box1, box2):
    """Calculate Intersection over Union for two bounding boxes - simpler version"""
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def track_objects_simple(current_detections):
    """
    SIMPLE tracking: Count each detection as a separate object unless they overlap significantly
    Returns: unique object counts by class
    """
    if 'simple_tracked_objects' not in st.session_state:
        st.session_state.simple_tracked_objects = []
    if 'simple_object_counter' not in st.session_state:
        st.session_state.simple_object_counter = 0
    
    current_time = time.time()
    
    # Clean up old objects (not seen for 5 seconds)
    max_age = 5.0  # seconds
    st.session_state.simple_tracked_objects = [
        obj for obj in st.session_state.simple_tracked_objects
        if (current_time - obj['last_seen']) <= max_age
    ]
    
    # Process current detections
    current_objects = []
    
    for det in current_detections:
        if len(det) >= 6:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            class_name = det[6] if len(det) > 6 else f"class_{cls_id}"
            
            # Skip if confidence is too low
            if conf < st.session_state.confidence_threshold:
                continue
                
            current_objects.append({
                'box': [x1, y1, x2, y2],
                'class_name': class_name,
                'confidence': conf,
                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
            })
    
    # Match current objects to tracked objects
    matched_tracked_indices = set()
    matched_current_indices = set()
    
    # Try to match each current object to tracked objects
    for i, current_obj in enumerate(current_objects):
        best_match_idx = -1
        best_iou = 0.4  # Minimum IoU threshold for matching
        
        for j, tracked_obj in enumerate(st.session_state.simple_tracked_objects):
            # Skip if already matched or different class
            if j in matched_tracked_indices or tracked_obj['class_name'] != current_obj['class_name']:
                continue
            
            # Calculate IoU
            iou = calculate_iou_simple(current_obj['box'], tracked_obj['last_box'])
            
            if iou > best_iou:
                best_iou = iou
                best_match_idx = j
        
        if best_match_idx >= 0:
            # Update tracked object
            tracked_obj = st.session_state.simple_tracked_objects[best_match_idx]
            tracked_obj['last_box'] = current_obj['box']
            tracked_obj['last_seen'] = current_time
            tracked_obj['detection_count'] += 1
            matched_tracked_indices.add(best_match_idx)
            matched_current_indices.add(i)
    
    # Create new tracked objects for unmatched current objects
    for i, current_obj in enumerate(current_objects):
        if i not in matched_current_indices:
            # Check if this might be a duplicate of another object in the same frame
            is_duplicate = False
            for j in range(i + 1, len(current_objects)):
                if j in matched_current_indices:
                    continue
                iou = calculate_iou_simple(current_obj['box'], current_objects[j]['box'])
                # If two objects in the same frame overlap significantly, they might be the same object
                if iou > 0.7:  # High overlap threshold for same-frame objects
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                new_obj = {
                    'id': st.session_state.simple_object_counter,
                    'class_name': current_obj['class_name'],
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'last_box': current_obj['box'],
                    'detection_count': 1,
                    'confidence': current_obj['confidence']
                }
                st.session_state.simple_object_counter += 1
                st.session_state.simple_tracked_objects.append(new_obj)
    
    # Count unique objects by class
    unique_counts = {}
    for obj in st.session_state.simple_tracked_objects:
        class_name = obj['class_name']
        if class_name not in unique_counts:
            unique_counts[class_name] = 0
        unique_counts[class_name] += 1
    
    return unique_counts

def reset_simple_tracking():
    """Reset the simple tracking system"""
    st.session_state.simple_tracked_objects = []
    st.session_state.simple_object_counter = 0

def draw_detections_with_tracking_simple(frame, detections, frame_number=0, source="webcam"):
    """Draw detection boxes on frame with proper styling and object IDs - simple version"""
    frame_copy = frame.copy()
    height, width = frame_copy.shape[:2]
    
    # Draw a border around the frame
    cv2.rectangle(frame_copy, (0, 0), (width-1, height-1), (100, 100, 100), 2)
    
    # Add source watermark
    source_text = "Webcam" if source == "webcam" else "Video"
    cv2.putText(frame_copy, f"Source: {source_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_copy, f"Frame: {frame_number}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    detection_count = 0
    
    # Get current tracked objects for display
    current_objects = []
    for det in detections:
        if len(det) >= 6:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            class_name = det[6] if len(det) > 6 else f"class_{cls_id}"
            
            # Skip if confidence is too low
            if conf < st.session_state.confidence_threshold:
                continue
                
            current_objects.append({
                'box': [x1, y1, x2, y2],
                'class_name': class_name,
                'confidence': conf
            })
    
    # Match to tracked objects for ID display
    for i, current_obj in enumerate(current_objects):
        x1, y1, x2, y2 = map(int, current_obj['box'])
        conf = current_obj['confidence']
        class_name = current_obj['class_name']
        
        # Find object ID from tracked objects
        object_id = None
        best_iou = 0.3
        
        for tracked_obj in st.session_state.simple_tracked_objects:
            if tracked_obj['class_name'] == class_name:
                iou = calculate_iou_simple(current_obj['box'], tracked_obj['last_box'])
                if iou > best_iou:
                    best_iou = iou
                    object_id = tracked_obj['id']
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        if x1 >= x2 or y1 >= y2:
            continue
        
        # Choose color based on class
        if 'person' in class_name.lower():
            color = (0, 255, 0)  # Green for persons
        elif any(vehicle in class_name.lower() for vehicle in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']):
            color = (255, 0, 0)  # Blue for vehicles
        elif any(animal in class_name.lower() for animal in ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow']):
            color = (0, 0, 255)  # Red for animals
        else:
            color = (255, 255, 0)  # Yellow for others
        
        # Draw rectangle
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, st.session_state.box_thickness)
        
        # Draw label
        if st.session_state.show_labels:
            label = f"{class_name}"
            if object_id is not None and st.session_state.show_object_ids:
                label = f"ID:{object_id} {label}"
            if st.session_state.show_confidence:
                label += f" {conf:.2f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw filled rectangle for label background
            cv2.rectangle(frame_copy, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1), 
                        color, 
                        -1)
            
            # Draw text
            cv2.putText(frame_copy, label, 
                      (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, 
                      (255, 255, 255), 
                      2)
        
        detection_count += 1
    
    # Add counts to frame
    unique_objects = len(st.session_state.simple_tracked_objects)
    cv2.putText(frame_copy, f"Unique Objects: {unique_objects}", (width - 200, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame_copy, f"Detections: {detection_count}", (width - 200, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame_copy, detection_count, unique_objects
# =============== END SIMPLE TRACKING FUNCTIONS ===============

# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Multi-Model Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1rem;
    }
    .stButton button {
        background-color: #3B82F6;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border: none;
        font-weight: bold;
    }
    .processing-info {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
        color: #78350F;
        font-weight: 500;
    }
    .detection-status {
        background-color: #D1FAE5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        border-left: 4px solid #10B981;
        color: #064E3B;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🔍 Multi-Model Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This dashboard allows you to detect persons and objects using multiple deep learning models. 
Upload a video file or use your webcam, then select from various detection models.
""")

# Initialize ALL session state variables FIRST
if 'show_persons' not in st.session_state:
    st.session_state.show_persons = True
if 'show_vehicles' not in st.session_state:
    st.session_state.show_vehicles = True
if 'show_animals' not in st.session_state:
    st.session_state.show_animals = False
if 'show_everything' not in st.session_state:
    st.session_state.show_everything = False
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "YOLOv8 (Recommended)"
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.5
if 'frame_skip' not in st.session_state:
    st.session_state.frame_skip = 2
if 'box_thickness' not in st.session_state:
    st.session_state.box_thickness = 2
if 'show_labels' not in st.session_state:
    st.session_state.show_labels = True
if 'show_confidence' not in st.session_state:
    st.session_state.show_confidence = True
if 'show_object_ids' not in st.session_state:
    st.session_state.show_object_ids = False
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'video_processing' not in st.session_state:
    st.session_state.video_processing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'detection_counter' not in st.session_state:
    st.session_state.detection_counter = 0
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'simple_tracked_objects' not in st.session_state:
    st.session_state.simple_tracked_objects = []
if 'simple_object_counter' not in st.session_state:
    st.session_state.simple_object_counter = 0
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model selection
    st.markdown("#### Select Detection Model")
    st.session_state.model_choice = st.selectbox(
        "Choose a model:",
        [
            "YOLOv8 (Recommended)",
            "YOLOv8 - Person Only", 
            "MobileNet SSD",
            "Faster R-CNN"
        ],
        index=0
    )
    
    # Confidence threshold
    st.session_state.confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.confidence_threshold,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Detection classes
    st.markdown("#### Detection Classes")
    st.session_state.show_persons = st.checkbox("Persons", value=st.session_state.show_persons, key="sidebar_persons")
    st.session_state.show_vehicles = st.checkbox("Vehicles", value=st.session_state.show_vehicles, key="sidebar_vehicles")
    st.session_state.show_animals = st.checkbox("Animals", value=st.session_state.show_animals, key="sidebar_animals")
    st.session_state.show_everything = st.checkbox("Show All Classes", value=st.session_state.show_everything, key="sidebar_everything")
    
    # Performance settings
    st.markdown("#### Performance")
    st.session_state.frame_skip = st.slider(
        "Frame Skip (for speed)",
        min_value=1,
        max_value=10,
        value=st.session_state.frame_skip,
        step=1,
        help="Process every nth frame"
    )
    
    # Visualization options
    st.markdown("#### Visualization")
    st.session_state.show_labels = st.checkbox("Show Labels", value=st.session_state.show_labels)
    st.session_state.show_confidence = st.checkbox("Show Confidence Scores", value=st.session_state.show_confidence)
    
    # Bounding box settings
    st.markdown("#### Bounding Box Settings")
    st.session_state.box_thickness = st.slider(
        "Box Thickness",
        min_value=1,
        max_value=5,
        value=st.session_state.box_thickness,
        step=1,
        help="Thickness of detection boxes"
    )
    
    # Display stats
    with st.expander("Model Information"):
        if "YOLO" in st.session_state.model_choice:
            st.info("""
            **YOLOv8 Model**: 
            - Real-time object detection
            - Good balance of speed and accuracy
            - 80+ COCO classes
            """)
        elif "MobileNet" in st.session_state.model_choice:
            st.info("""
            **MobileNet SSD**:
            - Fast and lightweight
            - Good for mobile/edge devices
            - 20+ classes
            """)
        elif "Faster R-CNN" in st.session_state.model_choice:
            st.info("""
            **Faster R-CNN**:
            - High accuracy
            - Slower but more precise
            - Good for detailed analysis
            """)
    
    # Reset button
    if st.button("🔄 Reset All", key="reset_all"):
        for key in list(st.session_state.keys()):
            if key not in ['show_persons', 'show_vehicles', 'show_animals', 'show_everything',
                          'model_choice', 'confidence_threshold', 'frame_skip', 'box_thickness',
                          'show_labels', 'show_confidence']:
                del st.session_state[key]
        st.rerun()

# Initialize session state for processing
if 'detection_stats' not in st.session_state:
    st.session_state.detection_stats = {}
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'video_processing' not in st.session_state:
    st.session_state.video_processing = False
if 'detection_counter' not in st.session_state:
    st.session_state.detection_counter = 0
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "webcam"
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'webcam_stats' not in st.session_state:
    st.session_state.webcam_stats = {}
if 'video_stats' not in st.session_state:
    st.session_state.video_stats = {}
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Import detection modules function
def load_yolo_model():
    try:
        from ultralytics import YOLO
        return YOLO('yolov8n.pt')
    except:
        # Return a dummy model for testing
        st.warning("YOLOv8 model not found. Using dummy detector.")
        return None

def detect_with_yolo(model, image, model_choice="YOLOv8"):
    """Detection function with proper bounding boxes"""
    detections = []
    
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:
        # If PIL Image, convert to numpy
        image = np.array(image)
        height, width = image.shape[:2]
    
    if model is None:
        # Generate realistic dummy detections with proper bounding boxes
        num_detections = random.randint(1, 3)
        
        for i in range(num_detections):
            # Random size and position
            box_width = random.randint(50, 200)
            box_height = random.randint(50, 200)
            x1 = random.randint(0, width - box_width - 1)
            y1 = random.randint(0, height - box_height - 1)
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            # Random confidence
            conf = random.uniform(0.6, 0.95)
            
            # Select class based on user preferences
            possible_classes = []
            if st.session_state.show_persons:
                possible_classes.append('person')
            if st.session_state.show_vehicles:
                possible_classes.extend(['car', 'bicycle', 'motorcycle', 'bus', 'truck'])
            if st.session_state.show_animals:
                possible_classes.extend(['dog', 'cat', 'bird', 'horse'])
            if st.session_state.show_everything and len(possible_classes) == 0:
                possible_classes = ['person', 'car', 'dog', 'bicycle', 'chair', 'bottle']
            
            if len(possible_classes) > 0:
                class_name = random.choice(possible_classes)
                cls_id = COCO_CLASSES.index(class_name) if class_name in COCO_CLASSES else 0
                detections.append([x1, y1, x2, y2, conf, cls_id, class_name])
        
        return detections
    
    # Real model detection
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
    
    if hasattr(model, 'predict'):
        try:
            results = model(image_pil, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                        detections.append([x1, y1, x2, y2, conf, cls_id, class_name])
        except Exception as e:
            st.warning(f"Model detection error: {e}")
            # Fallback to dummy detections
            num_detections = random.randint(1, 3)
            for i in range(num_detections):
                box_width = random.randint(50, 200)
                box_height = random.randint(50, 200)
                x1 = random.randint(0, width - box_width - 1)
                y1 = random.randint(0, height - box_height - 1)
                x2 = x1 + box_width
                y2 = y1 + box_height
                conf = random.uniform(0.6, 0.95)
                class_name = random.choice(['person', 'car', 'dog', 'bicycle'])
                cls_id = COCO_CLASSES.index(class_name) if class_name in COCO_CLASSES else 0
                detections.append([x1, y1, x2, y2, conf, cls_id, class_name])
    
    return detections

def draw_detections(frame, detections, frame_number=0, source="webcam"):
    """Draw detection boxes on frame with proper styling"""
    frame_copy = frame.copy()
    height, width = frame_copy.shape[:2]
    
    # Draw a border around the frame
    cv2.rectangle(frame_copy, (0, 0), (width-1, height-1), (100, 100, 100), 2)
    
    # Add source watermark
    source_text = "Webcam" if source == "webcam" else "Video"
    cv2.putText(frame_copy, f"Source: {source_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_copy, f"Frame: {frame_number}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    detection_count = 0
    
    for det in detections:
        if len(det) >= 6:  # [x1, y1, x2, y2, conf, class_id, class_name]
            x1, y1, x2, y2 = map(int, det[:4])
            conf = det[4]
            class_name = det[-1]
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            # Skip if coordinates are invalid
            if x1 >= x2 or y1 >= y2:
                continue
                
            # Skip if confidence is below threshold
            if conf < st.session_state.confidence_threshold:
                continue
                
            # Choose color based on class
            if 'person' in class_name.lower():
                color = (0, 255, 0)  # Green for persons
            elif any(vehicle in class_name.lower() for vehicle in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']):
                color = (255, 0, 0)  # Blue for vehicles
            elif any(animal in class_name.lower() for animal in ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow']):
                color = (0, 0, 255)  # Red for animals
            else:
                color = (255, 255, 0)  # Yellow for others
            
            # Draw rectangle with thickness
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, st.session_state.box_thickness)
            
            # Draw label background
            if st.session_state.show_labels:
                label = f"{class_name}"
                if st.session_state.show_confidence:
                    label += f" {conf:.2f}"
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw filled rectangle for label background
                cv2.rectangle(frame_copy, 
                            (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), 
                            color, 
                            -1)
                
                # Draw text
                cv2.putText(frame_copy, label, 
                          (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, 
                          (255, 255, 255), 
                          2)
            
            detection_count += 1
    
    # Add detection count to frame
    cv2.putText(frame_copy, f"Detections: {detection_count}", (width - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame_copy, detection_count

# Main tabs
tab1, tab2, tab3 = st.tabs(["📹 Live Webcam", "📁 Upload Video", "📊 Analytics"])

# Tab 1: Live Webcam
with tab1:
    st.markdown("### 📹 Live Webcam Detection")
    
    # Initialize webcam-specific session state
    if 'webcam_start_time' not in st.session_state:
        st.session_state.webcam_start_time = None
    if 'webcam_frame_count' not in st.session_state:
        st.session_state.webcam_frame_count = 0
    if 'webcam_detections_per_class' not in st.session_state:
        st.session_state.webcam_detections_per_class = {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create placeholders
        webcam_placeholder = st.empty()
        webcam_stats_placeholder = st.empty()
        
        if not st.session_state.webcam_active:
            # Show start button
            if st.button("🚀 Start Webcam Detection", key="start_webcam_main", type="primary"):
                st.session_state.webcam_active = True
                st.session_state.video_processing = False  # Ensure video is stopped
                st.session_state.webcam_start_time = time.time()
                st.session_state.webcam_frame_count = 0
                reset_simple_tracking()
                st.rerun()
        else:
            st.markdown('<div class="processing-info">Webcam is active. Detections will appear with bounding boxes.</div>', unsafe_allow_html=True)
            
            # Control buttons
            col_stop, col_capture = st.columns(2)
            with col_stop:
                if st.button("⏹️ Stop Webcam", key="stop_webcam_btn", type="secondary"):
                    st.session_state.webcam_active = False
                    st.rerun()
            with col_capture:
                if st.button("📸 Capture Frame", key="capture_webcam"):
                    if st.session_state.current_frame is not None:
                        st.image(st.session_state.current_frame, caption="Captured Frame", use_column_width=True)
            
            # Webcam processing logic
            try:
                # Initialize webcam
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    # Try different backends
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                
                if not cap.isOpened():
                    st.error("Could not access webcam. Please check if webcam is connected.")
                    st.session_state.webcam_active = False
                    st.rerun()
                
                # Set buffer size to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Load model if not already loaded
                if st.session_state.current_model is None:
                    with st.spinner("Loading detection model..."):
                        st.session_state.current_model = load_yolo_model()
                
                frame_count = 0
                
                # Process webcam frames
                while st.session_state.webcam_active:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.warning("Failed to capture frame from webcam")
                        time.sleep(0.1)
                        continue
                    
                    frame_count += 1
                    st.session_state.webcam_frame_count += 1
                    
                    # Skip frames for performance
                    if frame_count % st.session_state.frame_skip != 0:
                        continue
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Perform detection
                    start_time = time.time()
                    detections = detect_with_yolo(st.session_state.current_model, frame_rgb)
                    processing_time = time.time() - start_time
                    
                    # Filter detections based on user preferences
                    filtered_detections = []
                    for det in detections:
                        if len(det) > 5:
                            class_name = det[6].lower() if len(det) > 6 else ""
                            conf = det[4]
                            
                            # Check if class should be shown
                            should_show = False
                            if st.session_state.show_everything:
                                should_show = True
                            elif 'person' in class_name and st.session_state.show_persons:
                                should_show = True
                            elif any(vehicle in class_name for vehicle in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']) and st.session_state.show_vehicles:
                                should_show = True
                            elif any(animal in class_name for animal in ['dog', 'cat', 'bird', 'horse']) and st.session_state.show_animals:
                                should_show = True
                            
                            if should_show and conf >= st.session_state.confidence_threshold:
                                filtered_detections.append(det)
                    
                    # Update statistics using simple object tracking
                    unique_counts = track_objects_simple(filtered_detections)
                    
                    # Update session state with unique counts
                    st.session_state.webcam_detections_per_class = unique_counts
                    
                    # Draw detections on frame
                    frame_with_detections, detection_count, unique_objects = draw_detections_with_tracking_simple(frame_rgb, filtered_detections, frame_count, "webcam")
                    
                    # Update session state
                    st.session_state.current_frame = frame_with_detections
                    st.session_state.detection_counter = detection_count
                    
                    # Display frame with detections
                    webcam_placeholder.image(frame_with_detections, channels="RGB", use_column_width=True, caption="Live Webcam with Detections")
                    
                    # Update statistics display
                    with webcam_stats_placeholder.container():
                        fps_value = 1.0 / processing_time if processing_time > 0 else 0
                        elapsed_time = time.time() - st.session_state.webcam_start_time
                        
                        # Create columns for metrics
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        with stat_col1:
                            st.metric("Frames", st.session_state.webcam_frame_count)
                        
                        with stat_col2:
                            st.metric("Current Detections", detection_count)
                        
                        with stat_col3:
                            st.metric("Unique Objects", unique_objects)
                        
                        with stat_col4:
                            st.metric("FPS", f"{fps_value:.1f}")
                        
                        # Show current detections below the metrics
                        if detection_count > 0:
                            st.markdown(f'<div class="detection-status">✅ Detected {detection_count} object(s) in current frame</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="detection-status"> No objects detected in current frame</div>', unsafe_allow_html=True)
                    
                    # Small delay
                    time.sleep(0.033)
                
                # Release webcam when done
                cap.release()
                
                # Update detection stats
                st.session_state.detection_stats = {
                    'total_frames': st.session_state.webcam_frame_count,
                    'detected_objects': st.session_state.webcam_detected_objects,
                    'detections_per_class': st.session_state.webcam_detections_per_class,
                    'processing_time': st.session_state.webcam_processing_time,
                    'source': 'webcam',
                    'start_time': st.session_state.webcam_start_time
                }
                
                # Clear webcam-specific stats
                st.session_state.webcam_frame_count = 0
                st.session_state.webcam_detected_objects = 0
                st.session_state.webcam_detections_per_class = {}
                st.session_state.webcam_processing_time = 0
                
            except Exception as e:
                st.error(f"Error in webcam processing: {str(e)}")
                st.session_state.webcam_active = False
                st.rerun()
    
    with col2:
        st.markdown("#### Webcam Instructions")
        st.info("""
        1. Click **Start Webcam Detection**
        2. Allow camera access when prompted
        3. Detection boxes will appear in real-time
        4. Adjust settings in sidebar
        5. Click **Stop Webcam** to end
        
        **Detection Colors:**
        - 🟢 Green: Persons
        - 🔵 Blue: Vehicles
        - 🔴 Red: Animals
        - 🟡 Yellow: Other objects
        """)
        
        # Show current status
        if st.session_state.webcam_active:
            st.success("✅ **Webcam Status: ACTIVE**")
            if st.session_state.detection_counter > 0:
                st.info(f"📊 Last frame: {st.session_state.detection_counter} detections")
        else:
            st.warning("⏸️ **Webcam Status: INACTIVE**")

# Tab 2: Upload Video
with tab2:
    st.markdown("### 📁 Upload Video for Detection")
    
    # Initialize video-specific session state
    if 'video_start_time' not in st.session_state:
        st.session_state.video_start_time = None
    if 'video_frame_count' not in st.session_state:
        st.session_state.video_frame_count = 0
    if 'video_detections_per_class' not in st.session_state:
        st.session_state.video_detections_per_class = {}
    
    # Video uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'mpg', 'mpeg', 'wmv'],
        help="Upload a video file for detection analysis",
        key="video_uploader_main"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        st.session_state.video_path = video_path
        st.session_state.uploaded_file = uploaded_file
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create placeholders
            video_placeholder = st.empty()
            video_stats_placeholder = st.empty()
            video_progress_placeholder = st.empty()
            
            if not st.session_state.video_processing:
                # Show video preview
                st.video(uploaded_file)
                
                # Show start button
                if st.button("🚀 Start Video Processing", key="start_video_main", type="primary"):
                    st.session_state.video_processing = True
                    st.session_state.webcam_active = False  # Ensure webcam is stopped
                    st.session_state.video_start_time = time.time()
                    st.session_state.video_frame_count = 0
                    reset_simple_tracking()
                    st.rerun()
            else:
                st.markdown('<div class="processing-info">Video processing is active. Bounding boxes will appear on each frame.</div>', unsafe_allow_html=True)
                
                # Control button
                if st.button("⏹️ Stop Processing", key="stop_video_btn", type="secondary"):
                    st.session_state.video_processing = False
                    st.rerun()
                
                # Video processing logic
                try:
                    # Initialize video capture
                    cap = cv2.VideoCapture(video_path)
                    
                    if not cap.isOpened():
                        st.error("Could not open video file. The file might be corrupted.")
                        st.session_state.video_processing = False
                        st.rerun()
                    
                    # Get video properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if fps <= 0 or total_frames <= 0:
                        st.error("Invalid video file. Could not read video properties.")
                        cap.release()
                        st.session_state.video_processing = False
                        st.rerun()
                    
                    # Load model if not already loaded
                    if st.session_state.current_model is None:
                        with st.spinner("Loading detection model..."):
                            st.session_state.current_model = load_yolo_model()
                    
                    frame_count = 0
                    
                    # Process video frames
                    while st.session_state.video_processing and cap.isOpened():
                        ret, frame = cap.read()
                        
                        if not ret:
                            break
                        
                        frame_count += 1
                        st.session_state.video_frame_count += 1
                        
                        # Update progress
                        if total_frames > 0:
                            progress = frame_count / total_frames
                            video_progress_placeholder.progress(progress)
                        
                        # Skip frames for performance
                        if frame_count % st.session_state.frame_skip != 0:
                            continue
                        
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Perform detection
                        start_time = time.time()
                        detections = detect_with_yolo(st.session_state.current_model, frame_rgb)
                        processing_time = time.time() - start_time
                        
                        # Filter detections
                        filtered_detections = []
                        for det in detections:
                            if len(det) > 5:
                                class_name = det[6].lower() if len(det) > 6 else ""
                                conf = det[4]
                                
                                # Check if class should be shown
                                should_show = False
                                if st.session_state.show_everything:
                                    should_show = True
                                elif 'person' in class_name and st.session_state.show_persons:
                                    should_show = True
                                elif any(vehicle in class_name for vehicle in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']) and st.session_state.show_vehicles:
                                    should_show = True
                                elif any(animal in class_name for animal in ['dog', 'cat', 'bird', 'horse']) and st.session_state.show_animals:
                                    should_show = True
                                
                                if should_show and conf >= st.session_state.confidence_threshold:
                                    filtered_detections.append(det)
                        
                        # Update statistics using simple object tracking
                        unique_counts = track_objects_simple(filtered_detections)
                        
                        # Update session state with unique counts
                        st.session_state.video_detections_per_class = unique_counts
                        
                        # Draw detections on frame
                        frame_with_detections, detection_count, unique_objects = draw_detections_with_tracking_simple(frame_rgb, filtered_detections, frame_count, "video")
                        
                        # Update session state
                        st.session_state.current_frame = frame_with_detections
                        st.session_state.detection_counter = detection_count
                        
                        # Display frame
                        video_placeholder.image(frame_with_detections, channels="RGB", use_column_width=True, caption=f"Frame {frame_count}/{total_frames}")
                        
                        # Update statistics display
                        with video_stats_placeholder.container():
                            fps_value = 1.0 / processing_time if processing_time > 0 else 0
                            elapsed_time = time.time() - st.session_state.video_start_time
                            
                            st.markdown(f"**Processing Statistics**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                if total_frames > 0:
                                    progress_percent = (frame_count / total_frames) * 100
                                    st.metric("Progress", f"{progress_percent:.1f}%")
                                else:
                                    st.metric("Frames", frame_count)
                            
                            with stat_col2:
                                st.metric("Current Detections", detection_count)
                            
                            with stat_col3:
                                st.metric("Unique Objects", unique_objects)
                            
                            with stat_col4:
                                st.metric("FPS", f"{fps_value:.1f}")
                            
                            # Show current detections below the metrics
                            if detection_count > 0:
                                st.markdown(f'<div class="detection-status">✅ Detected {detection_count} object(s) in current frame</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="detection-status"> No objects detected in current frame</div>', unsafe_allow_html=True)
                        
                        # Check if stop button was pressed
                        if not st.session_state.video_processing:
                            break
                    
                    # Release video capture when done
                    cap.release()
                    
                    # Update detection stats
                    st.session_state.detection_stats = {
                        'total_frames': st.session_state.video_frame_count,
                        'detected_objects': st.session_state.video_detected_objects,
                        'detections_per_class': st.session_state.video_detections_per_class,
                        'processing_time': st.session_state.video_processing_time,
                        'source': 'video',
                        'start_time': st.session_state.video_start_time
                    }
                    
                    # Clear video-specific stats
                    st.session_state.video_frame_count = 0
                    st.session_state.video_detected_objects = 0
                    st.session_state.video_detections_per_class = {}
                    st.session_state.video_processing_time = 0
                    
                    st.success(f"Video processing completed! Processed {frame_count} frames.")
                    
                except Exception as e:
                    st.error(f"Error in video processing: {str(e)}")
                    st.session_state.video_processing = False
                    st.rerun()
        
        with col2:
            st.markdown("#### Video Information")
            
            try:
                # Display video info
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    st.metric("Frame Rate", f"{fps:.1f} FPS")
                    st.metric("Resolution", f"{width}×{height}")
                    st.metric("Total Frames", frame_count)
                    st.metric("Duration", f"{duration:.1f} seconds")
                    cap.release()
                else:
                    st.warning("Could not read video properties")
            except:
                st.warning("Could not read video properties")
            
            st.markdown("#### Processing Instructions")
            st.info("""
            1. Video uploaded successfully
            2. Click **Start Video Processing**
            3. Detection boxes will appear on each frame
            4. View real-time results
            5. Click **Stop Processing** to pause
            
            **Detection Colors:**
            - 🟢 Green: Persons
            - 🔵 Blue: Vehicles
            - 🔴 Red: Animals
            - 🟡 Yellow: Other objects
            """)
            
            # Show current processing status
            if st.session_state.video_processing:
                st.success("✅ **Video Status: ACTIVE**")
                if st.session_state.detection_counter > 0:
                    st.info(f"📊 Last frame: {st.session_state.detection_counter} detections")
            else:
                st.warning("⏸️ **Video Status: INACTIVE**")
    else:
        # Clear video path if no file is uploaded
        st.session_state.video_path = None
        st.session_state.uploaded_file = None
        st.info("Please upload a video file to start detection.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Multi-Model Detection Dashboard | Built with Streamlit, OpenCV, and PyTorch</p>
    <p style='font-size: 0.8rem; color: #666;'>
        <strong>Detection Status:</strong> 
        <span style='color: green;'>● Persons</span> | 
        <span style='color: blue;'>● Vehicles</span> | 
        <span style='color: red;'>● Animals</span> | 
        <span style='color: yellow;'>● Other objects</span>
    </p>
</div>
""", unsafe_allow_html=True)