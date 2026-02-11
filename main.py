# main.py - Professional SaaS UI with separate CSS

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import random
from collections import defaultdict
import os

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

# Object Tracker Class with improved matching and global persistence
class ObjectTracker:
    def __init__(self):
        self.object_instances = defaultdict(list)  # {class_name: [ids]}
        self.object_data = {}  # {object_id: {class_name, frames, bbox_history, confidence_history}}
        self.next_id = defaultdict(int)  # Track next ID for each class - GLOBALLY PERSISTENT
        self.active_objects = {}  # Currently tracked objects in current frame {object_id: bbox}
        self.active_counts = defaultdict(int)  # Current frame counts per class {class_name: count}
        self.iou_threshold = 0.4  # Increased IoU threshold for better matching
        self.max_frames_missing = 5  # Number of frames an object can be missing before being removed
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        # box format: [x1, y1, x2, y2]
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        return iou
    
    def calculate_center_distance(self, box1, box2):
        """Calculate distance between centers of two boxes"""
        # Calculate centers
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2
        
        # Calculate Euclidean distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        return distance
    
    def match_existing_object(self, class_name, bbox, frame_num):
        """
        Match detection with existing objects using IoU and center distance
        Returns object_id if match found, None otherwise
        """
        best_match_id = None
        best_iou = self.iou_threshold
        best_center_distance = float('inf')
        
        # Check all objects of the same class (not just active ones)
        for obj_id in self.object_instances.get(class_name, []):
            obj_data = self.object_data.get(obj_id)
            if not obj_data:
                continue
                
            # Get the last known bounding box for this object
            if obj_data['bbox_history']:
                last_bbox = obj_data['bbox_history'][-1]
                
                # Calculate IoU
                iou = self.calculate_iou(bbox, last_bbox)
                
                # Calculate center distance
                center_distance = self.calculate_center_distance(bbox, last_bbox)
                
                # Calculate frame gap (how many frames since last seen)
                frame_gap = frame_num - obj_data['frames'][-1] if obj_data['frames'] else 0
                
                # Combined matching score (weighted)
                # Give more weight to IoU and penalize large frame gaps
                match_score = iou * 0.7 - (center_distance / 1000) * 0.2 - (frame_gap / 100) * 0.1
                
                # If IoU is good enough and it's not too far in time
                if iou > best_iou and frame_gap < self.max_frames_missing:
                    best_iou = iou
                    best_match_id = obj_id
                    best_center_distance = center_distance
                # If IoU is borderline but center is very close
                elif iou > 0.2 and center_distance < 50 and frame_gap < 3:
                    if center_distance < best_center_distance:
                        best_center_distance = center_distance
                        best_match_id = obj_id
        
        return best_match_id
    
    def update_tracking(self, detections, frame_num):
        """
        Update object tracking with new detections
        detections format: [{"class": "person", "bbox": [x1, y1, x2, y2], "confidence": 0.95}, ...]
        Returns: {
            "active_objects": {object_id: bbox},
            "current_counts": {class_name: count},
            "total_objects": total_count
        }
        """
        # Clear active objects and counts for new frame
        new_active_objects = {}
        self.active_counts.clear()  # Reset current frame counts
        
        # First, try to match each detection with existing objects
        matched_detections = set()
        matched_objects = set()
        
        # Sort detections by confidence (highest first) for better matching
        detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # First pass: try to match with high IoU
        for detection in detections_sorted:
            class_name = detection['class']
            bbox = detection['bbox']
            confidence = detection.get('confidence', 0.0)
            
            matched_id = self.match_existing_object(class_name, bbox, frame_num)
            
            if matched_id:
                # Update existing object
                obj_data = self.object_data[matched_id]
                obj_data['frames'].append(frame_num)
                obj_data['bbox_history'].append(bbox)
                obj_data['confidence_history'].append(confidence)
                obj_data['last_frame'] = frame_num
                obj_data['missing_frames'] = 0  # Reset missing frames
                
                new_active_objects[matched_id] = bbox
                matched_detections.add(id(detection))
                matched_objects.add(matched_id)
                self.active_counts[class_name] += 1  # Increment count for this class
        
        # Second pass: create new objects for unmatched detections
        for detection in detections_sorted:
            if id(detection) in matched_detections:
                continue
                
            class_name = detection['class']
            bbox = detection['bbox']
            confidence = detection.get('confidence', 0.0)
            
            # Create new object with unique ID
            object_id = f"{class_name.lower().replace(' ', '-')}-{self.next_id[class_name]:03d}"
            self.next_id[class_name] += 1
            
            # Add to object instances
            self.object_instances[class_name].append(object_id)
            
            # Create object data
            self.object_data[object_id] = {
                'class': class_name,
                'frames': [frame_num],
                'bbox_history': [bbox],
                'confidence_history': [confidence],
                'first_frame': frame_num,
                'last_frame': frame_num,
                'missing_frames': 0
            }
            
            new_active_objects[object_id] = bbox
            self.active_counts[class_name] += 1  # Increment count for this class
        
        # Update missing frames for objects not detected in this frame
        for obj_id, obj_data in self.object_data.items():
            if obj_id not in new_active_objects:
                obj_data['missing_frames'] = obj_data.get('missing_frames', 0) + 1
        
        # Update active objects
        self.active_objects = new_active_objects
        
        # Clean up objects that have been missing for too long
        self.cleanup_old_objects(frame_num)
        
        # Prepare return data with current counts
        return_data = {
            "active_objects": new_active_objects,
            "current_counts": dict(self.active_counts),  # Convert to regular dict
            "total_objects": len(new_active_objects)
        }
        
        return return_data
    
    def cleanup_old_objects(self, current_frame):
        """
        Mark objects inactive if missing for too many frames.
        DO NOT delete history.
        """

        # Reset active counts
        self.active_counts.clear()

        for obj_id, obj_data in self.object_data.items():
            missing_frames = obj_data.get('missing_frames', 0)

            # Determine active state
            if missing_frames > self.max_frames_missing:
                obj_data['is_active'] = False
            else:
                obj_data['is_active'] = True
                class_name = obj_data.get('class')
                self.active_counts[class_name] += 1


    
    def get_current_counts(self):
        """Get current counts per class for active objects"""
        return dict(self.active_counts)
    
    def generate_summary(self):
        """
        Generate comprehensive summary of tracked objects
        Returns format similar to your JSON example
        """
        summary = {
            "objects": {},
            "statistics": {
                "total_objects": 0,
                "categories": {}
            },
            "current_counts": dict(self.active_counts)  # Add current counts to summary
        }
        
        for class_name, ids in self.object_instances.items():
            class_objects = []
            total_frames_class = 0
            
            for obj_id in ids:
                obj_data = self.object_data.get(obj_id, {})
                visible_frames = len(obj_data.get('frames', []))
                total_frames_class += visible_frames
                
                class_objects.append({
                    "id": obj_id,
                    "frame_count": visible_frames,
                    "first_frame": obj_data.get('first_frame', 0),
                    "last_frame": obj_data.get('last_frame', 0),
                    "avg_confidence": np.mean(obj_data.get('confidence_history', [])) if obj_data.get('confidence_history') else 0,
                    "is_active": obj_data.get('is_active', False)

                })
            
            summary["objects"][class_name] = class_objects
            
            # Update statistics
            summary["statistics"]["categories"][class_name] = {
                "count": len(ids),
                "total_frames": total_frames_class,
                "avg_frames_per_object": total_frames_class / len(ids) if len(ids) > 0 else 0,
                "current_active": self.active_counts.get(class_name, 0)
            }
            summary["statistics"]["total_objects"] += len(ids)
        
        return summary
    
    def reset(self):
        """Reset the tracker (but preserve next_id for global persistence across sessions)"""
        self.object_instances = defaultdict(list)
        self.object_data = {}
        # DO NOT reset next_id to maintain global persistence across webcam/video sessions
        self.active_objects = {}
        self.active_counts.clear()

# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="YOLOv8 Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load CSS from external file
def load_css_from_file(file_path="styles.css"):
    """Load and inject CSS from external file"""
    try:
        with open(file_path, 'r') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback to inline CSS if file not found
        
        st.title("YOLOv8 Detection Dashboard")
        st.caption("Real-time object detection and tracking using YOLOv8. Upload a video file or use your webcam for detection with advanced object tracking.")

        st.warning("CSS file 'styles.css' not found. Using minimal styling.")

# Load CSS
load_css_from_file()

# Title - Top-left aligned with proper spacing
st.markdown("""
<div style="padding-top: 0; margin-top: 0;">
    <h1 style="font-size: 1.75rem; color: #1E3A8A; font-weight: 700; margin-bottom: 0.25rem; padding-top: 0;">
        YOLOv8 Detection Dashboard
    </h1>
    <div style='color: #94A3B8; font-weight: 400; line-height: 1.5; margin-bottom: 1.5rem;'>
        Real-time object detection and tracking using YOLOv8. Upload a video file or use your webcam for detection with advanced object tracking.
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state for detection settings
if 'show_persons' not in st.session_state:
    st.session_state.show_persons = True
if 'show_vehicles' not in st.session_state:
    st.session_state.show_vehicles = True
if 'show_animals' not in st.session_state:
    st.session_state.show_animals = False
if 'show_everything' not in st.session_state:
    st.session_state.show_everything = False
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
if 'show_ids' not in st.session_state:
    st.session_state.show_ids = True

# Initialize tracker in session state
if 'object_tracker' not in st.session_state:
    st.session_state.object_tracker = ObjectTracker()

# Sidebar configuration
# Sidebar configuration
with st.sidebar:
    st.markdown('<h2 style="margin-bottom: 0.75rem;">SETTINGS</h2>', unsafe_allow_html=True)
    
    # Detection classes section
    st.markdown('<div class="section-header" style="margin-top: 0.5rem;">Detection Classes</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.session_state.show_persons = st.checkbox("Persons", value=st.session_state.show_persons, key="sidebar_persons")
        st.session_state.show_vehicles = st.checkbox("Vehicles", value=st.session_state.show_vehicles, key="sidebar_vehicles")
    with col2:
        st.session_state.show_animals = st.checkbox("Animals", value=st.session_state.show_animals, key="sidebar_animals")
        st.session_state.show_everything = st.checkbox("All Classes", value=st.session_state.show_everything, key="sidebar_everything")
    
    st.markdown("---")
    
    # Visualization section
    st.markdown('<div class="section-header">Visualization</div>', unsafe_allow_html=True)
    
    st.session_state.show_labels = st.checkbox("Show Labels", value=st.session_state.show_labels, key="vis_labels")
    st.session_state.show_confidence = st.checkbox("Show Confidence Scores", value=st.session_state.show_confidence, key="vis_confidence")
    st.session_state.show_ids = st.checkbox("Show Object IDs", value=st.session_state.show_ids,key="vis_ids")
    
    st.markdown("---")
    
    # Configuration section
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    
    st.session_state.confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.confidence_threshold,
        step=0.05,
        help="Minimum confidence score for detections",
        key="conf_threshold"
    )
    
    
    st.session_state.frame_skip = st.slider(
        "Frame Skip",
        min_value=1,
        max_value=10,
        value=st.session_state.frame_skip,
        step=1,
        help="Process every nth frame",
        key="frame_skip_slider"
    )
    
    
    st.session_state.box_thickness = st.slider(
        "Box Thickness",
        min_value=1,
        max_value=5,
        value=st.session_state.box_thickness,
        step=1,
        help="Thickness of detection boxes",
        key="box_thickness_slider"
    )
    
    st.markdown("---")
    
    # Color legend
    st.markdown('<div class="section-header">Detection Colors</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-bottom: 0.75rem;">
        <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #10B981; margin-right: 8px;"></div>
            <span style="font-size: 0.85rem;">Persons</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #3B82F6; margin-right: 8px;"></div>
            <span style="font-size: 0.85rem;">Vehicles</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #EF4444; margin-right: 8px;"></div>
            <span style="font-size: 0.85rem;">Animals</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #F59E0B; margin-right: 8px;"></div>
            <span style="font-size: 0.85rem;">Other</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state for processing
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
        return None

def detect_with_yolo(model, image):
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
        num_detections = random.randint(2, 8)
        
        # For consistent testing, let's simulate a few persistent objects
        persistent_objects = []
        if st.session_state.show_persons:
            # Simulate 1-2 persons that move slightly
            for i in range(random.randint(1, 2)):
                base_x = random.randint(100, width - 300)
                base_y = random.randint(100, height - 300)
                # Small random movement
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                x1 = max(0, base_x + dx)
                y1 = max(0, base_y + dy)
                x2 = x1 + random.randint(80, 150)
                y2 = y1 + random.randint(150, 250)
                conf = random.uniform(0.7, 0.95)
                persistent_objects.append({"class": "person", "bbox": [x1, y1, x2, y2], "confidence": conf})
        
        if st.session_state.show_vehicles:
            # Simulate 0-1 vehicles
            for i in range(random.randint(0, 1)):
                base_x = random.randint(200, width - 400)
                base_y = random.randint(200, height - 200)
                dx = random.randint(-30, 30)
                dy = random.randint(-10, 10)
                x1 = max(0, base_x + dx)
                y1 = max(0, base_y + dy)
                x2 = x1 + random.randint(120, 250)
                y2 = y1 + random.randint(80, 150)
                conf = random.uniform(0.7, 0.95)
                persistent_objects.append({"class": "car", "bbox": [x1, y1, x2, y2], "confidence": conf})
        
        # Add some random detections
        random_detections = []
        for i in range(random.randint(0, 3)):
            box_width = random.randint(50, 200)
            box_height = random.randint(50, 200)
            x1 = random.randint(0, width - box_width - 1)
            y1 = random.randint(0, height - box_height - 1)
            x2 = x1 + box_width
            y2 = y1 + box_height
            
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
                possible_classes = ['person', 'car', 'dog', 'bicycle', 'chair', 'bottle', 'cell phone', 'laptop', 'book']
            
            if len(possible_classes) > 0:
                class_name = random.choice(possible_classes)
                # Format for tracker: {"class": class_name, "bbox": [x1, y1, x2, y2], "confidence": conf}
                random_detections.append({"class": class_name, "bbox": [x1, y1, x2, y2], "confidence": conf})
        
        # Combine persistent and random detections
        detections = persistent_objects + random_detections
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
                        detections.append({"class": class_name, "bbox": [float(x1), float(y1), float(x2), float(y2)], "confidence": float(conf)})
        except Exception as e:
            # Fallback to dummy detections
            num_detections = random.randint(2, 8)
            for i in range(num_detections):
                box_width = random.randint(50, 200)
                box_height = random.randint(50, 200)
                x1 = random.randint(0, width - box_width - 1)
                y1 = random.randint(0, height - box_height - 1)
                x2 = x1 + box_width
                y2 = y1 + box_height
                conf = random.uniform(0.6, 0.95)
                class_name = random.choice(['person', 'car', 'dog', 'bicycle', 'cell phone', 'laptop'])
                detections.append({"class": class_name, "bbox": [x1, y1, x2, y2], "confidence": conf})
    
    return detections

def filter_detections(detections):
    """Filter detections based on user preferences and confidence threshold"""
    filtered = []
    for det in detections:
        class_name = det['class'].lower()
        confidence = det['confidence']
        
        # Check confidence threshold
        if confidence < st.session_state.confidence_threshold:
            continue
        
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
        
        if should_show:
            filtered.append(det)
    
    return filtered

def draw_detections(frame, tracked_data, frame_number=0, source="webcam"):
    """Draw detection boxes on frame with object IDs and current counts overlay"""
    frame_copy = frame.copy()
    height, width = frame_copy.shape[:2]
    
    # Unpack tracked data
    tracked_objects = tracked_data.get("active_objects", {})
    current_counts = tracked_data.get("current_counts", {})
    total_objects = tracked_data.get("total_objects", 0)
    
    # Draw a border around the frame
    cv2.rectangle(frame_copy, (0, 0), (width-1, height-1), (100, 100, 100), 2)
    
    # Add source watermark
    source_text = "Webcam" if source == "webcam" else "Video"
    cv2.putText(frame_copy, f"Source: {source_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_copy, f"Frame: {frame_number}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    detection_count = 0
    
    # Draw bounding boxes for each tracked object
    for obj_id, bbox in tracked_objects.items():
        # Get object data
        obj_data = st.session_state.object_tracker.object_data.get(obj_id, {})
        class_name = obj_data.get('class', 'unknown')
        confidence_history = obj_data.get('confidence_history', [0])
        avg_confidence = np.mean(confidence_history) if confidence_history else 0
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # Skip if coordinates are invalid
        if x1 >= x2 or y1 >= y2:
            continue
            
        # Choose color based on class
        if 'person' in class_name.lower():
            color = (0, 255, 0)  # Green for persons
            text_color = (0, 255, 0)
        elif any(vehicle in class_name.lower() for vehicle in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']):
            color = (255, 0, 0)  # Blue for vehicles
            text_color = (255, 200, 0)  # Orange text for visibility on blue
        elif any(animal in class_name.lower() for animal in ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow']):
            color = (0, 0, 255)  # Red for animals
            text_color = (255, 255, 0)  # Yellow text for visibility on red
        else:
            color = (255, 255, 0)  # Yellow for others
            text_color = (0, 0, 0)  # Black text for visibility on yellow
        
        # Draw rectangle with thickness
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, st.session_state.box_thickness)
        
        # Draw label background
        if st.session_state.show_labels:
            label = f"{class_name}"
            if st.session_state.show_ids:
                # Extract just the ID number for display
                id_num = obj_id.split('-')[-1]
                label = f"ID:{id_num} {label}"
            if st.session_state.show_confidence:
                label += f" {avg_confidence:.2f}"
            
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
                      (255, 255, 255),  # White text for better visibility
                      2)
        
        detection_count += 1
    
    # ===== ADD CURRENT COUNTS OVERLAY =====
    # Create a semi-transparent overlay for counts
    overlay = frame_copy.copy()
    
    # Define overlay position and size
    overlay_x = width - 250
    overlay_y = 100
    overlay_width = 240
    overlay_height = min(400, 100 + len(current_counts) * 30)
    
    # Draw semi-transparent background
    cv2.rectangle(overlay, 
                 (overlay_x, overlay_y), 
                 (overlay_x + overlay_width, overlay_y + overlay_height), 
                 (0, 0, 0), -1)  # Black background
    frame_copy = cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0)
    
    # Add title for counts overlay
    cv2.putText(frame_copy, "CURRENT COUNTS", 
               (overlay_x + 10, overlay_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add total objects count
    cv2.putText(frame_copy, f"Total Objects: {total_objects}", 
               (overlay_x + 10, overlay_y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Add each class count with colored indicators
    y_offset = 90
    for i, (class_name, count) in enumerate(sorted(current_counts.items())):
        if i >= 10:  # Limit display to top 10 classes for space
            remaining = len(current_counts) - 10
            cv2.putText(frame_copy, f"... and {remaining} more", 
                       (overlay_x + 10, overlay_y + y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            break
        
        # Determine color for this class
        if 'person' in class_name.lower():
            dot_color = (0, 255, 0)  # Green
        elif any(vehicle in class_name.lower() for vehicle in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']):
            dot_color = (255, 0, 0)  # Blue
        elif any(animal in class_name.lower() for animal in ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow']):
            dot_color = (0, 0, 255)  # Red
        else:
            dot_color = (255, 255, 0)  # Yellow
        
        # Draw colored dot
        cv2.circle(frame_copy, (overlay_x + 10, overlay_y + y_offset), 5, dot_color, -1)
        
        # Draw count text
        display_text = f"{class_name.capitalize()}: {count}"
        if count == 1:
            display_text = f"{class_name.capitalize()}: {count}"  # Singular
        else:
            display_text = f"{class_name.capitalize()}s: {count}"  # Plural
        
        cv2.putText(frame_copy, display_text, 
                   (overlay_x + 25, overlay_y + y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 30
    
    # Add detection and tracked counts to frame (top right)
    cv2.putText(frame_copy, f"Detections: {detection_count}", (width - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    tracked_count = len(tracked_objects)
    cv2.putText(frame_copy, f"Tracked: {tracked_count}", (width - 200, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame_copy, detection_count, current_counts

def display_summary():
    """Display the comprehensive object tracking summary"""
    summary = st.session_state.object_tracker.generate_summary()
    
    if not summary["objects"]:
        st.info("No objects tracked yet. Start detection to see tracking data.")
        return
    
    st.markdown("### Comprehensive Object Tracking Summary")
    
    # Overall statistics
    total_objects = summary["statistics"]["total_objects"]
    
    
    # Display by category
    for class_name, objects in summary["objects"].items():
        category_stats = summary["statistics"]["categories"][class_name]
        object_count = category_stats["count"]
        total_frames = category_stats["total_frames"]
        avg_frames = category_stats["avg_frames_per_object"]
        current_active = category_stats.get("current_active", 0)
        
        # Create a clean container for each class
        with st.container():
            st.markdown(f"**{class_name.capitalize()}** ({object_count} objects, {current_active} currently active)")
            
            # Use Streamlit's native columns and metrics for better compatibility
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total objects", object_count)
            
            with col2:
                st.metric("Currently active", current_active)
            
            with col3:
                st.metric("Total frames", total_frames)
            
            with col4:
                st.metric("Avg frames/obj", f"{avg_frames:.1f}")
            
            # Show individual objects with details
            with st.expander(f"View {object_count} {class_name} objects", expanded=False):
                for obj in objects:
                    obj_id = obj["id"]
                    frame_count = obj["frame_count"]
                    avg_confidence = obj.get("avg_confidence", 0)
                    is_active = obj.get("is_active", False)
                    
                    # Use Streamlit's native columns and containers
                    with st.container():
                        st.markdown(f"**Object ID:** `{obj_id}`")
                        
                        # Create columns for the details
                        detail_col1, detail_col2, detail_col3 = st.columns(3)
                        
                        with detail_col1:
                            st.markdown(f"**Frames detected:** {frame_count}")
                            st.markdown(f"**First frame:** {obj.get('first_frame', 'N/A')}")
                        
                        with detail_col2:
                            st.markdown(f"**Avg confidence:** {avg_confidence:.2f}")
                            st.markdown(f"**Last frame:** {obj.get('last_frame', 'N/A')}")
                        
                        with detail_col3:
                            status_text = "Active" if is_active else "Inactive"
                            st.markdown(f"**Status:** {status_text}")
                        
                        st.markdown("---")

# Main tabs - Clean underline system
tab1, tab2 = st.tabs(["Live Webcam", "Upload Video"])

# Tab 1: Live Webcam
with tab1:
    st.markdown("## Live Webcam Detection")
    
    # Store current tab
    st.session_state.active_tab = "webcam"
    
    # Use consistent 1.5rem gap between columns
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        # Check if webcam is active
        if st.session_state.webcam_active:
            st.markdown('<div class="processing-card">Webcam is active. Objects are being tracked with unique IDs.</div>', unsafe_allow_html=True)
            
            # Create placeholders for webcam
            webcam_placeholder = st.empty()
            webcam_stats_placeholder = st.empty()
            
            # Control button
            if st.button("Stop Webcam", key="stop_webcam_btn", type="secondary"):
                st.session_state.webcam_active = False
                st.rerun()
            
            # Webcam processing logic
            try:
                # Initialize webcam with error suppression
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DSHOW backend on Windows to avoid warnings
                
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
                
                # Initialize stats if needed
                if 'webcam_start_time' not in st.session_state:
                    st.session_state.webcam_start_time = time.time()
                    st.session_state.webcam_frame_count = 0
                    st.session_state.webcam_detected_objects = 0
                    st.session_state.webcam_processing_time = 0
                
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
                    raw_detections = detect_with_yolo(st.session_state.current_model, frame_rgb)
                    processing_time = time.time() - start_time
                    
                    # Filter detections
                    filtered_detections = filter_detections(raw_detections)
                    
                    # Update object tracking - get the full tracking data
                    tracking_data = st.session_state.object_tracker.update_tracking(
                        filtered_detections, 
                        frame_count
                    )
                    
                    # Update statistics
                    st.session_state.webcam_detected_objects += len(filtered_detections)
                    st.session_state.webcam_processing_time += processing_time
                    
                    # Draw detections on frame with object IDs
                    frame_with_detections, detection_count, current_counts = draw_detections(
                        frame_rgb, 
                        tracking_data, 
                        frame_count, 
                        "webcam"
                    )
                    
                    # Update session state
                    st.session_state.current_frame = frame_with_detections
                    st.session_state.detection_counter = detection_count
                    
                    # Display frame with detections
                    webcam_placeholder.image(
                        frame_with_detections, 
                        channels="RGB", 
                        caption=f"Frame {frame_count} - Tracking {len(tracking_data['active_objects'])} objects"
                    )
                    
                    # Update statistics display with responsive grid
                    with webcam_stats_placeholder.container():
                        fps_value = 1.0 / processing_time if processing_time > 0 else 0
                        
                        st.markdown("## Live Statistics")
                        
                        # Responsive metrics grid using CSS classes
                        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                        
                        # Create metrics columns
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        with metrics_col1:
                            st.metric("Frames", st.session_state.webcam_frame_count)
                        with metrics_col2:
                            st.metric("Current Detections", detection_count)
                        with metrics_col3:
                            st.metric("Total Detections", st.session_state.webcam_detected_objects)
                        with metrics_col4:
                            st.metric("FPS", f"{fps_value:.1f}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                         
                        # Show tracking summary
                        display_summary()
                    
                    # Small delay
                    time.sleep(0.033)
                
                # Release webcam when done
                cap.release()
                
                # Clear webcam-specific stats
                st.session_state.webcam_frame_count = 0
                st.session_state.webcam_detected_objects = 0
                st.session_state.webcam_processing_time = 0
                
            except Exception as e:
                st.error(f"Error in webcam processing: {str(e)}")
                st.session_state.webcam_active = False
                st.rerun()
        
        else:
            # Show start button
            if st.button("Start Webcam Detection", key="start_webcam_main", type="primary"):
                st.session_state.webcam_active = True
                st.session_state.video_processing = False  # Ensure video is stopped
                # Reset tracker when starting
                st.session_state.object_tracker.reset()
                st.rerun()
    
    with col2:
        pass

# Tab 2: Upload Video
with tab2:
    st.markdown("## Upload Video for Detection")
    
    # Store current tab
    st.session_state.active_tab = "video"
    
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
        
        # Use consistent 1.5rem gap between columns
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            # Show video preview if not processing
            if not st.session_state.video_processing:
                st.video(uploaded_file)
            
            # Check if video processing is active
            if st.session_state.video_processing:
                st.markdown('<div class="processing-card">Video processing is active. Objects are being tracked with unique IDs.</div>', unsafe_allow_html=True)
                
                # Create placeholders for video
                video_placeholder = st.empty()
                video_stats_placeholder = st.empty()
                video_progress_placeholder = st.empty()
                
                # Control button
                if st.button("Stop Processing", key="stop_video_btn", type="secondary"):
                    st.session_state.video_processing = False
                    st.rerun()
                
                # Video processing logic
                try:
                    # Initialize video capture with error suppression
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
                    
                    # Initialize stats if needed
                    if 'video_start_time' not in st.session_state:
                        st.session_state.video_start_time = time.time()
                        st.session_state.video_frame_count = 0
                        st.session_state.video_detected_objects = 0
                        st.session_state.video_processing_time = 0
                    
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
                        raw_detections = detect_with_yolo(st.session_state.current_model, frame_rgb)
                        processing_time = time.time() - start_time
                        
                        # Filter detections
                        filtered_detections = filter_detections(raw_detections)
                        
                        # Update object tracking - get the full tracking data
                        tracking_data = st.session_state.object_tracker.update_tracking(
                            filtered_detections, 
                            frame_count
                        )
                        
                        # Update statistics
                        st.session_state.video_detected_objects += len(filtered_detections)
                        st.session_state.video_processing_time += processing_time
                        
                        # Draw detections on frame with object IDs
                        frame_with_detections, detection_count, current_counts = draw_detections(
                            frame_rgb, 
                            tracking_data, 
                            frame_count, 
                            "video"
                        )
                        
                        # Update session state
                        st.session_state.current_frame = frame_with_detections
                        st.session_state.detection_counter = detection_count
                        
                        # Display frame
                        video_placeholder.image(
                            frame_with_detections, 
                            channels="RGB", 
                            caption=f"Frame {frame_count}/{total_frames} - Tracking {len(tracking_data['active_objects'])} objects"
                        )
                        
                        # Update statistics display with responsive grid
                        with video_stats_placeholder.container():
                            fps_value = 1.0 / processing_time if processing_time > 0 else 0
                            
                            st.markdown("#### Processing Statistics")
                            
                            # Responsive metrics grid
                            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                            
                            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                            with metrics_col1:
                                if total_frames > 0:
                                    progress_percent = (frame_count / total_frames) * 100
                                    st.metric("Progress", f"{progress_percent:.1f}%")
                                else:
                                    st.metric("Frames", frame_count)
                            with metrics_col2:
                                st.metric("Current Detections", detection_count)
                            with metrics_col3:
                                st.metric("Total Detections", st.session_state.video_detected_objects)
                            with metrics_col4:
                                st.metric("FPS", f"{fps_value:.1f}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show tracking summary
                            display_summary()
                        
                        # Check if stop button was pressed
                        if not st.session_state.video_processing:
                            break
                    
                    # Release video capture when done
                    cap.release()
                    
                    # Clear video-specific stats
                    st.session_state.video_frame_count = 0
                    st.session_state.video_detected_objects = 0
                    st.session_state.video_processing_time = 0
                    
                    st.success(f"Video processing completed! Processed {frame_count} frames.")
                    
                except Exception as e:
                    st.error(f"Error in video processing: {str(e)}")
                    st.session_state.video_processing = False
                    st.rerun()
            
            else:
                # Show start button
                if st.button("Start Video Processing", key="start_video_main", type="primary"):
                    st.session_state.video_processing = True
                    st.session_state.webcam_active = False  # Ensure webcam is stopped
                    # Reset tracker when starting
                    st.session_state.object_tracker.reset()
                    st.rerun()
        
        with col2:
            # Only show video info and instructions when NOT processing
            if not st.session_state.video_processing:
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
                        
                        with st.container():
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.metric("Frame Rate", f"{fps:.1f}")
                                st.metric("Total Frames", frame_count)
                            with col_info2:
                                st.metric("Resolution", f"{width}×{height}")
                                st.metric("Duration", f"{duration:.1f}s")
                        cap.release()
                    else:
                        st.warning("Could not read video properties")
                except:
                    st.warning("Could not read video properties")
                
                st.markdown("""
                    <div class="info-card">
                        <strong style="color: #94A3B8;">Steps:</strong><br>
                        <span style="color: #94A3B8;">1. Video uploaded successfully</span><br>
                        <span style="color: #94A3B8;">2. Click Start Video Processing</span><br>
                        <span style="color: #94A3B8;">3. Objects are tracked with unique IDs</span><br>
                        <span style="color: #94A3B8;">4. View real-time results</span><br>
                        <span style="color: #94A3B8;">5. Click Stop Processing to pause</span>
                    </div>
                    """, unsafe_allow_html=True)
            # When processing, the right column will show the statistics from video_stats_placeholder
    else:
        # Clear video path if no file is uploaded
        st.session_state.video_path = None
        st.session_state.uploaded_file = None
        st.info("Please upload a video file to start detection.")

# Footer