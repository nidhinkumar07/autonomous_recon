# core/detector.py
import cv2
import numpy as np
import random
from PIL import Image
from config.classes import COCO_CLASSES, VEHICLE_KEYWORDS, ANIMAL_KEYWORDS
import streamlit as st

def load_yolo_model():
    """Load YOLO model with fallback"""
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
        elif any(vehicle in class_name for vehicle in VEHICLE_KEYWORDS) and st.session_state.show_vehicles:
            should_show = True
        elif any(animal in class_name for animal in ANIMAL_KEYWORDS) and st.session_state.show_animals:
            should_show = True
        
        if should_show:
            filtered.append(det)
    
    return filtered