# detection_models.py - Detection model implementations

import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore') 

# COCO class names for different models
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
    'refrigerator', 'book', 'clock', 'vase','scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# YOLOv8 Model Functions
def load_yolo_model():
    """Load YOLOv8 model"""
    try:
        # Try to import ultralytics YOLO
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Using nano version for speed
        return model
    except ImportError:
        # Fallback to OpenCV DNN if ultralytics not available
        print("Ultralytics not available, using OpenCV DNN as fallback")
        net = cv2.dnn.readNetFromDarknet(
            "yolov3.cfg", 
            "yolov3.weights"
        )
        return net

def detect_with_yolo(model, image, model_choice="YOLOv8"):
    """Perform detection with YOLO model"""
    detections = []
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
    
    # Check model type
    if hasattr(model, 'predict'):  # Ultralytics YOLO
        results = model(image_pil, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                    
                    # Filter for person-only mode
                    if "Person Only" in model_choice and cls_name != "person":
                        continue
                    
                    detections.append([x1, y1, x2, y2, conf, cls_id, cls_name])
    else:  # OpenCV DNN fallback
        height, width = image.shape[:2]
        
        # Prepare image for DNN
        blob = cv2.dnn.blobFromImage(
            image, 1/255.0, (416, 416), 
            swapRB=True, crop=False
        )
        model.setInput(blob)
        
        # Get detection layers
        layer_names = model.getLayerNames()
        output_layers = [
            layer_names[i - 1] 
            for i in model.getUnconnectedOutLayers()
        ]
        
        # Forward pass
        outputs = model.forward(output_layers)
        
        # Process outputs
        conf_threshold = 0.5
        nms_threshold = 0.4
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, 
            conf_threshold, nms_threshold
        )
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                x1, y1, x2, y2 = x, y, x + w, y + h
                conf = confidences[i]
                cls_id = class_ids[i]
                cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                
                # Filter for person-only mode
                if "Person Only" in model_choice and cls_name != "person":
                    continue
                
                detections.append([x1, y1, x2, y2, conf, cls_id, cls_name])
    
    return detections

# MobileNet SSD Model Functions
def load_mobilenet_model():
    """Load MobileNet SSD model"""
    # Load pre-trained MobileNet SSD
    model = models.detection.ssdlite320_mobilenet_v3_large(
        weights=models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    )
    model.eval()
    return model

def detect_with_mobilenet(model, image):
    """Perform detection with MobileNet SSD"""
    # Transform image
    transform = T.Compose([
        T.ToTensor(),
    ])
    
    # Convert to tensor
    if isinstance(image, np.ndarray):
        image_tensor = transform(image)
    else:
        image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    # Perform detection
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Process predictions
    detections = []
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    # MobileNet SSD COCO class mapping (first 90 classes)
    coco_labels = COCO_CLASSES[:91]
    
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = box
            cls_id = label
            cls_name = coco_labels[cls_id] if cls_id < len(coco_labels) else f"class_{cls_id}"
            
            detections.append([x1, y1, x2, y2, score, cls_id, cls_name])
    
    return detections

# Faster R-CNN Model Functions
def load_fasterrcnn_model():
    """Load Faster R-CNN model"""
    # Load pre-trained Faster R-CNN
    model = models.detection.fasterrcnn_resnet50_fpn(
        weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()
    return model

def detect_with_fasterrcnn(model, image):
    """Perform detection with Faster R-CNN"""
    # Transform image
    transform = T.Compose([
        T.ToTensor(),
    ])
    
    # Convert to tensor
    if isinstance(image, np.ndarray):
        image_tensor = transform(image)
    else:
        image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    # Perform detection
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Process predictions
    detections = []
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    # Faster R-CNN uses COCO classes
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = box
            cls_id = label - 1  # COCO labels start from 1
            cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
            
            detections.append([x1, y1, x2, y2, score, cls_id, cls_name])
    
    return detections

# Utility function for custom models
def load_custom_model(model_path):
    """Load a custom trained model"""
    try:
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        return model
    except:
        print(f"Error loading custom model from {model_path}")
        return None