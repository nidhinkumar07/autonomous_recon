# core/tracker.py
from collections import defaultdict
import numpy as np

class ObjectTracker:
    """Advanced object tracker with global persistence and unique ID management"""
    
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