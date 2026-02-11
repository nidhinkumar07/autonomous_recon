# ui/draw_utils.py
import cv2
import numpy as np
import streamlit as st
from config.classes import CLASS_COLORS, VEHICLE_KEYWORDS, ANIMAL_KEYWORDS

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
            color = CLASS_COLORS['person']  # Green for persons
            text_color = (0, 255, 0)
        elif any(vehicle in class_name.lower() for vehicle in VEHICLE_KEYWORDS):
            color = CLASS_COLORS['vehicle']  # Blue for vehicles
            text_color = (255, 200, 0)  # Orange text for visibility on blue
        elif any(animal in class_name.lower() for animal in ANIMAL_KEYWORDS):
            color = CLASS_COLORS['animal']  # Red for animals
            text_color = (255, 255, 0)  # Yellow text for visibility on red
        else:
            color = CLASS_COLORS['other']  # Yellow for others
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
            dot_color = CLASS_COLORS['person']  # Green
        elif any(vehicle in class_name.lower() for vehicle in VEHICLE_KEYWORDS):
            dot_color = CLASS_COLORS['vehicle']  # Blue
        elif any(animal in class_name.lower() for animal in ANIMAL_KEYWORDS):
            dot_color = CLASS_COLORS['animal']  # Red
        else:
            dot_color = CLASS_COLORS['other']  # Yellow
        
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