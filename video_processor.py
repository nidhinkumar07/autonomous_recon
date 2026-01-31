import cv2


def draw_boxes(frame, detections):
    counts = {}
    for (x1, y1, x2, y2, label, conf) in detections:
        counts[label] = counts.get(label, 0) + 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return frame, counts