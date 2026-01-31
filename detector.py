from ultralytics import YOLO
import cv2


class ObjectDetector:
    def  __init__(self, model_path='yolov8n.pt', conf=0.4):
            self.model = YOLO(model_path)
            self.conf = conf


def detect(self, frame):
    results = self.model(frame, conf=self.conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]
            conf = float(box.conf[0])
            detections.append((x1, y1, x2, y2, label, conf))
            return detections