import sys
import numpy as np
from collections import defaultdict, deque

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

# Maximum history entries kept per object — prevents unbounded memory growth (fix #3,4,5)
_MAX_HISTORY = 30


class ObjectTracker:
    """
    IoU + centre-distance object tracker with bounded memory.

    Fixes applied vs previous version:
    - bbox_history, confidence_history, frames are all capped at _MAX_HISTORY
      entries via collections.deque — no more unbounded list growth.
    - frame_count stored as a plain int (incremented) instead of a full list.
    - avg_confidence stored as a running mean — generate_summary() now reads it
      directly instead of calling np.mean() on the history list.
    - Dead match_score variable removed.
    - summary_dirty flag added so generate_summary() can be skipped when nothing changed.
    """

    def __init__(self):
        self.object_instances  = defaultdict(list)   # {class_name: [ids]}
        self.object_data       = {}                  # {object_id: data dict}
        self.next_id           = defaultdict(int)    # globally persistent ID counters
        self.active_objects    = {}                  # {object_id: bbox} for current frame
        self.active_counts     = defaultdict(int)    # {class_name: count} for current frame
        self.iou_threshold     = 0.4
        self.max_frames_missing = 5
        self._summary_dirty    = True                # invalidated on every state change
        self._cached_summary   = None

    # ── Geometry helpers ───────────────────────────────────────────────────

    def calculate_iou(self, box1, box2) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x1_1, x1_2);  yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2);  yi2 = min(y2_1, y2_2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a1    = (x2_1 - x1_1) * (y2_1 - y1_1)
        a2    = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def calculate_center_distance(self, box1, box2) -> float:
        cx1 = (box1[0] + box1[2]) / 2;  cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2;  cy2 = (box2[1] + box2[3]) / 2
        return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))

    # ── Matching ───────────────────────────────────────────────────────────

    def match_existing_object(self, class_name, bbox, frame_num):
        """
        Return the best-matching object_id for `bbox` or None.
        Uses IoU as the primary criterion; falls back to centre distance
        for borderline cases.  The unused match_score variable is removed (fix #2).
        """
        best_match_id       = None
        best_iou            = self.iou_threshold
        best_center_dist    = float('inf')

        for obj_id in self.object_instances.get(class_name, []):
            obj_data = self.object_data.get(obj_id)
            if not obj_data or not obj_data['bbox_history']:
                continue

            last_bbox  = obj_data['bbox_history'][-1]
            iou        = self.calculate_iou(bbox, last_bbox)
            frame_gap  = frame_num - obj_data['last_frame']

            if iou > best_iou and frame_gap < self.max_frames_missing:
                best_iou        = iou
                best_match_id   = obj_id
                best_center_dist = self.calculate_center_distance(bbox, last_bbox)

            elif iou > 0.2 and frame_gap < 3:
                cd = self.calculate_center_distance(bbox, last_bbox)
                if cd < 50 and cd < best_center_dist:
                    best_center_dist = cd
                    best_match_id    = obj_id

        return best_match_id

    # ── Main update ────────────────────────────────────────────────────────

    def update_tracking(self, detections: list, frame_num: int) -> dict:
        """
        Update tracker with new detections.  Returns:
            {active_objects, current_counts, total_objects}
        """
        new_active  = {}
        self.active_counts.clear()
        matched_det = set()

        # Sort by confidence (highest first) for stable greedy matching
        detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        # Pass 1 — match existing objects
        for det in detections_sorted:
            class_name = det['class']
            bbox       = det['bbox']
            confidence = det.get('confidence', 0.0)

            matched_id = self.match_existing_object(class_name, bbox, frame_num)
            if matched_id:
                od = self.object_data[matched_id]
                # Bounded deque append — O(1), no unbounded growth (fix #3,4,5)
                od['bbox_history'].append(bbox)
                od['confidence_history'].append(confidence)
                od['frame_count']   += 1
                od['last_frame']     = frame_num
                od['missing_frames'] = 0
                # Running average — O(1), no np.mean on growing list (fix #1)
                n = od['frame_count']
                od['avg_confidence'] = od['avg_confidence'] * (n - 1) / n + confidence / n

                new_active[matched_id] = bbox
                matched_det.add(id(det))
                self.active_counts[class_name] += 1

        # Pass 2 — create new objects for unmatched detections
        for det in detections_sorted:
            if id(det) in matched_det:
                continue
            class_name = det['class']
            bbox       = det['bbox']
            confidence = det.get('confidence', 0.0)

            obj_id = f"{class_name.lower().replace(' ', '-')}-{self.next_id[class_name]:03d}"
            self.next_id[class_name] += 1
            self.object_instances[class_name].append(obj_id)

            self.object_data[obj_id] = {
                'class':              class_name,
                'frame_count':        1,
                'bbox_history':       deque([bbox],       maxlen=_MAX_HISTORY),
                'confidence_history': deque([confidence], maxlen=_MAX_HISTORY),
                'avg_confidence':     confidence,
                'first_frame':        frame_num,
                'last_frame':         frame_num,
                'missing_frames':     0,
                'is_active':          True,
            }

            new_active[obj_id] = bbox
            self.active_counts[class_name] += 1

        # Increment missing-frame counter for absent objects
        for obj_id, od in self.object_data.items():
            if obj_id not in new_active:
                od['missing_frames'] = od.get('missing_frames', 0) + 1

        self.active_objects  = new_active
        self._summary_dirty  = True          # state changed — invalidate cache
        self.cleanup_old_objects()

        return {
            "active_objects": new_active,
            "current_counts": dict(self.active_counts),
            "total_objects":  len(new_active),
        }

    # ── Cleanup ────────────────────────────────────────────────────────────

    def cleanup_old_objects(self):
        """Mark objects inactive when missing too long. Counts already reset above."""
        self.active_counts.clear()
        for od in self.object_data.values():
            if od.get('missing_frames', 0) > self.max_frames_missing:
                od['is_active'] = False
            else:
                od['is_active'] = True
                self.active_counts[od['class']] += 1

    # ── Public helpers ─────────────────────────────────────────────────────

    def get_current_counts(self) -> dict:
        return dict(self.active_counts)

    def generate_summary(self) -> dict:
        """
        Build a summary dict.  Uses cached result when nothing has changed
        (fix #11 — avoids full rebuild every throttle interval).
        avg_confidence now reads the stored running mean, never np.mean (fix #1).
        frame_count is a plain int — no len() of a deque needed.
        """
        if not self._summary_dirty and self._cached_summary is not None:
            return self._cached_summary

        summary = {
            "objects":    {},
            "statistics": {"total_objects": 0, "categories": {}},
            "current_counts": dict(self.active_counts),
        }

        for class_name, ids in self.object_instances.items():
            class_objects      = []
            total_frames_class = 0

            for obj_id in ids:
                od = self.object_data.get(obj_id, {})
                fc  = od.get('frame_count', 0)
                total_frames_class += fc
                class_objects.append({
                    "id":             obj_id,
                    "frame_count":    fc,
                    "first_frame":    od.get('first_frame', 0),
                    "last_frame":     od.get('last_frame', 0),
                    "avg_confidence": od.get('avg_confidence', 0.0),   # fix #1
                    "is_active":      od.get('is_active', False),
                })

            n = len(ids)
            summary["objects"][class_name] = class_objects
            summary["statistics"]["categories"][class_name] = {
                "count":                n,
                "total_frames":         total_frames_class,
                "avg_frames_per_object": total_frames_class / n if n else 0,
                "current_active":       self.active_counts.get(class_name, 0),
            }
            summary["statistics"]["total_objects"] += n

        self._cached_summary = summary
        self._summary_dirty  = False
        return summary

    def export_to_dict(self) -> dict:
        """
        Return full tracking data as a plain JSON-serialisable dict.
        Used by the CSV/JSON export feature.
        """
        rows = []
        for class_name, ids in self.object_instances.items():
            for obj_id in ids:
                od = self.object_data.get(obj_id, {})
                rows.append({
                    "object_id":      obj_id,
                    "class":          class_name,
                    "frame_count":    od.get('frame_count', 0),
                    "first_frame":    od.get('first_frame', 0),
                    "last_frame":     od.get('last_frame', 0),
                    "avg_confidence": round(od.get('avg_confidence', 0.0), 4),
                    "is_active":      od.get('is_active', False),
                })
        return {"objects": rows, "summary": self.generate_summary()}

    def reset(self):
        """Reset tracker state but preserve next_id for global ID persistence."""
        self.object_instances = defaultdict(list)
        self.object_data      = {}
        self.active_objects   = {}
        self.active_counts.clear()
        self._summary_dirty   = True
        self._cached_summary  = None
        # next_id intentionally NOT reset
