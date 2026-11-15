import numpy as np
from typing import Tuple, Dict, Any


def compute_iou(boxA, boxB) -> float:
    """Computes IoU for boxes in [ymin, xmin, ymax, xmax] normalized format."""
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


class RealtimeMotionTracker:
    """
    Robust Centroid & Velocity Extrapolation Tracker for Traffic Inference.
    
    Eliminates trailing bounding box lag caused by asynchronous inference latency
    without using fragile optical flow (which wanders on night headlight glare & asphalt).
    
    1. Tracks verified YOLO detections across frames.
    2. Estimates linear trajectory velocities (vy, vx) clamped to realistic physical limits.
    3. Extrapolates bounding boxes forward to bridge the inference latency gap.
    4. Automatically locks stationary vehicles (stopped at signals/parked) with zero drift.
    5. Purges stale tracks immediately to eliminate phantom wandering boxes.
    """
    def __init__(self, max_speed_norm: float = 0.005, iou_thresh: float = 0.10):
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_id: int = 0
        self.max_speed = max_speed_norm
        self.iou_thresh = iou_thresh

    def update_detections(
        self,
        det_boxes: np.ndarray,
        det_classes: np.ndarray,
        det_scores: np.ndarray,
        sent_frame_id: int,
        current_frame_id: int,
    ):
        """
        Integrates fresh YOLO detections and calculates velocities.
        Extrapolates box positions to current_frame_id based on estimated velocity.
        """
        new_tracks = {}
        unmatched_dets = list(range(len(det_boxes)))

        # Match new detections to existing tracks via IoU
        for tid, tr in self.tracks.items():
            t_box = tr["box"]
            best_iou = 0.0
            best_d = -1
            for d_idx in unmatched_dets:
                d_box = det_boxes[d_idx]
                iou = compute_iou(t_box, d_box)
                if iou > best_iou:
                    best_iou = iou
                    best_d = d_idx

            if best_iou >= self.iou_thresh and best_d != -1:
                d_box = det_boxes[best_d]
                delta_f = max(current_frame_id - tr["last_frame"], 1)

                cy_old = (t_box[0] + t_box[2]) * 0.5
                cx_old = (t_box[1] + t_box[3]) * 0.5
                cy_new = (d_box[0] + d_box[2]) * 0.5
                cx_new = (d_box[1] + d_box[3]) * 0.5

                # Velocity per frame clamped to physical traffic speeds
                vy = float(np.clip((cy_new - cy_old) / delta_f, -self.max_speed, self.max_speed))
                vx = float(np.clip((cx_new - cx_old) / delta_f, -self.max_speed, self.max_speed))

                # Stationary deadband (vehicles waiting at lights or stopped)
                if abs(vy) < 0.0003:
                    vy = 0.0
                if abs(vx) < 0.0003:
                    vx = 0.0

                # Exponential smoothing of velocity
                prev_vy, prev_vx = tr["vel"]
                vy = 0.6 * vy + 0.4 * prev_vy
                vx = 0.6 * vx + 0.4 * prev_vx

                # Extrapolate box forward to close the background inference latency gap
                lead_frames = max(current_frame_id - sent_frame_id, 0)
                extrapolated_box = [
                    float(np.clip(d_box[0] + vy * lead_frames, 0.0, 1.0)),
                    float(np.clip(d_box[1] + vx * lead_frames, 0.0, 1.0)),
                    float(np.clip(d_box[2] + vy * lead_frames, 0.0, 1.0)),
                    float(np.clip(d_box[3] + vx * lead_frames, 0.0, 1.0)),
                ]

                new_tracks[tid] = {
                    "box": extrapolated_box,
                    "vel": [vy, vx],
                    "cls": int(det_classes[best_d]),
                    "score": float(det_scores[best_d]),
                    "last_frame": current_frame_id,
                }
                unmatched_dets.remove(best_d)
            # Notice: Unmatched old tracks are dropped immediately to avoid ghost boxes

        # Register newly appeared vehicles
        for d_idx in unmatched_dets:
            new_tracks[self.next_id] = {
                "box": [float(x) for x in det_boxes[d_idx]],
                "vel": [0.0, 0.0],
                "cls": int(det_classes[d_idx]),
                "score": float(det_scores[d_idx]),
                "last_frame": current_frame_id,
            }
            self.next_id += 1

        self.tracks = new_tracks

    def step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Advances all moving vehicles by their velocity for 1 video frame.
        Runs at native 30 FPS (< 0.05ms execution).
        """
        for tid, tr in self.tracks.items():
            vy, vx = tr["vel"]
            if vy != 0.0 or vx != 0.0:
                b = tr["box"]
                tr["box"] = [
                    float(np.clip(b[0] + vy, 0.0, 1.0)),
                    float(np.clip(b[1] + vx, 0.0, 1.0)),
                    float(np.clip(b[2] + vy, 0.0, 1.0)),
                    float(np.clip(b[3] + vx, 0.0, 1.0)),
                ]

        if not self.tracks:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
                0,
            )

        boxes = np.array([tr["box"] for tr in self.tracks.values()], dtype=np.float32)
        classes = np.array([tr["cls"] for tr in self.tracks.values()], dtype=np.int32)
        scores = np.array([tr["score"] for tr in self.tracks.values()], dtype=np.float32)
        return boxes, classes, scores, len(boxes)
