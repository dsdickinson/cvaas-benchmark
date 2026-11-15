import re
import cv2 as cv
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_label_map(label_path: str) -> Dict[int, str]:
    """
    Parses TensorFlow/proto style labels.txt into a dict of {id: display_name}.
    Example format:
      item {
        name: "/m/01g317"
        id: 1
        display_name: "person"
      }
    """
    labels: Dict[int, str] = {}
    try:
        with open(label_path, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Failed to load label map from {label_path}: {e}")
        return labels

    items = re.findall(r"item\s*\{([^}]+)\}", content)
    for item in items:
        id_match = re.search(r"id:\s*(\d+)", item)
        name_match = re.search(r'display_name:\s*"([^"]+)"', item)
        if not name_match:
            name_match = re.search(r'name:\s*"([^"]+)"', item)
        if id_match and name_match:
            labels[int(id_match.group(1))] = name_match.group(1)

    return labels


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    labels: Dict[int, str],
    min_score_thresh: float = 0.25,
    max_boxes: int = 50,
) -> np.ndarray:
    """
    Draws bounding boxes and labels on an OpenCV image array.
    boxes format: [ymin, xmin, ymax, xmax] normalized to [0, 1].
    """
    h, w = frame.shape[:2]
    num_boxes = min(len(boxes), max_boxes)

    for i in range(num_boxes):
        score = float(scores[i])
        if score < min_score_thresh:
            continue

        cls_id = int(classes[i])
        label = labels.get(cls_id, f"class_{cls_id}")
        caption = f"{label}: {score:.2f}"

        ymin, xmin, ymax, xmax = boxes[i]
        pt1 = (int(xmin * w), int(ymin * h))
        pt2 = (int(xmax * w), int(ymax * h))

        # Draw rectangle
        cv.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw label background and text
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (tw, th), _ = cv.getTextSize(caption, font, font_scale, font_thickness)
        
        bg_pt2 = (pt1[0] + tw + 4, max(pt1[1] - th - 6, 0) + th + 6)
        cv.rectangle(frame, (pt1[0], max(pt1[1] - th - 6, 0)), bg_pt2, (0, 255, 0), -1)
        cv.putText(
            frame,
            caption,
            (pt1[0] + 2, max(pt1[1] - 4, th + 2)),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv.LINE_AA,
        )

    return frame
