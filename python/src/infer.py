import os
import sys
import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2 as cv

os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"

try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    from ultralytics import YOLO
    LOCAL_YOLO_AVAILABLE = True
except ImportError:
    LOCAL_YOLO_AVAILABLE = False

# Standard COCO traffic & vehicle classes:
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 7]


def enhance_night_frame(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the luminance channel to reveal vehicle contours and road markings in dark scenes.
    """
    lab = cv.cvtColor(frame_rgb, cv.COLOR_RGB2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    return cv.cvtColor(cv.merge([l_enhanced, a, b]), cv.COLOR_LAB2RGB)


class TritonInferenceClient:
    def __init__(
        self,
        server_url: str,
        protocol: str = "http",
        model_name: str = "object_detection",
        model_version: str = "1",
        mock_mode: bool = False,
        local_mode: bool = False,
        conf_thresh: float = 0.15,
        imgsz: int = 1280,
        enhance_night: bool = True,
        traffic_only: bool = True,
        model_size: str = "medium",
    ):
        self.server_url = server_url
        self.protocol = protocol.lower()
        self.model_name = model_name
        self.model_version = model_version
        self.mock_mode = mock_mode
        self.local_mode = local_mode
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.enhance_night = enhance_night
        self.traffic_only = traffic_only
        self.model_size = model_size.lower()
        self.client = None
        self.local_model = None

        if not self.mock_mode and not self.local_mode:
            self._init_client()

        # If Triton was not reachable or local mode was requested, initialize local detector
        if (self.mock_mode or self.local_mode) and LOCAL_YOLO_AVAILABLE:
            self._init_local_detector()

    def _init_local_detector(self):
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        
        if self.model_size == "small":
            candidates = ["yolov8s.pt", "yolov8m.pt", "yolov8n.pt"]
        elif self.model_size == "nano":
            candidates = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
        else:
            candidates = ["yolov8m.pt", "yolov8s.pt", "yolov8n.pt"]

        chosen_path = None
        for candidate in candidates:
            p = os.path.join(models_dir, candidate)
            if os.path.exists(p):
                chosen_path = p
                break

        if chosen_path:
            try:
                print(f"Loading high-accuracy local model: {os.path.basename(chosen_path)} (imgsz={self.imgsz})...")
                self.local_model = YOLO(chosen_path)
                print(f"Local detector loaded. Night enhancement: {self.enhance_night} | Traffic filter: {self.traffic_only}")
                self.local_mode = True
                self.mock_mode = False
            except Exception as e:
                print(f"Warning: Failed to load local YOLO model: {e}")

    def _init_client(self):
        if not TRITON_AVAILABLE:
            print("Notice: tritonclient library not installed. Falling back to local model.")
            self.local_mode = True
            return

        try:
            if self.protocol == "http":
                self.client = httpclient.InferenceServerClient(url=self.server_url)
                if not self.client.is_server_ready():
                    print(f"Notice: Triton HTTP server at {self.server_url} is not reachable.")
                    print("Falling back to real-time local model detection.")
                    self.local_mode = True
            elif self.protocol == "grpc":
                self.client = grpcclient.InferenceServerClient(url=self.server_url)
                if not self.client.is_server_ready():
                    print(f"Notice: Triton gRPC server at {self.server_url} is not reachable.")
                    print("Falling back to real-time local model detection.")
                    self.local_mode = True
            else:
                raise ValueError(f"Unsupported protocol: {self.protocol}. Use 'http' or 'grpc'.")
        except Exception as e:
            print(f"Notice: Connection to Triton ({self.protocol}://{self.server_url}) failed: {e}.")
            print("Falling back to real-time local model detection.")
            self.local_mode = True

    def infer(self, image_np: np.ndarray, labels_dict: Optional[Dict[int, str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        """
        Executes inference against remote Triton, local YOLO model, or synthetic mock.
        Returns:
            boxes: [N, 4] (ymin, xmin, ymax, xmax) normalized [0, 1]
            classes: [N]
            scores: [N]
            num_detections: int
            infer_latency_sec: float
        """
        # Ensure 4D tensor (1, H, W, C)
        if len(image_np.shape) == 3:
            input_tensor = np.expand_dims(image_np, axis=0)
            raw_rgb = image_np
        else:
            input_tensor = image_np
            raw_rgb = image_np[0]

        t_start = time.perf_counter()

        # 1. Local real-time high-accuracy YOLO detector
        if self.local_mode and self.local_model is not None:
            # Apply night enhancement if enabled
            feed_rgb = enhance_night_frame(raw_rgb) if self.enhance_night else raw_rgb
            target_classes = TRAFFIC_CLASSES if self.traffic_only else None

            results = self.local_model(
                feed_rgb,
                imgsz=self.imgsz,
                conf=self.conf_thresh,
                classes=target_classes,
                verbose=False,
            )
            t_infer = time.perf_counter() - t_start

            boxes_obj = results[0].boxes
            if len(boxes_obj) == 0:
                return (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0,), dtype=np.float32),
                    0,
                    t_infer,
                )

            xyxyn = boxes_obj.xyxyn.cpu().numpy()
            # Convert [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax]
            boxes = np.stack([xyxyn[:, 1], xyxyn[:, 0], xyxyn[:, 3], xyxyn[:, 2]], axis=-1).astype(np.float32)
            classes = boxes_obj.cls.cpu().numpy().astype(np.int32)
            scores = boxes_obj.conf.cpu().numpy().astype(np.float32)

            # Update labels dict with model names if available
            if labels_dict is not None and hasattr(self.local_model, "names"):
                for cls_id, name in self.local_model.names.items():
                    labels_dict[cls_id] = name

            return boxes, classes, scores, len(boxes), t_infer

        # 2. Synthetic mock mode (only if local detector unavailable)
        if self.mock_mode:
            time.sleep(0.005)
            t_infer = time.perf_counter() - t_start
            num_dets = 2
            boxes = np.array([[0.2, 0.3, 0.6, 0.7], [0.5, 0.1, 0.8, 0.4]], dtype=np.float32)
            classes = np.array([3, 4], dtype=np.int32)
            scores = np.array([0.88, 0.75], dtype=np.float32)
            return boxes, classes, scores, num_dets, t_infer

        # 3. Real Remote Triton inference
        if self.protocol == "http":
            inputs = [httpclient.InferInput("image_tensor", input_tensor.shape, "UINT8")]
            inputs[0].set_data_from_numpy(input_tensor)
            outputs = [
                httpclient.InferRequestedOutput("num_detections", binary_data=False),
                httpclient.InferRequestedOutput("detection_classes", binary_data=False),
                httpclient.InferRequestedOutput("detection_boxes", binary_data=False),
                httpclient.InferRequestedOutput("detection_scores", binary_data=False),
            ]
            response = self.client.infer(self.model_name, inputs=inputs, outputs=outputs)
        else:  # grpc
            inputs = [grpcclient.InferInput("image_tensor", input_tensor.shape, "UINT8")]
            inputs[0].set_data_from_numpy(input_tensor)
            outputs = [
                grpcclient.InferRequestedOutput("num_detections"),
                grpcclient.InferRequestedOutput("detection_classes"),
                grpcclient.InferRequestedOutput("detection_boxes"),
                grpcclient.InferRequestedOutput("detection_scores"),
            ]
            response = self.client.infer(self.model_name, inputs=inputs, outputs=outputs)

        t_infer = time.perf_counter() - t_start

        num_detections = int(response.as_numpy("num_detections")[0])
        detection_classes = response.as_numpy("detection_classes")[0].astype(np.int32)
        detection_boxes = response.as_numpy("detection_boxes")[0].astype(np.float32)
        detection_scores = response.as_numpy("detection_scores")[0].astype(np.float32)

        return detection_boxes, detection_classes, detection_scores, num_detections, t_infer
