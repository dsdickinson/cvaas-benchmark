import time
from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class TritonInferenceClient:
    def __init__(
        self,
        server_url: str,
        protocol: str = "http",
        model_name: str = "object_detection",
        model_version: str = "1",
        mock_mode: bool = False,
    ):
        self.server_url = server_url
        self.protocol = protocol.lower()
        self.model_name = model_name
        self.model_version = model_version
        self.mock_mode = mock_mode
        self.client = None

        if not self.mock_mode:
            self._init_client()

    def _init_client(self):
        if not TRITON_AVAILABLE:
            print("Warning: tritonclient library not installed. Falling back to mock mode.")
            self.mock_mode = True
            return

        try:
            if self.protocol == "http":
                # Expects e.g. localhost:8000
                self.client = httpclient.InferenceServerClient(url=self.server_url)
                if not self.client.is_server_ready():
                    print(f"Notice: Triton HTTP server at {self.server_url} is not ready. Using mock mode.")
                    self.mock_mode = True
            elif self.protocol == "grpc":
                # Expects e.g. localhost:8001
                self.client = grpcclient.InferenceServerClient(url=self.server_url)
                if not self.client.is_server_ready():
                    print(f"Notice: Triton gRPC server at {self.server_url} is not ready. Using mock mode.")
                    self.mock_mode = True
            else:
                raise ValueError(f"Unsupported protocol: {self.protocol}. Use 'http' or 'grpc'.")
        except Exception as e:
            print(f"Notice: Connection to Triton ({self.protocol}://{self.server_url}) failed: {e}. Falling back to mock mode.")
            self.mock_mode = True

    def infer(self, image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Executes inference against Triton (or mock mode).
        Returns:
            boxes: [N, 4] (ymin, xmin, ymax, xmax)
            classes: [N]
            scores: [N]
            num_detections: int
            infer_latency_sec: float
        """
        # Ensure 4D tensor (1, H, W, C)
        if len(image_np.shape) == 3:
            input_tensor = np.expand_dims(image_np, axis=0)
        else:
            input_tensor = image_np

        t_start = time.perf_counter()

        if self.mock_mode:
            # Simulate network round-trip and provide synthetic detections
            time.sleep(0.005)  # 5ms mock inference
            t_infer = time.perf_counter() - t_start
            num_dets = 2
            boxes = np.array([[0.2, 0.3, 0.6, 0.7], [0.5, 0.1, 0.8, 0.4]], dtype=np.float32)
            classes = np.array([3, 4], dtype=np.int32)  # car, motorcycle
            scores = np.array([0.88, 0.75], dtype=np.float32)
            return boxes, classes, scores, num_dets, t_infer

        # Real Triton inference
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
