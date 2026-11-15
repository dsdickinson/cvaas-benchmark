# Python Computer Vision Inference Engine

This directory contains the Python implementation of the CVaaS traffic video inferencing pipeline. It ingests video feeds from local video files (e.g., MP4) or RTSP camera streams, pre-processes frames into model input tensors, submits inference requests to an NVIDIA Triton Inference Server via **HTTP** or **gRPC**, overlays detected bounding boxes and class labels, and optionally streams telemetry and frames to an MQTT broker.

---

## Directory Layout

```
python/
├── requirements.txt         # Core dependencies: opencv-python, tritonclient[all], numpy, paho-mqtt
├── README.md                # This documentation file
└── src/
    ├── main.py              # CLI entrypoint, frame capture loop, and benchmark timer
    ├── infer.py             # TritonInferenceClient supporting HTTP v2 and gRPC
    ├── utils.py             # Label map parser (labels.txt) and bounding box drawing routines
    └── mqtt_client.py       # Asynchronous MQTT publisher for telemetry and encoded frames
```

---

## Requirements & Setup

1. **Python Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r python/requirements.txt
   ```

Key packages included:
- `tritonclient[all]`: Official NVIDIA client supporting HTTP v2 and gRPC interfaces
- `opencv-python`: Video decoding, frame resizing, and HighGUI visualization
- `numpy`: Tensor transformations and array slicing
- `paho-mqtt`: Telemetry broadcasting to MQTT brokers
- `pyyaml` & `python-dotenv`: Configuration file and credential management

---

## Architecture & How It Works

1. **Ingestion (`src/main.py`)**:
   - Opens the video source (file path or RTSP stream URL) using `cv2.VideoCapture`.
   - Records high-resolution frame capture timestamps.
2. **Preprocessing (`src/main.py`)**:
   - Resizes frames to the target tensor dimensions (default: `1024x600`).
   - Converts BGR frames to RGB format expected by computer vision models.
   - Expands dimensions to form batch tensors `(1, H, W, 3)` with `UINT8` dtype.
3. **Inference Client (`src/infer.py`)**:
   - Manages connection pools to Triton Inference Server.
   - **HTTP**: Submits requests via `tritonclient.http.InferenceServerClient` (default port `8000`).
   - **gRPC**: Submits requests via `tritonclient.grpc.InferenceServerClient` (default port `8001`).
   - Extracts four output tensors: `num_detections`, `detection_classes`, `detection_boxes`, `detection_scores`.
   - Includes a `--mock` mode to simulate inference latency during offline benchmarking or CI.
4. **Postprocessing & Output (`src/utils.py` & `src/mqtt_client.py`)**:
   - **Interactive Mode**: Renders bounding boxes and label text on the video frame using OpenCV.
   - **Headless Mode (`--headless`)**: Bypasses GUI rendering and logs detailed per-frame timing metrics.
   - **MQTT Telemetry (`--mqtt-enabled`)**: Publishes detected bounding boxes and scores to `gg/video/inf/<stream_id>`.

---

## CLI Options & Usage

```bash
python python/src/main.py [OPTIONS]
```

### Options:
- `--source`, `-s`: Path to video file or RTSP stream URL (default: `video/intersection-night.mp4`)
- `--protocol`, `-p`: `http` or `grpc` (default: `http`)
- `--server`: Triton server address (`host:port`). Defaults to `localhost:8000` for HTTP or `localhost:8001` for gRPC.
- `--model`, `-m`: Model name deployed on Triton (default: `object_detection`)
- `--local`: Run real-time local object detection model (`models/yolov8m.pt`) without needing a remote Triton server
- `--model-size`: Local model capacity (`medium` for max accuracy, `small` for higher FPS, or `nano`)
- `--sync`: Run in blocking synchronous mode instead of smooth native 30 FPS async playback
- `--no-track`: Disable buffered keyframe interpolation (display static keyframe detections)
- `--conf`: Confidence score threshold for detections (default: `0.15`)
- `--imgsz`: Input image resolution for local detector (default: `1280`, preserves detail on 1080p aerial traffic)
- `--no-enhance`: Disable night-time CLAHE contrast enhancement
- `--all-classes`: Detect all 80 COCO classes instead of filtering strictly to traffic (vehicles/pedestrians)
- `--headless`: Run without GUI window creation (essential for unbiased benchmarking)
- `--max-frames`: Limit number of frames processed (`0` for unlimited)
- `--mqtt-enabled`: Publish detection coordinates to MQTT
- `--mqtt-broker`: MQTT broker address (default: `localhost`)
- `--metrics-out`: Path to export per-frame latency records and summary stats as JSON
- `--mock`: Run with synthetic inference (no live Triton server required)

---

## Execution Examples

### 1. Interactive Video Playback with Bounding Boxes
```bash
# Using gRPC protocol against local Triton
python python/src/main.py --source video/intersection-night.mp4 --protocol grpc

# Using HTTP protocol against a remote Triton host
python python/src/main.py --source video/intersection-night.mp4 --protocol http --server 192.168.1.80:8000
```

### 2. Live RTSP Camera Ingestion
```bash
python python/src/main.py --source "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV001.stream" --protocol grpc
```

### 3. Headless Benchmark Run with Metrics Output
```bash
python python/src/main.py \
  --source video/intersection-night.mp4 \
  --protocol grpc \
  --headless \
  --max-frames 200 \
  --metrics-out benchmarks/results/python_grpc_sample.json
```

---

## Metrics Emitted

When running in benchmark mode, the pipeline outputs:
- **`capture_ms`**: OpenCV frame decoding latency.
- **`preprocess_ms`**: Frame resize and color conversion time.
- **`infer_ms`**: Triton network round-trip and remote model execution time.
- **`postprocess_ms`**: Detection parsing, label mapping, and display/publishing time.
- **`total_ms`**: End-to-end frame turnaround duration.
- **Percentiles**: Aggregate `mean`, `p50`, `p90`, `p95`, and `p99` latencies and average `FPS`.
