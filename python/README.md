# Python Computer Vision Inference Engine

This directory contains the Python implementation of the CVaaS traffic video inferencing pipeline. It ingests video feeds from local video files (e.g., MP4) or RTSP camera streams, pre-processes frames into model input tensors, submits inference requests to an NVIDIA Triton Inference Server via **HTTP** or **gRPC**, overlays detected bounding boxes and class labels, and optionally streams telemetry and frames to an MQTT broker.

---

## Directory Layout

```
python/
├── requirements.txt         # Core dependencies: opencv-python, tritonclient[all], ultralytics, numpy, paho-mqtt
├── README.md                # This documentation file
└── src/
    ├── main.py              # CLI entrypoint, async display loop, and benchmark timer
    ├── infer.py             # Dual Triton client (HTTP/gRPC) and local YOLO fallback
    ├── tracker.py           # BufferedVideoInterpolator for butter-smooth 30 FPS playback
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
   - Decoupled display architecture buffers frames to deliver smooth native 30 FPS playback.
2. **Night-Time Contrast Enhancement (`src/infer.py`)**:
   - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the luminance channel in LAB color space.
   - Reveals vehicle contours, motorcycles, and road markings obscured by night darkness and specular headlight glare.
3. **Dual Inference Mode (`src/infer.py`)**:
   - **Triton Remote Client**: Connects via HTTP v2 (`localhost:8000`) or gRPC (`localhost:8001`) to query remote models.
   - **Local YOLO Auto-Fallback**: If Triton is unreachable or `--local` is specified, seamlessly falls back to local PyTorch models (`models/yolov8s.pt`, `yolov8m.pt`, or `yolov8n.pt`).
   - **Traffic Class Filtering**: Automatically filters detections to traffic categories (cars, motorcycles, buses, trucks, persons, bicycles) with customizable confidence (`--conf 0.15`).
4. **Buffered Keyframe Interpolator (`src/tracker.py`)**:
   - Solves both video choppiness, trailing lag, and wandering boxes.
   - Runs a background worker thread inferring keyframes while the display thread renders at native 30 FPS.
   - Buffers ~180ms (6 frames) and linearly interpolates box positions between confirmed keyframes ($K_0 \to K_1$).
   - **Zero wandering**: Every box is strictly anchored between real YOLO detections; no drifting into buildings or sidewalks.
   - **Zero trailing**: Boxes arrive at the exact vehicle position on time.
   - **Rock-solid stopped vehicles**: Cars waiting at red lights stay 100% stationary.
5. **Output & Telemetry (`src/utils.py` & `src/mqtt_client.py`)**:
   - **Interactive Mode**: HighGUI rendering with bounding boxes, confidence tags, and live FPS metrics.
   - **Headless Mode (`--headless`)**: Bypasses GUI rendering for precise latency benchmarking.
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
- `--local`: Force real-time local YOLO model instead of remote Triton server
- `--model-size`: Local model capacity:
  - `small` *(default)*: Sweet spot — ~70ms latency, updates every 2 frames, zero lag.
  - `medium`: Maximum detection recall for distant aerial objects (~230ms latency).
  - `nano`: Ultra-fast (~40ms latency) for high-FPS edge scenarios.
- `--sync`: Run in blocking synchronous mode instead of smooth native 30 FPS async playback
- `--no-track`: Disable buffered keyframe interpolation (display static keyframe detections)
- `--device`: Target execution device: `auto`, `gpu`, `cuda`, `mps`, or `cpu` (default: `auto`)
- `--gpu`: Enable/force GPU processing (routes local inference to CUDA/MPS and verifies remote Triton GPU execution)
- `--cpu`: Force CPU-only execution (disables GPU acceleration locally and targets remote CPU instances)
- `--conf`: Confidence score threshold for detections (default: `0.15`)
- `--imgsz`: Input image resolution for local detector (default: `1024`, preserves detail on 1080p aerial traffic)
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

### 1. Interactive 30 FPS Playback with Smooth Tracking (Default)
```bash
# Smooth 30 FPS video playback with buffered keyframe interpolation
python python/src/main.py --source video/intersection-night.mp4

# Test with maximum recall model
python python/src/main.py --source video/intersection-night.mp4 --model-size medium

# Fast edge mode with nano model
python python/src/main.py --source video/intersection-night.mp4 --model-size nano
```

### 2. Static Keyframe Mode (No Interpolation)
```bash
python python/src/main.py --source video/intersection-night.mp4 --no-track
```

### 3. Remote GPU vs. CPU Inference (Triton Server)
```bash
# Query remote Triton GPU server via gRPC
python python/src/main.py --source video/intersection-night.mp4 --protocol grpc --server 192.168.1.80:8001 --gpu

# Query remote Triton GPU server via HTTP
python python/src/main.py --source video/intersection-night.mp4 --protocol http --server 192.168.1.80:8000 --gpu

# Query remote Triton CPU server
python python/src/main.py --source video/intersection-night.mp4 --protocol grpc --server 192.168.1.80:8001 --cpu
```

### 4. Live RTSP Camera Ingestion
```bash
python python/src/main.py --source "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV001.stream" --protocol grpc --gpu
```

### 5. Headless Benchmark Run with Metrics Output
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
