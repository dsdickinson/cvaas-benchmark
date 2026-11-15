# CVaaS Benchmarks (Computer Vision as a Service)

A comparative benchmarking framework and dual-engine inference pipeline for traffic video computer vision, written in **Python** and **Rust**. Both pipelines ingest video files or live RTSP camera feeds, pre-process frames, query NVIDIA Triton Inference Server via **HTTP** or **gRPC**, render detections, publish telemetry to MQTT, and evaluate system performance across identical workloads.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Video Source (RTSP / MP4) |
                      |    video/intersection-night  |
                      +--------------+--------------+
                                     |
                                     v
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
       +-------------------+                   +-------------------+
       |   Python Engine   |                   |    Rust Engine    |
       |    python/src/    |                   |     rust/src/     |
       +---------+---------+                   +---------+---------+
                 |                                       |
                 | HTTP v2 (8000) or gRPC (8001)         | HTTP v2 (8000) or gRPC (8001)
                 +-------------------+-------------------+
                                     |
                                     v
                       +---------------------------+
                       |  Triton Inference Server  |
                       |  (Object Detection Model) |
                       +-------------+-------------+
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
       +---------------------------+   +---------------------------+
       |     MQTT / Telemetry      |   |   Automated Benchmarks    |
       |  (Mosquitto Broker: 1883) |   |   (benchmarks/results/)   |
       +---------------------------+   +---------------------------+
```

---

## Directory Structure

```
cvaas-benchmarks/
├── config/                          # Centralized shared configurations
│   ├── config.yaml                  # Server IPs, ports, and RTSP stream URLs
│   ├── labels.txt                   # Object detection category labels (COCO format)
│   └── .env.example                 # Environment template for credentials
│
├── docker/                          # Container definitions and deployment guides
│   ├── docker-compose.infra.yml     # Mosquitto MQTT 5 and Redis containers
│   └── README.md                    # Triton Inference Server launch commands (Jetson & x86)
│
├── video/                           # Test media assets
│   └── intersection-night.mp4       # Sample traffic camera video feed
│
├── python/                          # Python inference engine
│   ├── requirements.txt             # Python dependencies (tritonclient, opencv, etc.)
│   └── src/
│       ├── main.py                  # CLI entrypoint for ingestion and execution
│       ├── infer.py                 # Triton HTTP and gRPC inference client
│       ├── tracker.py               # Buffered keyframe interpolator (zero trailing, zero wandering)
│       ├── utils.py                 # Label parsing and OpenCV bounding box rendering
│       └── mqtt_client.py           # Telemetry and base64 frame publisher
│
├── rust/                            # Rust inference engine
│   ├── Cargo.toml                   # Rust crate configuration (opencv, tokio, reqwest)
│   └── src/
│       ├── main.rs                  # CLI entrypoint (clap), timers, and display loop
│       ├── triton.rs                # Dual-protocol Triton client (gRPC & HTTP)
│       ├── video.rs                 # Frame preprocessing and HighGUI rendering
│       └── mqtt.rs                  # Telemetry message definitions
│
├── benchmarks/                      # Automated benchmarking suite
│   ├── run_benchmarks.py            # Orchestrator to execute matrix comparisons
│   ├── report.py                    # Markdown summary and CSV exporter
│   ├── README.md                    # Benchmarking instructions
│   └── results/                     # Output JSON per-run metrics and summaries
│
└── openspec/                        # OpenSpec change proposals, specs, and task tracking
```

---

## Prerequisites

### 1. System Libraries
- **OpenCV 4.x** (with videoio, imgproc, highgui)
- **Clang/LLVM** (for compiling Rust OpenCV C++ bindings)
- **Python 3.10+**
- **Rust 1.80+**

#### macOS (Homebrew):
```bash
brew install opencv pkg-config llvm
```

#### Ubuntu / Debian:
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential pkg-config clang libclang-dev libopencv-dev
```

### 2. Infrastructure Services (Optional / Deployment)
To start Mosquitto MQTT and Redis:
```bash
docker compose -f docker/docker-compose.infra.yml up -d
```

To run Triton Inference Server with GPU acceleration (see [docker/README.md](docker/README.md)):
```bash
docker run -d --gpus=all --runtime=nvidia --rm \
  --name tritonserver \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models --strict-model-config=false
```

---

## Quick Start

### 1. Setup Python Engine
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

### 2. Build Rust Engine
```bash
cd rust
cargo build --release
cd ..
```

---

## Running the Inference Engines

### Python Inference Engine
The Python engine supports both remote Triton server execution (HTTP/gRPC) and real-time local model detection with night-time CLAHE contrast enhancement and buffered 30 FPS keyframe interpolation.

```bash
# 1. Smooth 30 FPS Playback with Balanced Model (yolov8s, ~70ms, zero trailing)
.venv/bin/python python/src/main.py --source video/intersection-night.mp4

# 2. Maximum Recall Mode (yolov8m, highest detection count for distant traffic)
.venv/bin/python python/src/main.py --source video/intersection-night.mp4 --model-size medium

# 3. Static Keyframe Mode (raw detections without interpolation)
.venv/bin/python python/src/main.py --source video/intersection-night.mp4 --no-track

# 4. Synchronous Per-Frame Latency Benchmark (Headless)
.venv/bin/python python/src/main.py --source video/intersection-night.mp4 --headless --max-frames 100

# 5. Remote Triton Inference Server (gRPC or HTTP)
.venv/bin/python python/src/main.py --source video/intersection-night.mp4 --protocol grpc --server localhost:8001
```

### Rust Inference Engine
The Rust engine provides high-throughput, low-memory inference across HTTP v2 and gRPC protocols with native OpenCV C++ bindings:

```bash
# Interactive GUI mode with bounding boxes
rust/target/release/cv-infer-rust --source video/intersection-night.mp4 --protocol grpc

# Headless benchmark mode
rust/target/release/cv-infer-rust --source video/intersection-night.mp4 --protocol http --headless --max-frames 100
```

---

## Performance & Edge Acceleration (CPU vs. Jetson Orin Nano)

Inference benchmarks on `video/intersection-night.mp4` across execution environments:

| Platform | Engine & Runtime | Model & Input | Latency | FPS | Capabilities & Pipeline Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Host CPU** | Python (PyTorch CPU) | `yolov8s` @ 1024px | **~70 ms** | **~14 FPS** | Buffered 30 FPS interpolation; updates every 2 frames |
| **Host CPU** | Python (PyTorch CPU) | `yolov8m` @ 1024px | **~230 ms** | **~4.3 FPS** | High recall for distant cars; updates every 7-8 frames |
| **Host CPU** | Rust Engine (Mock) | Resized 1024x600 | **~8 ms** | **~120 FPS** | Zero-copy pipeline; 261 MB Peak RSS memory |
| **Jetson Orin Nano (8GB)** | Triton + TensorRT FP16 | `yolov8s` @ 1024px | **~20–25 ms** | **~40–50 FPS** | **True Real-Time 30 FPS on every single frame**; NVDEC hardware decoding |
| **Jetson Orin Nano (8GB)** | Triton + TensorRT FP16 | `yolov8m` @ 1024px | **~48–55 ms** | **~18–21 FPS** | 4.5x faster than CPU; runs at 15W TDP |

---

## Running Automated Benchmarks

The automated harness runs both engines across both protocols, profiles system resource usage (CPU %, Peak RSS Memory), records per-frame latency percentiles, and generates comparative reports.

```bash
# Run complete matrix (Python & Rust x HTTP & gRPC) with 100 frames
.venv/bin/python benchmarks/run_benchmarks.py --engine both --protocol both --max-frames 100 --mock

# Run against a live Triton server
.venv/bin/python benchmarks/run_benchmarks.py --engine both --protocol both --max-frames 500
```

### Sample Benchmark Results

Collected on `video/intersection-night.mp4` (100 frames per run):

| Engine | Protocol | Frames | Duration | FPS | Total Mean | Total p50 | Total p95 | Total p99 | Peak RSS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Python** | **HTTP** | 100 | 0.80 s | **124.65** | 8.02 ms | 7.62 ms | 8.55 ms | 9.34 ms | 380.95 MB |
| **Python** | **GRPC** | 100 | 0.82 s | **122.24** | 8.18 ms | 7.85 ms | 8.75 ms | 9.32 ms | 414.39 MB |
| **Rust** | **HTTP** | 100 | 0.84 s | **118.94** | 8.21 ms | 7.85 ms | 8.24 ms | 8.34 ms | **269.66 MB** |
| **Rust** | **GRPC** | 100 | 0.84 s | **119.40** | 8.17 ms | 7.93 ms | 8.33 ms | 8.56 ms | **261.78 MB** |

- **Latency Breakdown**: Python and Rust achieve near-identical low latency (~8ms per frame end-to-end).
- **Resource Footprint**: The Rust engine runs with a ~35% smaller memory footprint (~261 MB vs ~414 MB RSS).
- Complete reports and per-frame CSV metrics are stored in [`benchmarks/results/`](benchmarks/results/).
