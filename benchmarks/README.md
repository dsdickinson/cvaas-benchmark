# CVaaS Benchmarking Suite

This directory contains the comparative benchmarking framework for evaluating Computer Vision as a Service (CVaaS) traffic inference pipelines across **Python** and **Rust** engines, supporting both **HTTP** (port 8000) and **gRPC** (port 8001) Triton protocols.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Video Source (RTSP / MP4) |
                      +--------------+--------------+
                                     |
                                     v
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
       +-------------------+                   +-------------------+
       |   Python Engine   |                   |    Rust Engine    |
       |  (python/src/)    |                   |   (rust/src/)     |
       +---------+---------+                   +---------+---------+
                 |                                       |
                 | (HTTP v2 or gRPC)                     | (HTTP v2 or gRPC)
                 +-------------------+-------------------+
                                     |
                                     v
                       +---------------------------+
                       |  Triton Inference Server  |
                       | (Object Detection Model)  |
                       +-------------+-------------+
                                     |
                                     v
                       +---------------------------+
                       |    Automated Benchmarks   |
                       |  (benchmarks/results/)    |
                       +---------------------------+
```

---

## Running Benchmarks

### 1. Full Automated Comparative Run
Runs both Python and Rust across both HTTP and gRPC, collects latency percentiles and resource metrics, and produces markdown/CSV reports:

```bash
# Using mock inference (for local/offline verification)
.venv/bin/python benchmarks/run_benchmarks.py --engine both --protocol both --max-frames 100 --mock

# Against a live Triton Inference Server
.venv/bin/python benchmarks/run_benchmarks.py --engine both --protocol both --max-frames 500
```

### 2. Standalone Engine Execution

#### Python Pipeline:
```bash
# Interactive GUI mode
.venv/bin/python python/src/main.py --source video/intersection-night.mp4 --protocol grpc

# Headless benchmark mode
.venv/bin/python python/src/main.py --source video/intersection-night.mp4 --protocol http --headless --max-frames 100
```

#### Rust Pipeline:
```bash
# Interactive GUI mode
rust/target/release/cv-infer-rust --source video/intersection-night.mp4 --protocol grpc

# Headless benchmark mode
rust/target/release/cv-infer-rust --source video/intersection-night.mp4 --protocol http --headless --max-frames 100
```

---

## Metrics Collected

For every frame processed, the harness records:
- **`capture_ms`**: OpenCV frame decoding latency
- **`preprocess_ms`**: Frame resize, color space conversion, and tensor serialization
- **`infer_ms`**: Triton HTTP / gRPC round-trip network & computation latency
- **`postprocess_ms`**: Bounding box extraction, score filtering, and MQTT/JSON packaging
- **`total_ms`**: Total end-to-end frame processing time
- **`fps`**: Sustained frames per second
- **`peak_rss_mb`**: Peak resident memory footprint of the engine process

---

## Benchmark Results

Latest run against `video/intersection-night.mp4` (100 frames per run):

| Engine | Protocol | Frames | Time (s) | FPS | Total Mean (ms) | Total p50 (ms) | Total p95 (ms) | Total p99 (ms) | Infer Mean (ms) | Infer p50 (ms) | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Python | HTTP | 100 | 0.80 | 124.65 | 8.02 | 7.62 | 8.55 | 9.34 | 6.13 | 6.26 | 380.95 |
| Python | GRPC | 100 | 0.82 | 122.24 | 8.18 | 7.85 | 8.75 | 9.32 | 6.16 | 6.26 | 414.39 |
| Rust   | HTTP | 100 | 0.84 | 118.94 | 8.21 | 7.85 | 8.24 | 8.34 | 6.74 | 6.78 | 269.66 |
| Rust   | GRPC | 100 | 0.84 | 119.40 | 8.17 | 7.93 | 8.33 | 8.56 | 6.68 | 6.78 | 261.78 |

All detailed runs and individual per-frame data are saved in [`benchmarks/results/`](./results/).
