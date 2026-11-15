## Context

The repository `cvaas-benchmarks` is being established to evaluate computer vision traffic inferencing across Python and Rust. The user originally created two standalone repositories:
- `cv-infer-py`: A Python prototype utilizing OpenCV, `tritonclient.http`, and experimental scripts for object detection against local/remote Triton servers.
- `cv-infer-rust`: A Rust project using `opencv` bindings, with initial work toward `triton-client` (gRPC) and MQTT.

To properly benchmark these systems, both engines must be consolidated into `cvaas-benchmarks`, achieve feature and protocol parity (supporting both Triton HTTP and gRPC, optional MQTT publishing, and headless operation), and share a standardized benchmarking harness and dataset.

## Goals / Non-Goals

**Goals:**
- Consolidate Python and Rust implementations into clean `python/` and `rust/` directories.
- Provide full support for both HTTP and gRPC Triton inference protocols in both Python and Rust.
- Implement headless mode (`--headless` or `--no-display`) in both engines for unbiased benchmark execution.
- Create shared configuration (`config/`) and test video assets (`video/`).
- Create an automated benchmark suite in `benchmarks/` measuring sustained FPS, latency percentiles (p50, p95, p99), timing breakdown (capture, preprocess, infer, postprocess), and system resource utilization (CPU, memory).
- Maintain working Docker Compose configurations for supporting telemetry and messaging services (MQTT Mosquitto, Redis).

**Non-Goals:**
- Training or fine-tuning computer vision models (we rely on standard deployed models on Triton, such as SSD/YOLO/DenseNet).
- Implementing custom in-engine deep learning runtimes (inference is offloaded to Triton Server).
- Building cloud infrastructure deployments (GCP/AWS deployment manifests are archived/secondary to local benchmarking).

## Decisions

### 1. Directory Structure
We will adopt the following layout:
- `python/`: Production-ready Python inference client, cleanly refactored from `cv-infer-py/gpu/all/00_cap_infer_play.py`.
- `rust/`: Production-ready Rust inference client with clean module separation (`main.rs`, `triton.rs`, `mqtt.rs`, `video.rs`).
- `config/`: Shared configuration files (`config.yaml` for Triton endpoints and camera feeds, `labels.txt` for class names).
- `video/`: Test video streams (`intersection-night.mp4`).
- `docker/`: Docker Compose files for supporting infrastructure (Mosquitto MQTT, Redis).
- `benchmarks/`: Harness scripts (`run_benchmarks.py`, `report.py`) to execute automated comparison runs.

*Alternative considered*: Keeping Python and Rust in separate repos and adding a third benchmarking repo. *Rationale*: A monorepo ensures shared assets, configs, and lockstep benchmarking scripts remain in sync.

### 2. Dual Protocol Implementation (HTTP & gRPC)
- **Python**: Use NVIDIA's official `tritonclient.http` and `tritonclient.grpc` packages, selectable via CLI flag `--protocol [http|grpc]` (ports 8000 and 8001 respectively).
- **Rust**:
  - gRPC: Use the `triton-client` crate (connecting to port 8001 via tonic/tokio).
  - HTTP: Use `reqwest` to issue JSON/binary tensor payloads to Triton's v2 REST API `http://<host>:8000/v2/models/<model_name>/infer`.
- Selectable via `--protocol [http|grpc]` in both engines with identical parameter naming.

*Alternative considered*: Only supporting gRPC. *Rationale*: The user explicitly required supporting both protocols to compare HTTP vs gRPC serialization and latency overhead.

### 3. Pipeline Timing & Headless Benchmarking
Both engines will incorporate high-resolution timers (`perf_counter` in Python, `std::time::Instant` in Rust) measuring:
1. `t_capture`: Time to decode frame from video/RTSP.
2. `t_preprocess`: Time to resize, transpose (HWC -> CHW), and normalize frame to tensor.
3. `t_infer`: Round-trip network inference latency to Triton.
4. `t_postprocess`: Decoding bounding boxes, scores, and class labels.
When `--headless` is active, GUI window creation and rendering (`imshow`) are bypassed, ensuring benchmarks reflect engine performance without display server bottlenecks.

### 4. Benchmarking Suite Design
- A central harness (`benchmarks/run_benchmarks.py`) will orchestrate benchmark runs:
  - Takes parameters: `--iterations`, `--warmup`, `--source`, `--protocol`, `--engine [python|rust|both]`.
  - Runs each engine in headless mode.
  - Records per-frame metrics to JSON/CSV.
  - Aggregates metrics: FPS, latency percentiles (p50, p90, p95, p99), frame jitter, and CPU/memory usage.
  - Generates comparison summary tables and charts.

## Risks / Trade-offs

- **[Risk] Rust Triton client crate stability**: Third-party Rust Triton crates can have version mismatches with latest Triton protobuf definitions.
  - *Mitigation*: Verify `triton-client` gRPC against the target Triton server version; implement clean error handling; fallback or supplement with raw tonic gRPC bindings if needed.
- **[Risk] OpenCV build dependencies in Rust**: Compiling `opencv` crate requires system OpenCV headers and `libclang`.
  - *Mitigation*: Document clear prerequisite setup for macOS (`brew install opencv llvm pkg-config`) and Ubuntu/Debian (`libopencv-dev clang libclang-dev`), and maintain consistent `Cargo.toml`.
- **[Risk] Preprocessing discrepancies between languages**: Differences in how OpenCV Python vs. OpenCV Rust resize or normalize frames could affect accuracy or timing.
  - *Mitigation*: Ensure identical tensor dimension ordering (e.g. 1x3x300x300 float32 or uint8) and document exact transformations in spec.
