## Why

The current Computer Vision as a Service (CVaaS) traffic inference pipelines were developed as two separate exploratory repositories (`cv-infer-py` and `cv-infer-rust`). Consolidating them into this repository (`cvaas-benchmarks`) alongside a shared test dataset and infrastructure allows completing the missing capabilities (Triton gRPC/HTTP client support in both engines, MQTT publishing, headless operation) and running reproducible, apples-to-apples performance benchmarks to evaluate throughput, latency, and resource efficiency.

## What Changes

- **Repository Consolidation**: Migrate and organize Python and Rust inference engines from their respective standalone repositories into `python/` and `rust/`, utilizing shared assets in `video/` and shared configurations in `config/`.
- **Dual-Protocol Triton Support**: Complete both Python and Rust inference engines to support both HTTP (port 8000) and gRPC (port 8001) protocols for Triton Inference Server requests.
- **Inference Pipeline Feature Parity**: Implement frame pre-processing (NCHW/NHWC tensor conversion, normalization), bounding box and class label decoding, optional MQTT telemetry/frame publishing, and headless (`--headless`) execution in both engines.
- **Benchmarking Suite**: Build a unified benchmarking suite in `benchmarks/` capable of measuring sustained FPS, latency percentiles (p50, p95, p99), frame ingestion/pre-processing/post-processing timing, and CPU/GPU/memory resource utilization across Python and Rust.
- **Shared Infrastructure**: Provide Docker Compose definitions for supporting services (Mosquitto MQTT, Redis, and Triton Inference Server).

## Capabilities

### New Capabilities
- `python-inference-pipeline`: Python-based video stream inference pipeline supporting video file/RTSP input, Triton HTTP and gRPC protocols, tensor transformation, detection decoding, and optional MQTT publishing.
- `rust-inference-pipeline`: Rust-based video stream inference engine supporting video file/RTSP input, Triton gRPC and HTTP client execution, frame tensorization, prediction parsing, and optional MQTT publishing.
- `benchmarking-suite`: Automated benchmarking harness and analysis scripts to compare Python vs. Rust pipelines on throughput, latency distributions, resource usage, and frame drop rates.

### Modified Capabilities
<!-- None: No existing specs in openspec/specs/ -->

## Impact

- **Codebases Consolidated**: Imports and code from `/Users/steve/Development/git/cv-infer-py` and `/Users/steve/Development/git/cv-infer-rust` will be migrated, refactored, and unified under `cvaas-benchmarks`.
- **Dependencies**: 
  - Python: `opencv-python`, `tritonclient[all]`, `numpy`, `paho-mqtt`, `pyyaml`, `python-dotenv`.
  - Rust: `opencv`, `clap`, `tokio`, `triton-client`, `reqwest` / HTTP client, `paho-mqtt`, `serde`.
- **System Requirements**: OpenCV system libraries, Clang/LLVM for Rust bindings, local Docker daemon for Triton/MQTT/Redis.
