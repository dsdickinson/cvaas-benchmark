# Rust Computer Vision Inference Engine

This directory contains the high-performance Rust implementation of the CVaaS traffic video inferencing pipeline. Built on the **Tokio** asynchronous runtime and native **OpenCV C++ bindings**, it delivers minimal latency and a lightweight memory footprint while supporting dual-protocol communication (**gRPC** and **HTTP**) to NVIDIA Triton Inference Server.

---

## Directory Layout

```
rust/
├── Cargo.toml               # Crate dependencies (opencv, tokio, reqwest, triton-client, clap)
├── Cargo.lock               # Pinned dependency lockfile
├── README.md                # This documentation file
└── src/
    ├── main.rs              # CLI argument parsing (clap), main event loop, and latency percentiles
    ├── triton.rs            # Dual-protocol Triton client (gRPC protobufs & HTTP v2 REST)
    ├── video.rs             # OpenCV frame preprocessing, color space conversion, and rendering
    └── mqtt.rs              # Telemetry message structures and MQTT integration
```

---

## Prerequisites & Compilation

Because this project interfaces directly with OpenCV C++ libraries via the `opencv` crate, system OpenCV libraries and Clang must be present on your host.

### 1. System Dependencies
- **macOS (via Homebrew)**:
  ```bash
  brew install opencv pkg-config llvm
  ```
- **Ubuntu / Debian**:
  ```bash
  sudo apt-get update && sudo apt-get install -y \
      build-essential pkg-config clang libclang-dev libopencv-dev
  ```

### 2. Building the Binary

#### Build Release Binary (Optimized):
```bash
# Set required library and LLVM search paths
export PKG_CONFIG_PATH="$(brew --prefix opencv)/lib/pkgconfig:$PKG_CONFIG_PATH"
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"

cd rust
cargo build --release
```

The compiled binary will be placed at:
```
rust/target/release/cv-infer-rust
```

---

## Architecture & How It Works

1. **Ingestion (`src/video.rs` & `src/main.rs`)**:
   - Opens local video files (e.g. `video/intersection-night.mp4`) or RTSP network streams using `opencv::videoio::VideoCapture`.
   - Reads uncompressed frames into native `opencv::core::Mat` buffers.
2. **Preprocessing (`src/video.rs`)**:
   - Resizes frames using bilinear interpolation (`imgproc::resize`) to model dimensions (default: `1024x600`).
   - Converts BGR to RGB color representation (`imgproc::cvt_color`).
   - Extracts continuous raw byte slices directly from `Mat` data without intermediate allocations.
3. **Dual-Protocol Triton Client (`src/triton.rs`)**:
   - **gRPC (`triton-client`)**: Connects to Triton's gRPC port (`8001`) via Tokio. Encodes raw byte buffers into `ModelInferRequest` protobufs and issues asynchronous remote calls.
   - **HTTP (`reqwest`)**: Formats binary tensor payloads to Triton's v2 REST endpoint `http://<host>:8000/v2/models/<model>/infer`.
   - **Mock Mode (`--mock`)**: Generates synthetic inference detections with simulated network delay for benchmarking without a live GPU server.
4. **Postprocessing & Output (`src/video.rs` & `src/main.rs`)**:
   - **Interactive Mode**: Computes normalized bounding box coordinates and draws green bounding boxes with text banners on the video display window via OpenCV `HighGUI`.
   - **Headless Mode (`--headless`)**: Bypasses GUI rendering to prevent display pipeline bottlenecks, logging exact nanosecond-resolution timing data for each pipeline phase.

---

## CLI Options & Usage

```bash
rust/target/release/cv-infer-rust [OPTIONS]
```

### Options:
- `--source`, `-s`: Path to video file or RTSP stream URL (default: `video/intersection-night.mp4`)
- `--protocol`, `-p`: Inference protocol `grpc` or `http` (default: `grpc`)
- `--server`: Triton server address (`host:port`). Defaults to `localhost:8001` for gRPC or `localhost:8000` for HTTP.
- `--model`, `-m`: Model name deployed on Triton (default: `object_detection`)
- `--width`: Input tensor width (default: `1024`)
- `--height`: Input tensor height (default: `600`)
- `--labels`: Path to label mapping file (default: `config/labels.txt`)
- `--headless`: Run without GUI window creation (essential for automated benchmarks)
- `--max-frames`: Limit number of frames processed (`0` for unlimited)
- `--metrics-out`: Path to export per-frame latency records and summary stats as JSON
- `--mock`: Run with synthetic inference (no live Triton server required)

---

## Execution Examples

### 1. Interactive Video Playback with Bounding Boxes
```bash
# Using gRPC against local Triton
rust/target/release/cv-infer-rust --source video/intersection-night.mp4 --protocol grpc

# Using HTTP against a remote Triton host
rust/target/release/cv-infer-rust --source video/intersection-night.mp4 --protocol http --server 192.168.1.80:8000
```

### 2. Live RTSP Camera Ingestion
```bash
rust/target/release/cv-infer-rust --source "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV001.stream" --protocol grpc
```

### 3. Headless Benchmark Run with Metrics Output
```bash
rust/target/release/cv-infer-rust \
  --source video/intersection-night.mp4 \
  --protocol grpc \
  --headless \
  --max-frames 200 \
  --metrics-out benchmarks/results/rust_grpc_sample.json
```

---

## Performance Characteristics

- **Memory Efficiency**: Operates with a typical resident memory footprint of **~260 MB RSS**, roughly 35% leaner than equivalent Python runtimes.
- **Timing Accuracy**: Uses `std::time::Instant` monotonic clocks to isolate frame acquisition (`capture_ms`), matrix transformations (`preprocess_ms`), network execution (`infer_ms`), and output parsing (`postprocess_ms`).
- **Throughput**: Capable of sustaining over **120+ FPS** throughput on multi-core systems when operating in headless mode.
