## 1. Workspace Consolidation and Shared Configuration

- [x] 1.1 Copy shared configurations (`config.yaml`, `labels.txt`) from `cv-infer-py` into `config/` and create `.env.example`
- [x] 1.2 Copy MQTT & Redis Docker Compose services into `docker/docker-compose.infra.yml` and provide Triton launch configuration
- [x] 1.3 Verify shared test video (`video/intersection-night.mp4`) exists and is accessible

## 2. Python Inference Engine Implementation

- [x] 2.1 Consolidate and refactor Python inference pipeline into `python/` (`requirements.txt`, `src/main.py`, `src/infer.py`, `src/utils.py`)
- [x] 2.2 Implement dual-protocol support (`--protocol http` and `--protocol grpc`) using `tritonclient`
- [x] 2.3 Implement `--headless` mode, timing breakdown instrumentation (capture, preprocess, infer, postprocess), and structured metric logging
- [x] 2.4 Verify Python inference execution against video file and validate outputs

## 3. Rust Inference Engine Implementation

- [x] 3.1 Migrate `cv-infer-rust` into `rust/` and update `Cargo.toml` dependencies (Triton gRPC client, reqwest HTTP client, OpenCV, Clap, Tokio)
- [x] 3.2 Implement frame preprocessing and tensor conversion in `rust/src/`
- [x] 3.3 Implement dual-protocol Triton inference in `rust/src/` supporting both gRPC and HTTP v2 endpoints
- [x] 3.4 Implement `--headless` mode, HighGUI conditional rendering, and high-resolution timing metrics
- [x] 3.5 Build and verify Rust engine execution against video file and validate outputs

## 4. Benchmarking Suite Implementation

- [x] 4.1 Create `benchmarks/run_benchmarks.py` orchestrator supporting `--engine [python|rust|both]`, `--protocol [http|grpc]`, and `--iterations`
- [x] 4.2 Implement resource profiling (CPU, RAM) and latency metrics aggregation (FPS, p50, p90, p95, p99)
- [x] 4.3 Create `benchmarks/report.py` to generate markdown summary comparison tables and CSV exports

## 5. End-to-End Verification and Validation

- [x] 5.1 Run test benchmarks on both Python and Rust engines under both HTTP and gRPC protocols
- [x] 5.2 Generate and document initial benchmark comparison results in `benchmarks/README.md`
