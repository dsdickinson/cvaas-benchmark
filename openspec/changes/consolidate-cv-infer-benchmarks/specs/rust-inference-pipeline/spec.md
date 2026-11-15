## ADDED Requirements

### Requirement: Rust Video Ingestion
The Rust inference engine SHALL accept input from local video files (e.g., MP4) or RTSP streams and decode sequential frames using the `opencv` videoio module.

#### Scenario: Ingest from local video file in Rust
- **WHEN** user executes rust binary with `--source ./video/intersection-night.mp4`
- **THEN** system decodes video frames sequentially until end of file or termination signal

#### Scenario: Ingest from RTSP stream in Rust
- **WHEN** user executes rust binary with `--source rtsp://example.com/live`
- **THEN** system connects to the RTSP URL and decodes incoming video stream frames

### Requirement: Rust Dual-Protocol Triton Inference
The Rust inference engine SHALL support both gRPC and HTTP protocols to communicate with Triton Inference Server.

#### Scenario: Rust gRPC inference execution
- **WHEN** user specifies `--protocol grpc --server localhost:8001 --model object_detection`
- **THEN** system serializes frame tensor data into Triton gRPC protobuf requests and parses returned detection outputs

#### Scenario: Rust HTTP inference execution
- **WHEN** user specifies `--protocol http --server localhost:8000 --model object_detection`
- **THEN** system submits v2 REST inference payloads to Triton and parses JSON/binary tensor responses

### Requirement: Rust Execution Modes
The Rust inference engine SHALL provide both interactive GUI display with bounding boxes and headless execution for automated benchmarking.

#### Scenario: Rust interactive display mode
- **WHEN** binary runs without `--headless`
- **THEN** system opens HighGUI window and renders annotated video frames with bounding boxes

#### Scenario: Rust headless benchmark mode
- **WHEN** binary runs with `--headless`
- **THEN** system skips HighGUI window creation and emits per-frame timing and metrics in JSON/CSV format

### Requirement: Rust Telemetry and MQTT Publishing
The Rust inference engine SHALL support optional asynchronous publishing of detection results and encoded frames to an MQTT broker.

#### Scenario: Rust MQTT publishing
- **WHEN** engine is run with `--mqtt-enabled true`
- **THEN** system connects to MQTT broker using `paho-mqtt` and publishes detection telemetry
