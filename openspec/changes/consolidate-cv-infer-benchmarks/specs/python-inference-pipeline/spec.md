## ADDED Requirements

### Requirement: Python Video Ingestion
The Python inference engine SHALL accept input from local video files (e.g., MP4) or RTSP streams and decode sequential frames using OpenCV.

#### Scenario: Ingest from local video file
- **WHEN** user executes python engine with `--source ./video/intersection-night.mp4`
- **THEN** system decodes frames sequentially until EOF or specified frame limit

#### Scenario: Ingest from RTSP stream
- **WHEN** user executes python engine with `--source rtsp://example.com/live`
- **THEN** system opens network stream and decodes continuous live frames

### Requirement: Python Dual-Protocol Triton Inference
The Python inference engine SHALL support both HTTP and gRPC protocols to execute inference requests against a Triton Inference Server.

#### Scenario: HTTP inference execution
- **WHEN** user specifies `--protocol http --server localhost:8000 --model object_detection`
- **THEN** system submits pre-processed frame tensors to Triton via `tritonclient.http` and receives prediction outputs

#### Scenario: gRPC inference execution
- **WHEN** user specifies `--protocol grpc --server localhost:8001 --model object_detection`
- **THEN** system submits pre-processed frame tensors to Triton via `tritonclient.grpc` and receives prediction outputs

### Requirement: Python Execution Modes
The Python inference engine SHALL support interactive GUI rendering with bounding box overlays and headless execution for benchmarking.

#### Scenario: Interactive display mode
- **WHEN** engine is run without `--headless`
- **THEN** system renders frames with detected bounding boxes and class labels in an OpenCV display window

#### Scenario: Headless benchmark mode
- **WHEN** engine is run with `--headless`
- **THEN** system suppresses all GUI window initialization and outputs structured per-frame latency metrics to stdout or file

### Requirement: Python Telemetry and MQTT Publishing
The Python inference engine SHALL support optional publishing of telemetry, detection bounding boxes, and base64-encoded frames to an MQTT broker.

#### Scenario: MQTT publishing enabled
- **WHEN** engine is executed with `--mqtt-enabled true`
- **THEN** system connects to configured MQTT broker and publishes detection coordinates to designated topics
