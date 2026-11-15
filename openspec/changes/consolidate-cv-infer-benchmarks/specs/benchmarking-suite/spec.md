## ADDED Requirements

### Requirement: Benchmark Orchestration
The benchmarking suite SHALL provide an automated harness to run both Python and Rust inference engines under identical configurations and workloads.

#### Scenario: Running dual-engine comparative benchmark
- **WHEN** user executes `benchmarks/run_benchmarks.py --engine both --protocol grpc --iterations 500`
- **THEN** harness executes Python engine then Rust engine sequentially under identical inputs and collects performance data

#### Scenario: Protocol comparison benchmark
- **WHEN** user executes benchmark comparing `--protocol http` and `--protocol grpc` on an engine
- **THEN** harness collects and compares metrics across both protocols for that engine

### Requirement: Granular Timing and Metric Collection
The benchmarking suite SHALL collect timing breakdown for each frame, including frame acquisition, pre-processing, network/inference round-trip, and post-processing.

#### Scenario: Per-frame latency logging
- **WHEN** an inference engine executes in benchmark mode
- **THEN** system logs timestamped latency records including ingestion, preprocessing, inference, and postprocessing durations

#### Scenario: Metric aggregation
- **WHEN** benchmark execution completes
- **THEN** system computes overall FPS, mean latency, p50, p90, p95, p99 percentiles, and dropped frame count

### Requirement: Resource Utilization Profiling
The benchmarking suite SHALL monitor and report host/accelerator CPU and system memory consumption during benchmark execution.

#### Scenario: Resource sampling during benchmark
- **WHEN** benchmark is running
- **THEN** harness periodically samples CPU usage and RSS memory usage of the target engine process and saves stats to output reports

### Requirement: Comparative Reporting and Visualization
The benchmarking suite SHALL generate human-readable summary tables and visual comparison charts from collected benchmark runs.

#### Scenario: Generating summary report
- **WHEN** benchmark finishes
- **THEN** system outputs a markdown summary table comparing Python vs Rust across throughput, latency percentiles, and resource usage
