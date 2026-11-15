mod mqtt;
mod triton;
mod video;

use clap::Parser;
use opencv::{highgui, prelude::*, videoio, Result};
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "cv-infer-rust", about = "CVaaS Rust Traffic Inference Pipeline")]
struct Args {
    /// Video file or RTSP stream URL
    #[arg(short, long, default_value = "video/intersection-night.mp4")]
    source: String,

    /// Triton protocol: http or grpc
    #[arg(short, long, default_value = "grpc")]
    protocol: String,

    /// Triton server address (host:port)
    #[arg(long)]
    server: Option<String>,

    /// Model name on Triton
    #[arg(short, long, default_value = "object_detection")]
    model: String,

    /// Input tensor width
    #[arg(long, default_value_t = 1024)]
    width: i32,

    /// Input tensor height
    #[arg(long, default_value_t = 600)]
    height: i32,

    /// Path to label map
    #[arg(long, default_value = "config/labels.txt")]
    labels: String,

    /// Headless mode (no GUI window)
    #[arg(long, default_value_t = false)]
    headless: bool,

    /// Maximum frames to process (0 = all)
    #[arg(long, default_value_t = 0)]
    max_frames: usize,

    /// Enable MQTT publishing
    #[arg(long, default_value_t = false)]
    mqtt_enabled: bool,

    /// MQTT broker address
    #[arg(long, default_value = "localhost:1883")]
    mqtt_broker: String,

    /// Path to dump JSON benchmark metrics
    #[arg(long)]
    metrics_out: Option<String>,

    /// Force mock inference mode
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// Optional execution device: auto, cpu, or cuda
    #[arg(long, default_value = "auto")]
    device: String,

    /// Enable GPU processing
    #[arg(long, default_value_t = false)]
    gpu: bool,

    /// Force CPU processing
    #[arg(long, default_value_t = false)]
    cpu: bool,
}

#[derive(Serialize)]
struct FrameMetric {
    frame: usize,
    capture_ms: f64,
    preprocess_ms: f64,
    infer_ms: f64,
    postprocess_ms: f64,
    total_ms: f64,
    num_detections: usize,
}

#[derive(Serialize)]
struct LatencyPercentiles {
    mean: f64,
    p50: f64,
    p90: f64,
    p95: f64,
    p99: f64,
}

#[derive(Serialize)]
struct BenchmarkSummary {
    engine: String,
    protocol: String,
    frames_processed: usize,
    total_elapsed_sec: f64,
    fps: f64,
    latency_total_ms: LatencyPercentiles,
    latency_infer_ms: LatencyPercentiles,
}

#[derive(Serialize)]
struct FullReport {
    summary: BenchmarkSummary,
    per_frame: Vec<FrameMetric>,
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn calc_percentiles(values: &[f64]) -> LatencyPercentiles {
    if values.is_empty() {
        return LatencyPercentiles { mean: 0.0, p50: 0.0, p90: 0.0, p95: 0.0, p99: 0.0 };
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    LatencyPercentiles {
        mean: (mean * 100.0).round() / 100.0,
        p50: (percentile(&sorted, 50.0) * 100.0).round() / 100.0,
        p90: (percentile(&sorted, 90.0) * 100.0).round() / 100.0,
        p95: (percentile(&sorted, 95.0) * 100.0).round() / 100.0,
        p99: (percentile(&sorted, 99.0) * 100.0).round() / 100.0,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let protocol = args.protocol.to_lowercase();
    let server_url = args.server.unwrap_or_else(|| {
        if protocol == "http" {
            "localhost:8000".to_string()
        } else {
            "localhost:8001".to_string()
        }
    });

    println!("=== CVaaS Rust Inference Pipeline ===");
    println!("Source:       {}", args.source);
    println!("Protocol:     {} -> {}", protocol.to_uppercase(), server_url);
    println!("Model:        {}", args.model);
    println!("Device:       {}", if args.cpu { "CPU" } else if args.gpu { "GPU (CUDA)" } else { "AUTO" });
    println!("Resolution:   {}x{}", args.width, args.height);
    println!("Headless:     {}", args.headless);
    println!(
        "Max Frames:   {}",
        if args.max_frames > 0 {
            args.max_frames.to_string()
        } else {
            "Unlimited".to_string()
        }
    );
    println!("======================================");

    let labels = video::load_labels(&args.labels);
    let triton = triton::TritonClient::new(
        server_url.clone(),
        protocol.clone(),
        args.model.clone(),
        args.mock,
    );
    let _mqtt = mqtt::MqttPublisher::new(
        args.mqtt_broker.clone(),
        "gg/video".to_string(),
        args.mqtt_enabled,
    );

    let mut cap = videoio::VideoCapture::from_file(&args.source, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cap)? {
        eprintln!("Error: Unable to open video source: {}", args.source);
        std::process::exit(1);
    }

    let window_name = format!("CVaaS Rust: {}", args.source);
    if !args.headless {
        highgui::named_window(&window_name, highgui::WINDOW_NORMAL)?;
        highgui::resize_window(&window_name, 960, 540)?;
    }

    let mut frame_count = 0usize;
    let mut metrics_records = Vec::new();
    let overall_start = Instant::now();

    loop {
        if args.max_frames > 0 && frame_count >= args.max_frames {
            break;
        }

        let t0 = Instant::now();

        // 1. Capture
        let mut frame = Mat::default();
        if !cap.read(&mut frame)? || frame.empty() {
            println!("End of video stream reached.");
            break;
        }
        let t_cap = Instant::now();

        // 2. Preprocess
        let rgb_mat = video::preprocess_frame(&frame, args.width, args.height)?;
        let t_prep = Instant::now();

        // 3. Infer
        let (dets, latency_sec) = match triton.infer(&rgb_mat).await {
            Ok(res) => res,
            Err(e) => {
                eprintln!("Inference error: {}", e);
                break;
            }
        };
        let t_inf_end = Instant::now();

        // 4. Postprocess & Display
        if !args.headless {
            video::draw_detections(&mut frame, &dets.boxes, &dets.classes, &dets.scores, &labels, 0.25)?;
            highgui::imshow(&window_name, &frame)?;
            let key = highgui::wait_key(1)?;
            if key == 113 || key == 27 {
                // 'q' or ESC
                break;
            }
        }
        let t_post = Instant::now();
        frame_count += 1;

        let dur_cap = (t_cap - t0).as_secs_f64() * 1000.0;
        let dur_prep = (t_prep - t_cap).as_secs_f64() * 1000.0;
        let dur_infer = latency_sec * 1000.0;
        let dur_post = (t_post - t_inf_end).as_secs_f64() * 1000.0;
        let dur_total = (t_post - t0).as_secs_f64() * 1000.0;

        metrics_records.push(FrameMetric {
            frame: frame_count,
            capture_ms: dur_cap,
            preprocess_ms: dur_prep,
            infer_ms: dur_infer,
            postprocess_ms: dur_post,
            total_ms: dur_total,
            num_detections: dets.num_detections,
        });

        if frame_count % 30 == 0 {
            println!(
                "Frame {:4} | Total: {:6.2}ms | Infer: {:6.2}ms | Dets: {}",
                frame_count, dur_total, dur_infer, dets.num_detections
            );
        }
    }

    cap.release()?;
    if !args.headless {
        highgui::destroy_all_windows()?;
    }

    let total_elapsed = overall_start.elapsed().as_secs_f64();

    if frame_count > 0 {
        let totals: Vec<f64> = metrics_records.iter().map(|m| m.total_ms).collect();
        let infers: Vec<f64> = metrics_records.iter().map(|m| m.infer_ms).collect();
        let fps = frame_count as f64 / total_elapsed;

        let summary = BenchmarkSummary {
            engine: "rust".to_string(),
            protocol,
            frames_processed: frame_count,
            total_elapsed_sec: (total_elapsed * 10000.0).round() / 10000.0,
            fps: (fps * 100.0).round() / 100.0,
            latency_total_ms: calc_percentiles(&totals),
            latency_infer_ms: calc_percentiles(&infers),
        };

        println!("\n=== Benchmark Summary ===");
        println!("Frames:          {}", summary.frames_processed);
        println!("Elapsed Time:    {:.4} s", summary.total_elapsed_sec);
        println!("Average FPS:     {:.2}", summary.fps);
        println!(
            "Total Latency:   Mean={}ms | p50={}ms | p95={}ms | p99={}ms",
            summary.latency_total_ms.mean,
            summary.latency_total_ms.p50,
            summary.latency_total_ms.p95,
            summary.latency_total_ms.p99
        );
        println!(
            "Inference Time:  Mean={}ms | p50={}ms",
            summary.latency_infer_ms.mean, summary.latency_infer_ms.p50
        );
        println!("=========================");

        if let Some(out_path) = args.metrics_out {
            if let Some(parent) = Path::new(&out_path).parent() {
                let _ = fs::create_dir_all(parent);
            }
            let full_report = FullReport {
                summary,
                per_frame: metrics_records,
            };
            if let Ok(json_str) = serde_json::to_string_pretty(&full_report) {
                let _ = fs::write(&out_path, json_str);
                println!("Metrics saved to {}", out_path);
            }
        }
    }

    Ok(())
}
