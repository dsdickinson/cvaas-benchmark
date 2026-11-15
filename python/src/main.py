#!/usr/bin/env python3
"""
Python Computer Vision Traffic Inference Pipeline.
Supports video stream/file ingestion, Triton HTTP/gRPC inference,
detection visualization, MQTT telemetry, and headless benchmarking.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2 as cv

from utils import load_label_map, draw_detections
from infer import TritonInferenceClient
from mqtt_client import MqttPublisher


def parse_args():
    parser = argparse.ArgumentParser(description="CVaaS Python Traffic Inference Pipeline")
    parser.add_argument("--source", "-s", default="video/intersection-night.mp4", help="Video file or RTSP stream URL")
    parser.add_argument("--protocol", "-p", choices=["http", "grpc"], default="http", help="Triton inference protocol")
    parser.add_argument("--server", default=None, help="Triton server address (host:port)")
    parser.add_argument("--model", "-m", default="object_detection", help="Model name on Triton")
    parser.add_argument("--width", type=int, default=1024, help="Inference input tensor width")
    parser.add_argument("--height", type=int, default=600, help="Inference input tensor height")
    parser.add_argument("--labels", default="config/labels.txt", help="Path to labels map file")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display for benchmarking")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process (0 = all)")
    parser.add_argument("--mqtt-enabled", action="store_true", help="Enable publishing detections to MQTT")
    parser.add_argument("--mqtt-broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--metrics-out", default=None, help="Path to save benchmark metrics JSON")
    parser.add_argument("--mock", action="store_true", help="Force mock inference without remote Triton server")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine default server port if not specified
    if args.server is None:
        args.server = "localhost:8000" if args.protocol == "http" else "localhost:8001"

    print("=== CVaaS Python Inference Pipeline ===")
    print(f"Source:       {args.source}")
    print(f"Protocol:     {args.protocol.upper()} -> {args.server}")
    print(f"Model:        {args.model}")
    print(f"Resolution:   {args.width}x{args.height}")
    print(f"Headless:     {args.headless}")
    print(f"Max Frames:   {args.max_frames if args.max_frames > 0 else 'Unlimited'}")
    print("========================================")

    # Load labels
    labels = load_label_map(args.labels)

    # Initialize Triton client
    triton = TritonInferenceClient(
        server_url=args.server,
        protocol=args.protocol,
        model_name=args.model,
        mock_mode=args.mock,
    )

    # Initialize MQTT if requested
    mqtt = MqttPublisher(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        enabled=args.mqtt_enabled,
    )

    # Open video capture
    cap = cv.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {args.source}")
        sys.exit(1)

    window_name = f"CVaaS Python: {args.source}"
    if not args.headless:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 960, 540)

    frame_count = 0
    metrics_records = []
    overall_start_time = time.perf_counter()

    try:
        while True:
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

            t0 = time.perf_counter()

            # 1. Capture Frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print("End of video stream reached.")
                break
            t_cap = time.perf_counter()

            # 2. Preprocess Frame
            resized = cv.resize(frame, (args.width, args.height))
            # Convert BGR to RGB for model tensor
            rgb_frame = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
            t_prep = time.perf_counter()

            # 3. Inference
            boxes, classes, scores, num_dets, t_infer = triton.infer(rgb_frame)
            t_inf_end = time.perf_counter()

            # 4. Postprocess / Visualization
            if not args.headless:
                annotated = draw_detections(frame, boxes, classes, scores, labels)
                cv.imshow(window_name, annotated)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.mqtt_enabled:
                mqtt.publish_detections(
                    stream_id=os.path.basename(args.source),
                    boxes=boxes.tolist(),
                    classes=classes.tolist(),
                    scores=scores.tolist(),
                    timestamp=time.time(),
                )

            t_post = time.perf_counter()
            frame_count += 1

            dur_cap = (t_cap - t0) * 1000.0
            dur_prep = (t_prep - t_cap) * 1000.0
            dur_infer = t_infer * 1000.0
            dur_post = (t_post - t_inf_end) * 1000.0
            dur_total = (t_post - t0) * 1000.0

            metrics_records.append({
                "frame": frame_count,
                "capture_ms": dur_cap,
                "preprocess_ms": dur_prep,
                "infer_ms": dur_infer,
                "postprocess_ms": dur_post,
                "total_ms": dur_total,
                "num_detections": num_dets,
            })

            if frame_count % 30 == 0:
                print(f"Frame {frame_count:4d} | Total: {dur_total:6.2f}ms | Infer: {dur_infer:6.2f}ms | Dets: {num_dets}")

    finally:
        total_elapsed = time.perf_counter() - overall_start_time
        cap.release()
        if not args.headless:
            cv.destroyAllWindows()
        mqtt.stop()

    if frame_count > 0:
        totals = [m["total_ms"] for m in metrics_records]
        infers = [m["infer_ms"] for m in metrics_records]
        avg_fps = frame_count / total_elapsed

        summary = {
            "engine": "python",
            "protocol": args.protocol,
            "frames_processed": frame_count,
            "total_elapsed_sec": round(total_elapsed, 4),
            "fps": round(avg_fps, 2),
            "latency_total_ms": {
                "mean": round(float(np.mean(totals)), 2),
                "p50": round(float(np.percentile(totals, 50)), 2),
                "p90": round(float(np.percentile(totals, 90)), 2),
                "p95": round(float(np.percentile(totals, 95)), 2),
                "p99": round(float(np.percentile(totals, 99)), 2),
            },
            "latency_infer_ms": {
                "mean": round(float(np.mean(infers)), 2),
                "p50": round(float(np.percentile(infers, 50)), 2),
                "p95": round(float(np.percentile(infers, 95)), 2),
                "p99": round(float(np.percentile(infers, 99)), 2),
            },
        }

        print("\n=== Benchmark Summary ===")
        print(f"Frames:          {summary['frames_processed']}")
        print(f"Elapsed Time:    {summary['total_elapsed_sec']} s")
        print(f"Average FPS:     {summary['fps']}")
        print(f"Total Latency:   Mean={summary['latency_total_ms']['mean']}ms | p50={summary['latency_total_ms']['p50']}ms | p95={summary['latency_total_ms']['p95']}ms | p99={summary['latency_total_ms']['p99']}ms")
        print(f"Inference Time:  Mean={summary['latency_infer_ms']['mean']}ms | p50={summary['latency_infer_ms']['p50']}ms")
        print("=========================")

        if args.metrics_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
            with open(args.metrics_out, "w") as f:
                json.dump({"summary": summary, "per_frame": metrics_records}, f, indent=2)
            print(f"Metrics saved to {args.metrics_out}")


if __name__ == "__main__":
    main()
