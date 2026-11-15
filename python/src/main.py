#!/usr/bin/env python3
"""
Python Computer Vision Traffic Inference Pipeline.
Supports video stream/file ingestion, Triton HTTP/gRPC inference,
detection visualization, MQTT telemetry, and headless benchmarking.
Features decoupled asynchronous display loop for smooth 30 FPS playback.
"""

import os
import sys
import time
import json
import argparse
import threading
import queue
from collections import deque
import numpy as np
import cv2 as cv

from utils import load_label_map, draw_detections
from infer import TritonInferenceClient
from mqtt_client import MqttPublisher
from tracker import BufferedVideoInterpolator


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
    parser.add_argument("--local", action="store_true", help="Force local real-time YOLO model instead of remote Triton")
    parser.add_argument("--model-size", choices=["nano", "small", "medium"], default="medium", help="Model size for local detection (default: medium)")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold for detections (default: 0.15)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size for local detector (default: 1280)")
    parser.add_argument("--no-enhance", action="store_true", help="Disable night-time CLAHE contrast enhancement")
    parser.add_argument("--all-classes", action="store_true", help="Detect all 80 COCO classes instead of filtering to traffic only")
    parser.add_argument("--sync", action="store_true", help="Run in blocking synchronous mode instead of smooth async display")
    parser.add_argument("--no-track", action="store_true", help="Disable buffered interpolation (display static keyframe detections)")
    return parser.parse_args()


def run_async_display(args, cap, triton, labels, mqtt, window_name):
    """
    Decoupled Asynchronous Display Loop with Buffered Keyframe Interpolation.
    Plays video smoothly at native frame rate (30 FPS) while running
    inference in a background worker thread.
    Uses buffered interpolation to ensure:
    - Zero trailing: Detections are anchored on future and past keyframes.
    - Zero wandering: No forward extrapolation; boxes are strictly bounded between real YOLO detections.
    - Butter-smooth 30 FPS playback with ~180ms buffer.
    """
    cap_fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / cap_fps

    interpolator = BufferedVideoInterpolator() if not args.no_track else None
    lock = threading.Lock()
    shared_state = {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int32),
        "scores": np.zeros((0,), dtype=np.float32),
        "num_dets": 0,
    }

    display_buffer = deque()
    infer_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def worker_loop():
        while not stop_event.is_set():
            try:
                sent_frame_id, frame_rgb = infer_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            boxes, classes, scores, num_dets, _ = triton.infer(frame_rgb, labels)
            with lock:
                if interpolator is not None:
                    interpolator.add_keyframe(sent_frame_id, boxes, classes, scores)
                else:
                    shared_state["boxes"] = boxes
                    shared_state["classes"] = classes
                    shared_state["scores"] = scores
                    shared_state["num_dets"] = num_dets
            infer_queue.task_done()

    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()

    # Pre-buffer initial frames (~6 frames / ~200ms)
    initial_buffer_size = 6
    read_frame_idx = 0
    for _ in range(initial_buffer_size):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        display_buffer.append((read_frame_idx, frame))
        if read_frame_idx == 0:
            rgb_init = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            infer_queue.put((0, rgb_init))
        read_frame_idx += 1

    # Wait for the initial keyframe
    if not infer_queue.empty():
        infer_queue.join()

    displayed_count = 0
    t_start = time.perf_counter()

    mode_label = "Buffered Keyframe Interpolation" if not args.no_track else "Static Keyframes"
    print(f"Smooth Async Display running at native ~{cap_fps:.1f} FPS with {mode_label} (press 'q' to quit)...")

    try:
        while True:
            if args.max_frames > 0 and displayed_count >= args.max_frames:
                break

            t_frame_start = time.perf_counter()

            # Maintain buffer depth of ~6-8 frames by reading ahead
            if len(display_buffer) < 10:
                ret, frame = cap.read()
                if ret and frame is not None:
                    display_buffer.append((read_frame_idx, frame))
                    if infer_queue.empty():
                        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                        try:
                            infer_queue.put_nowait((read_frame_idx, rgb_frame))
                        except queue.Full:
                            pass
                    read_frame_idx += 1
                elif len(display_buffer) == 0:
                    print("End of video stream reached.")
                    break

            if not display_buffer:
                break

            disp_idx, disp_frame = display_buffer.popleft()

            # Retrieve interpolated detections for this exact frame
            with lock:
                if interpolator is not None:
                    cur_boxes, cur_classes, cur_scores, cur_dets = interpolator.get_interpolated_detections(disp_idx)
                else:
                    cur_boxes = shared_state["boxes"]
                    cur_classes = shared_state["classes"]
                    cur_scores = shared_state["scores"]
                    cur_dets = shared_state["num_dets"]

            # Render overlay
            annotated = draw_detections(disp_frame, cur_boxes, cur_classes, cur_scores, labels, min_score_thresh=args.conf)

            # Display real-time video FPS counter on screen
            fps_text = f"Native Video: {cap_fps:.1f} FPS | Detections: {cur_dets}"
            cv.putText(annotated, fps_text, (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            cv.imshow(window_name, annotated)

            if args.mqtt_enabled and cur_dets > 0:
                mqtt.publish_detections(
                    stream_id=os.path.basename(args.source),
                    boxes=cur_boxes.tolist(),
                    classes=cur_classes.tolist(),
                    scores=cur_scores.tolist(),
                    timestamp=time.time(),
                )

            displayed_count += 1

            # Precise clock pacing for smooth 30 FPS
            elapsed = time.perf_counter() - t_frame_start
            sleep_ms = max(int((frame_interval - elapsed) * 1000), 1)
            key = cv.waitKey(sleep_ms) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        stop_event.set()
        total_time = time.perf_counter() - t_start
        print(f"\nPlayback ended. Displayed {displayed_count} frames in {total_time:.2f}s ({displayed_count/max(total_time,0.001):.1f} FPS).")


def run_sync_pipeline(args, cap, triton, labels, mqtt, window_name):
    """
    Synchronous Blocking Loop.
    Evaluates each frame sequentially (used for exact per-frame latency benchmarking).
    """
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
            rgb_frame = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
            t_prep = time.perf_counter()

            # 3. Inference
            boxes, classes, scores, num_dets, t_infer = triton.infer(rgb_frame, labels)
            t_inf_end = time.perf_counter()

            # 4. Postprocess / Visualization
            if not args.headless:
                annotated = draw_detections(frame, boxes, classes, scores, labels, min_score_thresh=args.conf)
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


def main():
    args = parse_args()

    if args.server is None:
        args.server = "localhost:8000" if args.protocol == "http" else "localhost:8001"

    print("=== CVaaS Python Inference Pipeline ===")
    print(f"Source:       {args.source}")
    print(f"Protocol:     {args.protocol.upper()} -> {args.server}")
    print(f"Model:        {args.model}")
    print(f"Model Size:   {args.model_size}")
    print(f"Headless:     {args.headless}")
    print(f"Mode:         {'Headless Benchmark' if args.headless else ('Sync Blocking' if args.sync else 'Smooth Async 30FPS')}")
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
        local_mode=args.local,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        enhance_night=(not args.no_enhance),
        traffic_only=(not args.all_classes),
        model_size=args.model_size,
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

    try:
        if not args.headless and not args.sync:
            # Smooth decoupled async display at native 30 FPS
            run_async_display(args, cap, triton, labels, mqtt, window_name)
        else:
            # Synchronous per-frame benchmark loop
            run_sync_pipeline(args, cap, triton, labels, mqtt, window_name)
    finally:
        cap.release()
        if not args.headless:
            cv.destroyAllWindows()
        mqtt.stop()


if __name__ == "__main__":
    main()
