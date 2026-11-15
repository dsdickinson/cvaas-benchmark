#!/usr/bin/env python3
"""
Automated Benchmarking Harness for CVaaS (Python vs Rust).
Executes comparative benchmarks across engines, protocols, and workloads.
"""

import os
import sys
import time
import argparse
import subprocess
import json
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_BIN = os.path.join(ROOT_DIR, ".venv", "bin", "python")
RUST_RELEASE_BIN = os.path.join(ROOT_DIR, "rust", "target", "release", "cv-infer-rust")
RUST_DEBUG_BIN = os.path.join(ROOT_DIR, "rust", "target", "debug", "cv-infer-rust")


def parse_args():
    parser = argparse.ArgumentParser(description="CVaaS Automated Benchmark Runner")
    parser.add_argument("--source", "-s", default="video/intersection-night.mp4", help="Video source file or stream")
    parser.add_argument("--engine", choices=["python", "rust", "both"], default="both", help="Inference engine")
    parser.add_argument("--protocol", choices=["http", "grpc", "both"], default="both", help="Triton protocol")
    parser.add_argument("--max-frames", "-n", type=int, default=100, help="Frames to process per run")
    parser.add_argument("--output-dir", default="benchmarks/results", help="Directory to save benchmark metrics")
    parser.add_argument("--mock", action="store_true", help="Run with mock inference for standalone testing")
    parser.add_argument("--server", default=None, help="Triton server host:port")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Execution device (default: auto)")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU processing")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only execution")
    return parser.parse_args()


def get_rust_binary():
    if os.path.exists(RUST_RELEASE_BIN):
        return RUST_RELEASE_BIN
    if os.path.exists(RUST_DEBUG_BIN):
        return RUST_DEBUG_BIN
    return None


def run_single_benchmark(engine: str, protocol: str, args, out_json: str):
    print(f"\n>>> Running Benchmark: Engine={engine.upper()} | Protocol={protocol.upper()} | Frames={args.max_frames} <<<")

    source_path = os.path.join(ROOT_DIR, args.source) if not os.path.isabs(args.source) else args.source

    if engine == "python":
        py_exec = PYTHON_BIN if os.path.exists(PYTHON_BIN) else sys.executable
        cmd = [
            py_exec,
            os.path.join(ROOT_DIR, "python", "src", "main.py"),
            "--source", source_path,
            "--protocol", protocol,
            "--headless",
            "--max-frames", str(args.max_frames),
            "--metrics-out", out_json,
        ]
        if args.mock:
            cmd.append("--mock")
        if args.server:
            cmd.extend(["--server", args.server])
        if args.cpu:
            cmd.append("--cpu")
        elif args.gpu:
            cmd.append("--gpu")
        elif args.device != "auto":
            cmd.extend(["--device", args.device])

    elif engine == "rust":
        rust_bin = get_rust_binary()
        if not rust_bin:
            raise FileNotFoundError("Rust binary not found. Please build rust first (`cargo build --release`).")

        cmd = [
            rust_bin,
            "--source", source_path,
            "--protocol", protocol,
            "--headless",
            "--max-frames", str(args.max_frames),
            "--metrics-out", out_json,
        ]
        if args.mock:
            cmd.append("--mock")
        if args.server:
            cmd.extend(["--server", args.server])
        if args.cpu:
            cmd.append("--cpu")
        elif args.gpu:
            cmd.append("--gpu")
        elif args.device != "auto":
            cmd.extend(["--device", args.device])
    else:
        raise ValueError(f"Unknown engine: {engine}")

    env = os.environ.copy()
    # Add OpenCV library paths if on macOS
    try:
        brew_prefix = subprocess.check_output(["brew", "--prefix", "opencv"], text=True).strip()
        compat_dir = os.path.join(ROOT_DIR, "rust", "compat_lib")
        cv2_dylibs = os.path.join(ROOT_DIR, ".venv", "lib", "python3.15", "site-packages", "cv2", ".dylibs")
        existing_dyld = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{compat_dir}:{cv2_dylibs}:{brew_prefix}/lib:{existing_dyld}"
    except Exception:
        pass

    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    peak_rss_mb = 0.0
    # Monitor resource usage while running
    while proc.poll() is None:
        try:
            ps_out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(proc.pid)], text=True).strip()
            if ps_out:
                rss_kb = float(ps_out)
                peak_rss_mb = max(peak_rss_mb, rss_kb / 1024.0)
        except Exception:
            pass
        time.sleep(0.05)

    stdout, _ = proc.communicate()
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"Error running {engine} {protocol}:\n{stdout}")
        return False

    print(stdout)
    print(f"Completed in {elapsed:.2f}s | Peak RSS: {peak_rss_mb:.2f} MB")

    # Inject peak memory metric into JSON
    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                data = json.load(f)
            data["summary"]["peak_rss_mb"] = round(peak_rss_mb, 2)
            with open(out_json, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    return True


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    engines = ["python", "rust"] if args.engine == "both" else [args.engine]
    protocols = ["http", "grpc"] if args.protocol == "both" else [args.protocol]

    results = []
    for eng in engines:
        for proto in protocols:
            out_file = os.path.join(args.output_dir, f"{eng}_{proto}_results.json")
            success = run_single_benchmark(eng, proto, args, out_file)
            if success:
                results.append(out_file)

    # Generate comparative report
    report_script = os.path.join(ROOT_DIR, "benchmarks", "report.py")
    if os.path.exists(report_script) and results:
        print("\n>>> Generating Comparative Report <<<")
        subprocess.run([sys.executable, report_script, args.output_dir])


if __name__ == "__main__":
    main()
