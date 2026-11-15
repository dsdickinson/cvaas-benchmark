#!/usr/bin/env python3
"""
Benchmark Report Generator.
Aggregates JSON metrics reports from Python and Rust benchmark runs
and produces Markdown summary tables and CSV exports.
"""

import os
import sys
import json
import glob
import csv
from typing import List, Dict, Any


def load_result_file(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data.get("summary", {})
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def format_markdown_table(summaries: List[Dict[str, Any]]) -> str:
    headers = [
        "Engine",
        "Protocol",
        "Frames",
        "Time (s)",
        "FPS",
        "Total Mean (ms)",
        "Total p50 (ms)",
        "Total p95 (ms)",
        "Total p99 (ms)",
        "Infer Mean (ms)",
        "Infer p50 (ms)",
    ]

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    rows = []
    for s in summaries:
        if not s:
            continue
        tot = s.get("latency_total_ms", {})
        inf = s.get("latency_infer_ms", {})
        row = [
            s.get("engine", "unknown").capitalize(),
            s.get("protocol", "unknown").upper(),
            str(s.get("frames_processed", 0)),
            f"{s.get('total_elapsed_sec', 0.0):.2f}",
            f"{s.get('fps', 0.0):.2f}",
            f"{tot.get('mean', 0.0):.2f}",
            f"{tot.get('p50', 0.0):.2f}",
            f"{tot.get('p95', 0.0):.2f}",
            f"{tot.get('p99', 0.0):.2f}",
            f"{inf.get('mean', 0.0):.2f}",
            f"{inf.get('p50', 0.0):.2f}",
        ]
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join([header_row, separator_row] + rows)


def export_csv(summaries: List[Dict[str, Any]], csv_path: str):
    fieldnames = [
        "engine",
        "protocol",
        "frames_processed",
        "total_elapsed_sec",
        "fps",
        "total_mean_ms",
        "total_p50_ms",
        "total_p95_ms",
        "total_p99_ms",
        "infer_mean_ms",
        "infer_p50_ms",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            tot = s.get("latency_total_ms", {})
            inf = s.get("latency_infer_ms", {})
            writer.writerow({
                "engine": s.get("engine"),
                "protocol": s.get("protocol"),
                "frames_processed": s.get("frames_processed"),
                "total_elapsed_sec": s.get("total_elapsed_sec"),
                "fps": s.get("fps"),
                "total_mean_ms": tot.get("mean"),
                "total_p50_ms": tot.get("p50"),
                "total_p95_ms": tot.get("p95"),
                "total_p99_ms": tot.get("p99"),
                "infer_mean_ms": inf.get("mean"),
                "infer_p50_ms": inf.get("p50"),
            })
    print(f"Exported CSV to {csv_path}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results"
    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    if not json_files:
        print(f"No benchmark JSON files found in {results_dir}")
        return

    summaries = []
    for f in sorted(json_files):
        s = load_result_file(f)
        if s:
            summaries.append(s)

    md_table = format_markdown_table(summaries)
    print("\n=== Comparative Benchmark Results ===\n")
    print(md_table)
    print("\n=====================================\n")

    summary_md_path = os.path.join(results_dir, "benchmark_summary.md")
    with open(summary_md_path, "w") as f:
        f.write("# CVaaS Benchmark Results\n\n" + md_table + "\n")
    print(f"Saved markdown summary to {summary_md_path}")

    csv_path = os.path.join(results_dir, "benchmark_summary.csv")
    export_csv(summaries, csv_path)


if __name__ == "__main__":
    main()
