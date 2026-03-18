"""
Day 4 — Task 4
Performance Benchmark: runs N inferences and reports latency stats.
Target: mean < 300ms, P95 < 500ms on entry-level hardware.

Usage:
    python day4/benchmark.py --model neoscreen_v1.tflite --runs 50
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.sclera_detection import preprocess_for_inference


def make_dummy_sclera():
    img = np.full((224, 224, 3), (180, 210, 235), dtype=np.uint8)
    img += np.random.randint(-15, 15, img.shape, dtype=np.int16).clip(-128, 127).astype(np.uint8)
    return img


def run(model_path, n_runs):
    print("=" * 55)
    print("NeoScreen — Day 4: Performance Benchmark")
    print("=" * 55)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    model_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nModel: {model_path}  ({model_mb:.2f} MB)")
    print(f"Runs:  {n_runs}")

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    # Warmup
    print("\nWarming up (5 runs)...")
    dummy = preprocess_for_inference(make_dummy_sclera())
    for _ in range(5):
        interp.set_tensor(inp[0]["index"], dummy)
        interp.invoke()

    # Benchmark
    print(f"Benchmarking ({n_runs} runs)...")
    latencies = []
    for i in range(n_runs):
        sclera = make_dummy_sclera()
        tensor = preprocess_for_inference(sclera)

        t0 = time.perf_counter()
        interp.set_tensor(inp[0]["index"], tensor)
        interp.invoke()
        _ = interp.get_tensor(out[0]["index"])[0]
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    mean_ms  = sum(latencies) / len(latencies)
    min_ms   = latencies[0]
    max_ms   = latencies[-1]
    p50_ms   = latencies[int(n_runs * 0.50)]
    p95_ms   = latencies[int(n_runs * 0.95)]
    p99_ms   = latencies[int(n_runs * 0.99)]

    print(f"\n── Latency Results ({n_runs} runs) ──────────────────")
    print(f"  Min:    {min_ms:>7.1f} ms")
    print(f"  Mean:   {mean_ms:>7.1f} ms  {'✓' if mean_ms < 300 else '⚠'} (target < 300ms)")
    print(f"  P50:    {p50_ms:>7.1f} ms")
    print(f"  P95:    {p95_ms:>7.1f} ms  {'✓' if p95_ms < 500 else '⚠'} (target < 500ms)")
    print(f"  P99:    {p99_ms:>7.1f} ms")
    print(f"  Max:    {max_ms:>7.1f} ms")
    print(f"  Model:  {model_mb:>7.2f} MB  {'✓' if model_mb < 5 else '⚠'} (target < 5MB)")

    all_pass = mean_ms < 300 and p95_ms < 500 and model_mb < 5
    print(f"\n{'✓ All targets met — ready for Moto E13 deployment.' if all_pass else '⚠ Some targets missed — check above.'}")

    print("\nNote: These benchmarks run on your dev machine.")
    print("On Moto E13 (ARM Cortex-A53), expect 1.5-2× slower.")
    print("Real device test in Day 5.")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="neoscreen_v1.tflite")
    parser.add_argument("--runs",  type=int, default=50)
    args = parser.parse_args()
    run(args.model, args.runs)
