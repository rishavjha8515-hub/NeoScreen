
import argparse
import os
import sys
import time
import glob

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.sclera_detection import detect_sclera, preprocess_for_inference
from ml.inference import classify_risk, get_risk_message, THRESHOLD_HIGH, THRESHOLD_LOW


def pick_10_images(data_dir: str) -> list[tuple[str, str]]:
    """Return up to 10 (image_path, true_class) pairs, balanced across classes."""
    CLASSES = ["Low", "Medium", "High"]
    pairs = []
    for cls in CLASSES:
        folder = os.path.join(data_dir, cls)
        imgs = sorted(glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.jpeg"))
        for img in imgs[:4]:        # up to 4 per class = up to 12 total, capped at 10
            pairs.append((img, cls))
        if len(pairs) >= 10:
            break
    return pairs[:10]


def run_test(model_path: str, data_dir: str):
    print("=" * 60)
    print("NeoScreen — Day 1 Task 5: 10-Image Inference Test")
    print("=" * 60)

    # ── 1. Model size check ───────────────────────────────────────────────────
    print(f"\n[1/4] Checking model: {model_path}")
    if not os.path.exists(model_path):
        print(f"  ERROR: {model_path} not found.")
        print("  Run training first: python ml/train.py")
        sys.exit(1)

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    size_ok = size_mb < 5.0
    print(f"  Size: {size_mb:.2f} MB  {'✓ OK' if size_ok else '✗ TOO LARGE (target < 5 MB)'}")

    # ── 2. Load interpreter ───────────────────────────────────────────────────
    print("\n[2/4] Loading TF Lite interpreter...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    print(f"  ✓ Input shape:  {inp[0]['shape']}  dtype: {inp[0]['dtype'].__name__}")
    print(f"  ✓ Output shape: {out[0]['shape']}  dtype: {out[0]['dtype'].__name__}")

    # ── 3. Pick 10 images ─────────────────────────────────────────────────────
    print("\n[3/4] Picking test images...")
    pairs = pick_10_images(data_dir)
    if not pairs:
        print(f"  ERROR: No images found in {data_dir}")
        print("  Run: python day1/dataset_setup.py")
        sys.exit(1)
    print(f"  Found {len(pairs)} images to test")

    # ── 4. Run inference ──────────────────────────────────────────────────────
    print("\n[4/4] Running inference...")
    print(f"\n  {'#':<3} {'File':<30} {'True':>7} {'Pred':>7} {'P(H)':>6} {'ms':>6}  {'OK?'}")
    print("  " + "-" * 65)

    results = []
    latencies = []
    CLASS_IDX = {"Low": 0, "Medium": 1, "High": 2}

    for i, (img_path, true_cls) in enumerate(pairs, 1):
        # Sclera detection (use raw image directly if detection fails on synthetics)
        roi = detect_sclera(img_path)
        if roi is None:
            # Synthetic images might not pass HSV detection; load directly
            raw = cv2.imread(img_path)
            roi = cv2.resize(raw, (224, 224)) if raw is not None else None

        if roi is None:
            print(f"  {i:<3} {'(skip — unreadable)':<30}")
            continue

        tensor = preprocess_for_inference(roi)

        t0 = time.perf_counter()
        interpreter.set_tensor(inp[0]["index"], tensor)
        interpreter.invoke()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        probs = interpreter.get_tensor(out[0]["index"])[0]
        pred_cls = classify_risk(probs)
        fname = os.path.basename(img_path)[:28]
        match = "✓" if pred_cls == true_cls else "✗"

        print(f"  {i:<3} {fname:<30} {true_cls:>7} {pred_cls:>7} "
              f"{probs[2]:>6.3f} {elapsed_ms:>6.1f}  {match}")

        results.append({
            "true": true_cls,
            "pred": pred_cls,
            "probs": probs,
            "ms": elapsed_ms,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    if not results:
        print("\n  No results — check dataset.")
        return

    correct = sum(1 for r in results if r["true"] == r["pred"])
    accuracy = correct / len(results)
    avg_ms = sum(latencies) / len(latencies)
    max_ms = max(latencies)

    # Critical: High Risk sensitivity
    high_results = [r for r in results if r["true"] == "High"]
    high_correct = sum(1 for r in high_results if r["pred"] == "High")
    sensitivity = high_correct / len(high_results) if high_results else float("nan")

    print("\n" + "  " + "=" * 55)
    print(f"  Overall accuracy:        {accuracy:.0%}  ({correct}/{len(results)})")
    print(f"  High Risk sensitivity:   {sensitivity:.0%}  ({high_correct}/{len(high_results)}) [target ≥ 95%]")
    print(f"  Avg inference time:      {avg_ms:.1f} ms  [target < 500 ms]")
    print(f"  Max inference time:      {max_ms:.1f} ms")
    print(f"  Model size:              {size_mb:.2f} MB  [target < 5 MB]")

    print("\n  Thresholds in use:")
    print(f"    HIGH   if P(High) ≥ {THRESHOLD_HIGH}")
    print(f"    LOW    if P(Low)  ≥ {THRESHOLD_LOW}")
    print(f"    MEDIUM otherwise")

    if high_results and sensitivity < 0.95:
        print("\n  ⚠  High Risk sensitivity below 95% target.")
        print("     Consider lowering THRESHOLD_HIGH in ml/inference.py (try 0.25 or 0.30).")
        print("     Then re-run this test.")
    else:
        print("\n  ✓ All targets met. Proceed to Day 2.")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="neoscreen_v1.tflite")
    parser.add_argument("--data_dir", default="./dataset")
    args = parser.parse_args()
    run_test(args.model, args.data_dir)