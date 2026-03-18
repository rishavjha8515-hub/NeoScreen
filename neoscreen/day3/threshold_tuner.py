"""
Day 3 — Task 3
Threshold Tuner: tries all HIGH/LOW threshold combinations and finds the pair
that maximises High Risk sensitivity while keeping specificity >= 55%.
Automatically patches ml/inference.py with the best values found.

Usage:
    python day3/threshold_tuner.py --model neoscreen_v1.tflite --data_dir ./dataset
"""

import argparse
import os
import sys
import re
import glob

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.sclera_detection import detect_sclera, preprocess_for_inference

CLASSES = ["Low", "Medium", "High"]


def load_all(data_dir):
    pairs = []
    for cls in CLASSES:
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in glob.glob(os.path.join(folder, ext)):
                pairs.append((p, cls))
    return pairs


def get_probs(model_path, pairs):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    results = []
    for img_path, true_cls in pairs:
        import cv2
        roi = detect_sclera(img_path)
        if roi is None:
            raw = cv2.imread(img_path)
            if raw is None:
                continue
            roi = cv2.resize(raw, (224, 224))
        tensor = preprocess_for_inference(roi)
        interp.set_tensor(inp[0]["index"], tensor)
        interp.invoke()
        probs = interp.get_tensor(out[0]["index"])[0]
        results.append({"true": true_cls, "probs": probs})
    return results


def evaluate_thresholds(results, thresh_high, thresh_low):
    preds = []
    for r in results:
        p = r["probs"]
        if p[2] >= thresh_high:
            pred = "High"
        elif p[0] >= thresh_low:
            pred = "Low"
        else:
            pred = "Medium"
        preds.append(pred)

    high_true = [r for r in results if r["true"] == "High"]
    high_correct = sum(1 for r, p in zip(results, preds) if r["true"] == "High" and p == "High")
    sensitivity = high_correct / len(high_true) if high_true else 0

    # Specificity: of non-High cases, how many are correctly not flagged High
    non_high = [r for r in results if r["true"] != "High"]
    non_high_correct = sum(1 for r, p in zip(results, preds) if r["true"] != "High" and p != "High")
    specificity = non_high_correct / len(non_high) if non_high else 0

    return sensitivity, specificity


def patch_inference_py(thresh_high, thresh_low):
    path = os.path.join("ml", "inference.py")
    with open(path, "r") as f:
        content = f.read()
    content = re.sub(r"THRESHOLD_HIGH\s*=\s*[\d.]+", f"THRESHOLD_HIGH = {thresh_high}", content)
    content = re.sub(r"THRESHOLD_LOW\s*=\s*[\d.]+",  f"THRESHOLD_LOW = {thresh_low}",  content)
    with open(path, "w") as f:
        f.write(content)
    print(f"  ✓  ml/inference.py patched → HIGH={thresh_high}  LOW={thresh_low}")


def run(model_path, data_dir):
    print("=" * 60)
    print("NeoScreen — Day 3: Threshold Tuner")
    print("=" * 60)

    pairs = load_all(data_dir)
    if not pairs:
        print(f"No images in {data_dir}. Run day1/dataset_setup.py first.")
        sys.exit(1)
    print(f"\nLoaded {len(pairs)} images. Running inference...")
    results = get_probs(model_path, pairs)
    print(f"Got probabilities for {len(results)} images.")

    # Grid search
    high_thresholds = [round(x, 2) for x in np.arange(0.20, 0.50, 0.05)]
    low_thresholds  = [round(x, 2) for x in np.arange(0.50, 0.80, 0.05)]

    print(f"\nTrying {len(high_thresholds) * len(low_thresholds)} threshold combinations...")

    best = None
    best_sens = 0
    rows = []

    for th in high_thresholds:
        for tl in low_thresholds:
            sens, spec = evaluate_thresholds(results, th, tl)
            rows.append((th, tl, sens, spec))
            # Goal: maximise sensitivity, keep specificity >= 0.55
            if sens > best_sens and spec >= 0.55:
                best_sens = sens
                best = (th, tl, sens, spec)

    # Print top 10
    rows.sort(key=lambda x: (-x[2], -x[3]))
    print(f"\n{'HIGH':>6} {'LOW':>6} {'Sens':>7} {'Spec':>7}")
    print("-" * 32)
    for row in rows[:10]:
        marker = " ← BEST" if best and row[0] == best[0] and row[1] == best[1] else ""
        print(f"  {row[0]:>4}   {row[1]:>4}   {row[2]:.3f}   {row[3]:.3f}{marker}")

    print("\n" + "=" * 60)
    if best:
        th, tl, sens, spec = best
        print(f"Best thresholds found:")
        print(f"  THRESHOLD_HIGH = {th}  → Sensitivity: {sens:.1%}")
        print(f"  THRESHOLD_LOW  = {tl}  → Specificity: {spec:.1%}")

        if sens >= 0.95:
            print(f"\n  ✓  Sensitivity {sens:.1%} meets target (>= 95%)")
        else:
            print(f"\n  ⚠  Sensitivity {sens:.1%} below 95% target.")
            print("     Need more High Risk training images.")

        patch_inference_py(th, tl)
    else:
        print("No threshold combination achieved specificity >= 55%.")
        print("Model needs more training data.")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="neoscreen_v1.tflite")
    parser.add_argument("--data_dir", default="./dataset")
    args = parser.parse_args()
    run(args.model, args.data_dir)
