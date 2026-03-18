
import argparse
import json
import os
import shutil
import sys
import glob
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.sclera_detection import detect_sclera, preprocess_for_inference
from ml.inference import classify_risk, THRESHOLD_HIGH, THRESHOLD_LOW


CLASSES = ["Low", "Medium", "High"]
REPORT_DIR = os.path.join("day1", "reports")
MISCLASS_DIR = os.path.join(REPORT_DIR, "misclassified_images")


def load_all_images(data_dir: str) -> list[tuple[str, str]]:
    pairs = []
    for cls in CLASSES:
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in glob.glob(os.path.join(folder, ext)):
                pairs.append((p, cls))
    return pairs


def analyse_misclassifications(results: list[dict]) -> str:
    """
    Identify patterns in misclassifications.
    Returns a plain-text summary.
    """
    lines = []

    # Confusion breakdown
    conf = {tc: {pc: 0 for pc in CLASSES} for tc in CLASSES}
    for r in results:
        conf[r["true"]][r["pred"]] += 1

    lines.append("Confusion Matrix (rows = true, cols = predicted):")
    lines.append(f"{'':12s}" + "".join(f"{c:>8s}" for c in CLASSES))
    for tc in CLASSES:
        row = f"{tc:<12s}" + "".join(f"{conf[tc][pc]:>8d}" for pc in CLASSES)
        lines.append(row)

    lines.append("")

    # Per-class analysis
    for tc in CLASSES:
        total = sum(conf[tc].values())
        correct = conf[tc][tc]
        if total == 0:
            continue
        recall = correct / total
        lines.append(f"{tc} class: {recall:.0%} recall ({correct}/{total})")

        # Most common error
        errors = [(pc, conf[tc][pc]) for pc in CLASSES if pc != tc and conf[tc][pc] > 0]
        errors.sort(key=lambda x: -x[1])
        for pred_cls, cnt in errors:
            lines.append(f"  → Misclassified as {pred_cls}: {cnt}×")

            # Summarise probability ranges for these cases
            subset = [r for r in results if r["true"] == tc and r["pred"] == pred_cls]
            if subset:
                avg_p_high = sum(r["probs"][2] for r in subset) / len(subset)
                avg_p_low = sum(r["probs"][0] for r in subset) / len(subset)
                lines.append(f"    avg P(High)={avg_p_high:.3f}  avg P(Low)={avg_p_low:.3f}")

    lines.append("")

    # Actionable recommendations
    lines.append("Recommendations:")
    high_results = [r for r in results if r["true"] == "High"]
    if high_results:
        missed_high = [r for r in high_results if r["pred"] != "High"]
        if missed_high:
            avg_missed_phigh = sum(r["probs"][2] for r in missed_high) / len(missed_high)
            lines.append(
                f"  {len(missed_high)} High Risk cases missed (avg P(High)={avg_missed_phigh:.3f})."
            )
            if avg_missed_phigh >= 0.25:
                lines.append(
                    f"  → Lower THRESHOLD_HIGH from {THRESHOLD_HIGH} to {avg_missed_phigh - 0.02:.2f}"
                )
            else:
                lines.append("  → Model needs more High Risk training data.")
        else:
            lines.append("  ✓ All High Risk cases correctly detected.")

    return "\n".join(lines)


def run(model_path: str, data_dir: str):
    print("=" * 58)
    print("NeoScreen — Day 1 Task 6: Misclassification Log")
    print("=" * 58)

    os.makedirs(MISCLASS_DIR, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Run training first.")
        sys.exit(1)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    # ── Collect all images ────────────────────────────────────────────────────
    pairs = load_all_images(data_dir)
    if not pairs:
        print(f"No images found in {data_dir}. Run dataset_setup.py first.")
        sys.exit(1)
    print(f"\nEvaluating {len(pairs)} images...")

    results = []
    for img_path, true_cls in pairs:
        roi = detect_sclera(img_path)
        if roi is None:
            raw = cv2.imread(img_path)
            roi = cv2.resize(raw, (224, 224)) if raw is not None else None
        if roi is None:
            continue

        tensor = preprocess_for_inference(roi)
        interpreter.set_tensor(inp[0]["index"], tensor)
        interpreter.invoke()
        probs = interpreter.get_tensor(out[0]["index"])[0]
        pred_cls = classify_risk(probs)

        results.append({
            "path": img_path,
            "true": true_cls,
            "pred": pred_cls,
            "probs": probs.tolist(),
            "correct": pred_cls == true_cls,
        })

    misclassified = [r for r in results if not r["correct"]]
    accuracy = sum(1 for r in results if r["correct"]) / len(results)

    # ── Copy misclassified images ─────────────────────────────────────────────
    print(f"\n{len(misclassified)} misclassified images ({1 - accuracy:.0%} error rate)")
    for r in misclassified:
        fname = os.path.basename(r["path"])
        dest = os.path.join(
            MISCLASS_DIR,
            f"TRUE_{r['true']}__PRED_{r['pred']}__{fname}",
        )
        shutil.copy2(r["path"], dest)

    # ── Write report ──────────────────────────────────────────────────────────
    analysis = analyse_misclassifications(results)

    report_lines = [
        "NeoScreen — Misclassification Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Model:     {model_path}  ({os.path.getsize(model_path) / 1e6:.2f} MB)",
        f"Dataset:   {data_dir}",
        f"Threshold: HIGH ≥ {THRESHOLD_HIGH}  LOW ≥ {THRESHOLD_LOW}",
        "",
        f"Total images:     {len(results)}",
        f"Correct:          {sum(1 for r in results if r['correct'])}",
        f"Misclassified:    {len(misclassified)}",
        f"Overall accuracy: {accuracy:.1%}",
        "",
        "=" * 50,
        "",
        analysis,
        "",
        "=" * 50,
        "",
        "Per-image log (misclassified only):",
        f"{'File':<35} {'True':>7} {'Pred':>7} {'P(L)':>6} {'P(M)':>6} {'P(H)':>6}",
        "-" * 70,
    ]
    for r in sorted(misclassified, key=lambda x: x["true"]):
        p = r["probs"]
        report_lines.append(
            f"{os.path.basename(r['path']):<35} {r['true']:>7} {r['pred']:>7} "
            f"{p[0]:>6.3f} {p[1]:>6.3f} {p[2]:>6.3f}"
        )

    report_text = "\n".join(report_lines)
    report_path = os.path.join(REPORT_DIR, "misclassification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Also save raw JSON for further analysis
    json_path = os.path.join(REPORT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved → {report_path}")
    print(f"JSON data  → {json_path}")
    print(f"Misclassified images → {MISCLASS_DIR}/")
    print()
    print(analysis)
    print("=" * 58)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="neoscreen_v1.tflite")
    parser.add_argument("--data_dir", default="./dataset")
    args = parser.parse_args()
    run(args.model, args.data_dir)