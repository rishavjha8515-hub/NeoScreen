"""
Day 2 — Task 11 (Afternoon, 1 hr)
Full end-to-end pipeline test — simulates exactly what the Flutter app does.

Pipeline:
  Synthetic eye image
    → CLAHE + white balance
    → Sclera ROI detection
    → TF Lite inference
    → Risk classification
    → Hindi + English output
    → (optional) PHC SMS alert

Usage:
    python day2/test_end_to_end.py --model neoscreen_v1.tflite
    python day2/test_end_to_end.py --model neoscreen_v1.tflite --send-sms
"""

import argparse
import os
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.sclera_detection import detect_sclera, preprocess_for_inference
from ml.inference import classify_jaundice, classify_risk, get_risk_message


def make_test_image(risk_hint: str = "High") -> str:
    """Generate a synthetic sclera image and save to a temp file."""
    colour_map = {
        "High":   (60,  150, 240),   # yellow-orange → High bilirubin
        "Medium": (140, 200, 240),
        "Low":    (210, 230, 245),
    }
    base = colour_map.get(risk_hint, colour_map["High"])
    size = 320

    img = np.full((size, size, 3), base, dtype=np.float32)
    img += np.random.default_rng(0).normal(0, 10, img.shape)

    mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(mask, (size // 2, size // 2),
                (int(size * 0.42), int(size * 0.28)), 0, 0, 360, 1.0, -1)
    img = np.clip(img * mask[:, :, None], 0, 255).astype(np.uint8)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, img)
    tmp.close()
    return tmp.name


def run(model_path: str, send_sms: bool = False, lang: str = "hi"):
    print("=" * 60)
    print("NeoScreen — Day 2 Task 11: End-to-End Pipeline Test")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"\n✗  Model not found: {model_path}")
        print("   Run Day 1 training first: python ml/train.py")
        sys.exit(1)

    total_start = time.perf_counter()

    # ── Step 1: Image input ───────────────────────────────────────────────────
    print("\n[Step 1] Generating synthetic High-Risk test image...")
    img_path = make_test_image("High")
    print(f"  ✓  Image: {img_path}")

    # ── Step 2: CLAHE + white balance + sclera detection ──────────────────────
    print("\n[Step 2] Sclera detection (CLAHE → white balance → HSV ROI)...")
    t0 = time.perf_counter()
    sclera = detect_sclera(img_path)
    detect_ms = (time.perf_counter() - t0) * 1000

    if sclera is None:
        # Synthetic images may not pass strict HSV — load directly
        raw = cv2.imread(img_path)
        sclera = cv2.resize(raw, (224, 224))
        print(f"  ⚠  HSV detection skipped (synthetic image) — using raw resize")
    else:
        print(f"  ✓  Sclera detected  ({detect_ms:.1f} ms)")

    os.unlink(img_path)

    # ── Step 3: TF Lite inference ─────────────────────────────────────────────
    print("\n[Step 3] TF Lite inference...")
    t0 = time.perf_counter()
    risk, message, probs = classify_jaundice(sclera, model_path=model_path, lang=lang)
    infer_ms = (time.perf_counter() - t0) * 1000

    print(f"  ✓  Inference complete  ({infer_ms:.1f} ms)")
    print(f"     P(Low)={probs[0]:.3f}  P(Medium)={probs[1]:.3f}  P(High)={probs[2]:.3f}")

    # ── Step 4: Risk classification + language output ─────────────────────────
    print("\n[Step 4] Risk classification + language output...")
    msg_hi = get_risk_message(risk, "hi")
    msg_en = get_risk_message(risk, "en")

    print(f"  ✓  Risk:    {risk}")
    print(f"     Hindi:   {msg_hi}")
    print(f"     English: {msg_en}")

    # ── Step 5: PHC alert (High Risk only) ───────────────────────────────────
    print("\n[Step 5] PHC referral check...")
    if risk == "HIGH":
        print(f"  ⚡ HIGH RISK — PHC alert would be triggered")
        if send_sms:
            from ml.referral import send_phc_alert
            from dotenv import load_dotenv
            load_dotenv()
            print("  Sending real SMS...")
            sent = send_phc_alert(
                risk="HIGH",
                baby_age_hrs=36,
                asha_id="TEST-ASHA-001",
                lat=18.5204,
                lon=73.8567,
            )
            if sent:
                print("  ✓  SMS sent — check Twilio console")
        else:
            print("  (Pass --send-sms to actually send Twilio alert)")
    else:
        print(f"  ✓  {risk} risk — no PHC alert needed")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_ms = (time.perf_counter() - total_start) * 1000
    model_mb = os.path.getsize(model_path) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print(f"  Result:         {risk} — {msg_en}")
    print(f"  Total time:     {total_ms:.0f} ms  (target < 30,000 ms)")
    print(f"  Inference only: {infer_ms:.1f} ms  (target < 500 ms)")
    print(f"  Model size:     {model_mb:.2f} MB  (target < 5 MB)")

    targets_met = infer_ms < 500 and model_mb < 5.0
    print(f"\n  {'✓ All targets met — ready for Day 2 APK build' if targets_met else '⚠ Check targets above'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="neoscreen_v1.tflite")
    parser.add_argument("--send-sms", action="store_true")
    parser.add_argument("--lang", default="hi", choices=["hi", "en"])
    args = parser.parse_args()
    run(args.model, args.send_sms, args.lang)
