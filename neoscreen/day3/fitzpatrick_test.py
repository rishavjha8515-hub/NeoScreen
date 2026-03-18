"""
Day 3 — Task 4
Fitzpatrick Skin Tone Test.
Confirms NeoScreen is skin-tone independent by testing sclera detection
on synthetic eyes across all 6 Fitzpatrick phototypes.

The key insight: we analyse the SCLERA (white of eye), not the skin.
Sclera colour is independent of skin melanin. This test validates that.

Usage:
    python day3/fitzpatrick_test.py --model neoscreen_v1.tflite
"""

import argparse
import os
import sys
import tempfile

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.sclera_detection import detect_sclera, preprocess_for_inference
from ml.inference import classify_risk

# Fitzpatrick scale — skin BGR colours (surrounding the eye)
FITZPATRICK_TYPES = [
    (1, "Very fair",        (220, 215, 210)),
    (2, "Fair",             (195, 175, 155)),
    (3, "Medium",           (160, 130, 100)),
    (4, "Olive",            (120, 95,  70)),
    (5, "Brown",            (80,  55,  35)),
    (6, "Dark brown/black", (45,  30,  20)),
]

# Sclera colours for High Risk (yellow) vs Low Risk (white)
SCLERA_HIGH = (60,  150, 240)   # yellow-orange tint (high bilirubin)
SCLERA_LOW  = (210, 230, 245)   # near-white (healthy)


def make_fitzpatrick_eye(skin_bgr, sclera_bgr, size=320):
    img = np.full((size, size, 3), skin_bgr, dtype=np.float32)
    # Sclera ellipse
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(mask, (size//2, size//2), (int(size*0.40), int(size*0.26)), 0, 0, 360, 1.0, -1)
    sclera_layer = np.full((size, size, 3), sclera_bgr, dtype=np.float32)
    img = img * (1 - mask[:,:,None]) + sclera_layer * mask[:,:,None]
    # Iris
    cv2.circle(img.astype(np.uint8), (size//2, size//2), int(size*0.10), (40, 70, 50), -1)
    # Pupil
    cv2.circle(img.astype(np.uint8), (size//2, size//2), int(size*0.05), (5, 5, 5), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def run(model_path):
    print("=" * 65)
    print("NeoScreen — Day 3 Task 4: Fitzpatrick Skin Tone Independence Test")
    print("=" * 65)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}. Train first.")
        sys.exit(1)

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    print(f"\n{'Type':<4} {'Skin':<22} {'Sclera':>8}  {'P(High)':>8}  {'P(Low)':>8}  {'Result':>8}  {'Pass?'}")
    print("-" * 72)

    all_pass = True

    for expected_sclera, sclera_bgr, expected_label in [
        ("HIGH", SCLERA_HIGH, "High"),
        ("LOW",  SCLERA_LOW,  "Low"),
    ]:
        for ftype, fname, skin_bgr in FITZPATRICK_TYPES:
            img = make_fitzpatrick_eye(skin_bgr, sclera_bgr)

            # Save to temp file for detect_sclera
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp.name, img)
            tmp.close()

            roi = detect_sclera(tmp.name)
            os.unlink(tmp.name)

            if roi is None:
                roi = cv2.resize(img, (224, 224))

            tensor = preprocess_for_inference(roi)
            interp.set_tensor(inp[0]["index"], tensor)
            interp.invoke()
            probs = interp.get_tensor(out[0]["index"])[0]
            result = classify_risk(probs)

            passed = result == expected_label
            if not passed:
                all_pass = False

            print(f"  {ftype:<3}  {fname:<22} {expected_sclera:>7}   "
                  f"{probs[2]:>7.3f}   {probs[0]:>7.3f}   {result:>8}   "
                  f"{'✓' if passed else '✗ FAIL'}")

    print("\n" + "=" * 65)
    if all_pass:
        print("✓ All 12 Fitzpatrick tests passed.")
        print("  Model is skin-tone independent — sclera analysis confirmed.")
    else:
        print("✗ Some tests failed.")
        print("  This may indicate the model is using skin colour, not sclera colour.")
        print("  Fix: ensure augmentation includes diverse skin surrounds in training.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="neoscreen_v1.tflite")
    args = parser.parse_args()
    run(args.model)
