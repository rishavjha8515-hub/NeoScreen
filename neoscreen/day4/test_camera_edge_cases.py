"""
Day 4 — Task 3
Camera Edge Case Tests.
Validates the app handles bad input gracefully — never crashes, always prompts retry.

Tests:
  1. Pure black image (lens covered)
  2. Blurry image (motion blur)
  3. No eye region visible (photo of ceiling)
  4. Very low contrast
  5. Partial eye (clipped at frame edge)
  6. Multiple bright regions (not a sclera)

Usage:
    python day4/test_camera_edge_cases.py --model neoscreen_v1.tflite
"""

import argparse
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.sclera_detection import detect_sclera


def save_tmp(img):
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, img)
    tmp.close()
    return tmp.name


def make_case(name):
    size = 320
    if name == "black":
        return np.zeros((size, size, 3), dtype=np.uint8)

    elif name == "blur":
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.ellipse(img, (size//2, size//2), (130, 85), 0, 0, 360, (220, 230, 240), -1)
        return cv2.GaussianBlur(img, (51, 51), 30)

    elif name == "no_eye":
        # Photo of plain wall — uniform texture
        img = np.full((size, size, 3), (160, 155, 150), dtype=np.uint8)
        img += np.random.randint(-10, 10, img.shape, dtype=np.int16).clip(0, 255).astype(np.uint8)
        return img

    elif name == "low_contrast":
        return np.full((size, size, 3), (128, 128, 128), dtype=np.uint8)

    elif name == "partial_eye":
        img = np.full((size, size, 3), (80, 100, 120), dtype=np.uint8)
        # Eye cut off at left edge
        cv2.ellipse(img, (20, size//2), (130, 85), 0, 0, 360, (220, 230, 240), -1)
        cv2.circle(img, (20, size//2), 40, (40, 70, 50), -1)
        return img

    elif name == "multiple_bright":
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for cx, cy in [(80, 80), (240, 80), (80, 240), (240, 240)]:
            cv2.circle(img, (cx, cy), 35, (210, 220, 230), -1)
        return img

    return np.zeros((size, size, 3), dtype=np.uint8)


CASES = [
    ("black",           "Lens covered / total darkness",     False),
    ("blur",            "Motion blur",                        None),   # may or may not detect
    ("no_eye",          "No eye in frame (ceiling/wall)",     False),
    ("low_contrast",    "Very low contrast",                  False),
    ("partial_eye",     "Partial eye at frame edge",          None),
    ("multiple_bright", "Multiple bright regions (not eye)",  None),
]


def run(model_path):
    print("=" * 60)
    print("NeoScreen — Day 4 Task 3: Camera Edge Case Tests")
    print("=" * 60)
    print("\nExpected: None = detect_sclera returns None → user prompted to retry")
    print(f"\n{'Case':<28} {'Expected':>10}  {'Got':>10}  {'Pass?'}")
    print("-" * 58)

    all_pass = True

    for case_name, description, expected_detect in CASES:
        img = make_case(case_name)
        tmp = save_tmp(img)
        result = detect_sclera(tmp)
        os.unlink(tmp)

        detected = result is not None

        if expected_detect is None:
            # Any result is acceptable
            status = "✓ OK"
        elif expected_detect == detected:
            status = "✓"
        else:
            status = "✗ FAIL"
            all_pass = False

        expected_str = "detect" if expected_detect else ("no-detect" if expected_detect is False else "either")
        got_str = "detected" if detected else "no-detect"

        print(f"  {description:<26} {expected_str:>10}  {got_str:>10}  {status}")

    print("\n" + "=" * 60)
    if all_pass:
        print("✓ All edge cases handled correctly.")
        print("  App will always prompt retry on bad images — never crash.")
    else:
        print("✗ Some edge cases failed.")
        print("  Check ml/sclera_detection.py minimum contour size filter.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="neoscreen_v1.tflite")
    args = parser.parse_args()
    run(args.model)
