"""
Day 1 — Task 2 (Morning, 1 hr)
Smoke-test the sclera detection pipeline with a synthetic eye image.
No real dataset needed — validates OpenCV pipeline is working.

Usage:
    python day1/test_sclera.py [--image path/to/real_eye.jpg]
"""

import sys
import os
import tempfile
import argparse

import cv2
import numpy as np

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.sclera_detection import (
    apply_clahe,
    apply_white_balance,
    detect_sclera,
    preprocess_for_inference,
)


def make_synthetic_eye(size: int = 480) -> np.ndarray:
    """Create a synthetic eye image with a visible white sclera region."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Dark surround (skin tone approximation)
    img[:] = (60, 90, 120)

    # Sclera — white ellipse, slight yellow tint
    cx, cy = size // 2, size // 2
    cv2.ellipse(img, (cx, cy), (int(size * 0.38), int(size * 0.22)), 0, 0, 360,
                (200, 230, 245), -1)

    # Iris (darker circle in centre)
    cv2.circle(img, (cx, cy), int(size * 0.12), (40, 80, 60), -1)

    # Pupil
    cv2.circle(img, (cx, cy), int(size * 0.06), (5, 5, 5), -1)

    # Subtle vein lines on sclera
    rng = np.random.default_rng(42)
    for _ in range(6):
        offset = lambda: rng.integers(-int(size * 0.3), int(size * 0.3))
        pt1 = (cx + offset(), cy + offset())
        pt2 = (cx + offset(), cy + offset())
        cv2.line(img, pt1, pt2, (160, 180, 200), 1)

    return img


def run_test(image_path: str | None):
    print("=" * 54)
    print("NeoScreen — Day 1 Task 2: Sclera Detection Test")
    print("=" * 54)

    # ── 1. Source image ───────────────────────────────────────────────────────
    if image_path:
        print(f"\n[1/5] Using provided image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ERROR: Cannot read {image_path}")
            sys.exit(1)
        tmp_path = image_path
    else:
        print("\n[1/5] No image provided — generating synthetic eye...")
        img = make_synthetic_eye()
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        cv2.imwrite(tmp_path, img)
        os.close(tmp_fd)
        print(f"  Synthetic eye saved → {tmp_path}")

    print(f"  Image shape: {img.shape}")

    # ── 2. CLAHE ──────────────────────────────────────────────────────────────
    print("\n[2/5] Applying CLAHE (low-light enhancement)...")
    clahe_img = apply_clahe(img)
    assert clahe_img.shape == img.shape, "CLAHE shape mismatch"
    print("  ✓ CLAHE applied")

    # ── 3. White balance ──────────────────────────────────────────────────────
    print("\n[3/5] Applying white balance correction...")
    wb_img = apply_white_balance(clahe_img)
    assert wb_img.shape == img.shape, "White balance shape mismatch"
    print("  ✓ White balance applied")

    # ── 4. Sclera detection ───────────────────────────────────────────────────
    print("\n[4/5] Detecting sclera ROI...")
    roi = detect_sclera(tmp_path)

    if roi is None:
        print("  ✗ Sclera NOT detected.")
        print("    This is expected for very dark / low-contrast images.")
        print("    In the real app the user is prompted to retake the photo.")
        if not image_path:
            os.unlink(tmp_path)
        return False
    else:
        assert roi.shape == (224, 224, 3), f"ROI shape should be (224,224,3), got {roi.shape}"
        cv2.imwrite("day1/sclera_test_output.jpg", roi)
        print(f"  ✓ Sclera detected — 224×224 ROI saved → day1/sclera_test_output.jpg")

    # ── 5. Preprocess for inference ───────────────────────────────────────────
    print("\n[5/5] Preprocessing for inference...")
    tensor = preprocess_for_inference(roi)
    assert tensor.shape == (1, 224, 224, 3), f"Tensor shape wrong: {tensor.shape}"
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0, "Pixel values not in [0, 1]"
    print(f"  ✓ Tensor shape: {tensor.shape}  range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    if not image_path:
        os.unlink(tmp_path)

    print("\n" + "=" * 54)
    print("All checks passed. Sclera pipeline is working correctly.")
    print("=" * 54)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to a real eye JPEG (optional)")
    args = parser.parse_args()
    success = run_test(args.image)
    sys.exit(0 if success else 1)
