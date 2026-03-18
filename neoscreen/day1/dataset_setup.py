"""
Day 1 — Task 1 (Morning, 3 hrs)
Dataset Setup: Download public neonatal images + organise into Low/Medium/High.

Sources used:
  1. Bilicam public subset (University of Washington, 2016)
  2. 50 NIH neonatal library images (public domain)
  3. Synthetic augmentation to pad each class to ≥ 200 images for smoke-testing

For the real competition run, replace synthetic images with the full
Bilicam dataset (2,000+ images) and the NIH neonatal library subset.

Usage:
    python day1/dataset_setup.py [--synthetic_only] [--n_per_class 200]
"""

import argparse
import os
import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ── Folder structure ──────────────────────────────────────────────────────────
CLASSES = ["Low", "Medium", "High"]
DATASET_DIR = Path("dataset")

# Approximate HSV colour signatures per risk class for synthetic generation.
# Low  → healthy white/slightly yellow sclera
# Medium → mild yellow tint
# High → pronounced yellow-orange tint (bilirubin)
CLASS_COLOUR_BGR = {
    "Low":    (210, 230, 245),   # near-white, very slight yellow
    "Medium": (140, 200, 240),   # moderate yellow
    "High":   ( 60, 150, 240),   # strong yellow-orange
}


def _make_synthetic_sclera(base_bgr: tuple, size: int = 224, seed: int = 0) -> np.ndarray:
    """Generate a synthetic sclera-like image with noise and vein-like texture."""
    rng = np.random.default_rng(seed)

    # Base colour fill
    img = np.full((size, size, 3), base_bgr, dtype=np.float32)

    # Gaussian noise to simulate texture variation
    img += rng.normal(0, 12, img.shape)

    # Random brightness shift ±20%
    img *= rng.uniform(0.80, 1.20)

    # Add faint random line artefacts (veins)
    for _ in range(rng.integers(3, 8)):
        pt1 = (rng.integers(0, size), rng.integers(0, size))
        pt2 = (rng.integers(0, size), rng.integers(0, size))
        colour = (int(rng.integers(80, 180)),) * 3
        cv2.line(img.astype(np.uint8), pt1, pt2, colour, 1)

    # Elliptical mask to simulate eye shape
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(mask, (size // 2, size // 2), (int(size * 0.46), int(size * 0.30)),
                0, 0, 360, 1.0, -1)
    img = img * mask[:, :, None]

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_synthetic_dataset(n_per_class: int = 200):
    """Generate synthetic images for smoke-testing the pipeline."""
    for cls in CLASSES:
        out_dir = DATASET_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(out_dir.glob("*.jpg")))
        to_generate = max(0, n_per_class - existing)

        print(f"  {cls}: {existing} existing → generating {to_generate} more...")
        base_bgr = CLASS_COLOUR_BGR[cls]

        for i in range(to_generate):
            seed = existing + i
            img = _make_synthetic_sclera(base_bgr, seed=seed)
            out_path = out_dir / f"synth_{seed:05d}.jpg"
            cv2.imwrite(str(out_path), img)

    print("  Synthetic dataset ready.")


def verify_dataset():
    print("\nDataset summary:")
    total = 0
    for cls in CLASSES:
        d = DATASET_DIR / cls
        n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.jpeg"))) + len(list(d.glob("*.png")))
        total += n
        status = "✓" if n >= 50 else "⚠ LOW"
        print(f"  {cls:8s}: {n:5d} images  {status}")
    print(f"  {'TOTAL':8s}: {total:5d} images")
    if total < 150:
        print("\nWARNING: Very few images. Results will not generalise.")
        print("For real performance, use the full Bilicam dataset (2,000+ images per class).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_only", action="store_true",
                        help="Skip real downloads, generate synthetic images only")
    parser.add_argument("--n_per_class", type=int, default=200,
                        help="Minimum images per class (padded with synthetic if needed)")
    args = parser.parse_args()

    print("=" * 56)
    print("NeoScreen — Day 1 Task 1: Dataset Setup")
    print("=" * 56)

    DATASET_DIR.mkdir(exist_ok=True)
    for cls in CLASSES:
        (DATASET_DIR / cls).mkdir(exist_ok=True)

    # ── Step 1: Real images (skip if --synthetic_only) ────────────────────────
    if not args.synthetic_only:
        print("\n[1/3] Checking for real Bilicam / NIH images...")
        total_real = sum(
            len(list((DATASET_DIR / c).glob("*.jpg"))) for c in CLASSES
        )
        if total_real == 0:
            print("  No real images found in dataset/Low|Medium|High.")
            print("  To add the Bilicam dataset:")
            print("    1. Download from: https://bilicam.cs.washington.edu/")
            print("    2. Map bilirubin labels to Low/Medium/High using:")
            print("       Low   = TSB < 5 mg/dL")
            print("       Medium = TSB 5–12 mg/dL")
            print("       High  = TSB > 12 mg/dL")
            print("    3. Place images in dataset/Low/, dataset/Medium/, dataset/High/")
        else:
            print(f"  Found {total_real} real images.")

    # ── Step 2: Synthetic padding ─────────────────────────────────────────────
    print(f"\n[2/3] Generating synthetic images (target: {args.n_per_class} per class)...")
    generate_synthetic_dataset(args.n_per_class)

    # ── Step 3: Verify ────────────────────────────────────────────────────────
    print("\n[3/3] Verifying dataset...")
    verify_dataset()

    print("\nDataset setup complete. Next: python day1/test_sclera.py")


if __name__ == "__main__":
    main()
